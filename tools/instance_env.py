#!/usr/bin/env python3
"""Pick Flash-Next's per-instance memory settings from the machine.

Every box runs the same engine line. What differs per box is where the
47.68 GiB PLE (per-layer embedding) table lives and how much host memory and
disk the caches get, and that follows from the machine: its RAM, its GPU and
whether it has local NVMe. This reads the machine and prints the settings as
`export` lines for a launcher to eval; the reasons go to stderr.

    eval "$(python3 tools/instance_env.py --overlay /data/flash-next-ple \\
                                        --l3-dir /mnt/nixl-l3)"

A setting already in the environment wins (PENNY_PLE_BACKEND=auto asks for a
choice). It is still checked, and one the machine cannot hold is a warning.
Exit status 1: no placement of the table leaves a usable host cache, or the
NVMe overlay it needs has not been prepared yet.

Settings:
  PENNY_PLE_BACKEND     ram: the table is pinned in host RAM
                        (--ple-offload-embedding). nvme: the reader in
                        tools/ple_nvme streams rows from a prepared overlay on
                        local NVMe, and host RAM holds only page cache. RAM
                        wins whenever it fits beside a usable host cache:
                        decode is faster and nothing depends on a disk.
  PENNY_PLE_NVME_MODEL  the prepared overlay, with nvme.
  HICACHE_SIZE          the host cache in GB (KV, Mamba and QSA shares).
  NIXL_STORAGE_BASE     the NIXL disk tier's directory, when one qualifies.

Rules, in GB (1e9 bytes, as --hicache-size counts):
  reserve       max(40, RAM / 4): page cache, NFS cache, JIT builds, tools
  overhead      6: the engine's processes outside the caches
  room          RAM - reserve - overhead - (the table, in ram mode)
  minimum       max(16, VRAM / 3): the host KV share must exceed the device
                KV pool (the engine asserts it); ~34 on a 96 GB card
  HICACHE_SIZE  min(75, room), rounded down to 5 GB so that a little MemTotal
                drift does not change the NIXL namespace, which hashes it. 75
                is mad-llm-1's qualified size (paused-agent reuse, 2026-10-26).
  NIXL tier     a mounted directory on local NVMe, writable, at least twice
                HICACHE_SIZE: smaller, it holds little the host cache does not.

These reproduce the two boxes that serve Flash-Next: Andrew's (100.3 GB RAM,
local NVMe: nvme, 50, NIXL on /mnt/nixl-l3) and mad-llm-1 (218 GB, a network
disk only: ram, 75, no NIXL tier).
"""

import argparse
import json
import math
import os
import shlex
import subprocess
import sys

TABLE_GB = 51.2  # 47.68 GiB of fp8 rows: Flash-Next's PLE table
CACHE_CAP_GB = 75
OVERHEAD_GB = 6
NETWORK_FS = {"nfs", "nfs4", "cifs", "smb3", "9p", "ceph", "glusterfs", "lustre"}


def reserve_gb(mem_gb: float) -> float:
    return max(40.0, mem_gb / 4)


def minimum_cache_gb(vram_gb) -> float:
    return max(16.0, vram_gb / 3) if vram_gb else 32.0


def cache_size_gb(room_gb: float) -> int:
    return int(math.floor(min(CACHE_CAP_GB, room_gb) / 5) * 5)


# ------------------------------------------------------------------ machine
def mem_total_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024 / 1e9
    raise RuntimeError("no MemTotal in /proc/meminfo")


def vram_gb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.split()
        return int(out[0]) * 2**20 / 1e9
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def _existing(path: str) -> str:
    path = os.path.realpath(path)
    while not os.path.exists(path):
        path = os.path.dirname(path)
    return path


def _mount_of(path: str):
    """(mount point, fstype, major:minor) of the filesystem holding path."""
    path, best = _existing(path), None
    with open("/proc/self/mountinfo") as f:
        for line in f:
            fields = line.split()
            mount, dev = fields[4], fields[2]
            fstype = fields[fields.index("-") + 1]
            inside = path == mount or path.startswith(mount.rstrip("/") + "/")
            if inside and (best is None or len(mount) > len(best[0])):
                best = (mount, fstype, dev)
    return best


def _block_is_nvme(dev: str, depth: int = 0) -> bool:
    """Whether block device major:minor is backed by NVMe, through loop and dm."""
    sysdir = f"/sys/dev/block/{dev}"
    if depth > 4 or not os.path.exists(sysdir):
        return False
    if "/nvme/" in os.path.realpath(sysdir):
        return True
    try:
        with open(f"{sysdir}/loop/backing_file") as f:
            backing = _mount_of(f.read().strip())
        return backing is not None and _block_is_nvme(backing[2], depth + 1)
    except OSError:
        pass
    slaves = f"{sysdir}/slaves"
    if os.path.isdir(slaves) and os.listdir(slaves):
        return all(
            _block_is_nvme(open(f"{slaves}/{s}/dev").read().strip(), depth + 1)
            for s in os.listdir(slaves)
        )
    return False


def describe_dir(path: str) -> dict:
    """Facts about a candidate directory for the overlay or the NIXL tier."""
    mount, fstype, dev = _mount_of(path)
    stat = os.statvfs(_existing(path))
    local_nvme = fstype not in NETWORK_FS and _block_is_nvme(dev)
    return {
        "path": path,
        "mount": mount,
        "is_mount": os.path.realpath(path) == mount,
        "fstype": fstype,
        "local_nvme": local_nvme,
        "free_gb": stat.f_bavail * stat.f_frsize / 1e9,
        "size_gb": stat.f_blocks * stat.f_frsize / 1e9,
        "writable": os.access(_existing(path), os.W_OK),
        "prepared": os.path.isfile(os.path.join(path, "ssd-stream.json")),
    }


def overlay_table_gb(path: str):
    try:
        with open(os.path.join(path, "ssd-stream.json")) as f:
            return sum(t["bytes"] for t in json.load(f)["tables"]) / 1e9
    except (OSError, KeyError, TypeError, ValueError):
        return None


# ------------------------------------------------------------------ the plan
def plan(mem_gb, vram, overlay, l3, env, table_gb=TABLE_GB):
    """Settings for one machine. Pure: overlay and l3 are describe_dir()
    results (or None), env the variables already set. Returns
    (settings, notes, ok)."""
    notes, ok = [], True
    reserve, minimum = reserve_gb(mem_gb), minimum_cache_gb(vram)
    room = {
        backend: mem_gb - reserve - OVERHEAD_GB - (table_gb if backend == "ram" else 0)
        for backend in ("ram", "nvme")
    }
    notes.append(
        f"RAM {mem_gb:.1f} GB, VRAM {vram or 0:.1f} GB: reserve {reserve:.0f}, "
        f"host cache minimum {minimum:.0f}; room for the cache "
        f"{room['ram']:.0f} GB with the table in RAM, {room['nvme']:.0f} on NVMe"
    )

    nvme_ok = bool(overlay and overlay["local_nvme"])
    if overlay is None:
        nvme_why = "no overlay directory given (--overlay)"
    elif not overlay["local_nvme"]:
        nvme_why = f"{overlay['path']} is on {overlay['fstype']}, not local NVMe"
    elif not overlay["prepared"] and overlay["free_gb"] < table_gb * 1.05:
        nvme_ok = False
        nvme_why = f"{overlay['path']}: {overlay['free_gb']:.0f} GB free, the table needs {table_gb:.0f}"
    else:
        nvme_why = ""

    backend = env.get("PENNY_PLE_BACKEND", "auto")
    if backend == "auto":
        if room["ram"] >= minimum:
            backend = "ram"
            notes.append("ram: the table fits beside a usable host cache")
        elif nvme_ok:
            backend = "nvme"
            notes.append(f"nvme: in RAM the table would leave {room['ram']:.0f} GB "
                         f"of cache, under the {minimum:.0f} GB minimum")
        else:
            notes.append(f"ERROR: in RAM the table leaves {room['ram']:.0f} GB of "
                         f"cache (minimum {minimum:.0f}), and the NVMe reader is "
                         f"out: {nvme_why}")
            return {}, notes, False
    elif backend == "ram" and room["ram"] < minimum:
        notes.append(f"WARNING: PENNY_PLE_BACKEND=ram leaves {room['ram']:.0f} GB "
                     f"of host cache, under the {minimum:.0f} GB minimum")
    elif backend == "nvme" and not nvme_ok:
        notes.append(f"WARNING: PENNY_PLE_BACKEND=nvme, but {nvme_why}")

    settings = {"PENNY_PLE_BACKEND": backend}
    if backend == "nvme" and overlay:
        settings["PENNY_PLE_NVME_MODEL"] = overlay["path"]
        if not overlay["prepared"]:
            notes.append(f"ERROR: prepare the overlay first: python tools/ple_nvme/"
                         f"prepare_ple_nvme.py --source <checkpoint> --output {overlay['path']}")
            ok = False

    cache = cache_size_gb(room[backend])
    if "HICACHE_SIZE" in env:
        cache = float(env["HICACHE_SIZE"])
        if cache > room[backend]:
            notes.append(f"WARNING: HICACHE_SIZE={env['HICACHE_SIZE']} exceeds "
                         f"the {room[backend]:.0f} GB of room")
    elif cache < minimum:
        notes.append(f"ERROR: only {cache} GB of host cache fits, under the "
                     f"{minimum:.0f} GB minimum; set HICACHE_SIZE to override")
        ok = False
    settings["HICACHE_SIZE"] = env.get("HICACHE_SIZE", str(cache))

    if "NIXL_STORAGE_BASE" in env:
        settings["NIXL_STORAGE_BASE"] = env["NIXL_STORAGE_BASE"]
    elif l3 is None:
        notes.append("no NIXL tier: no directory given (--l3-dir)")
    else:
        why = (
            "not a mount point" if not l3["is_mount"]
            else f"on {l3['fstype']}, not local NVMe" if not l3["local_nvme"]
            else "not writable" if not l3["writable"]
            else f"{l3['size_gb']:.0f} GB, under twice the host cache" if l3["size_gb"] < 2 * cache
            else ""
        )
        if why:
            notes.append(f"no NIXL tier: {l3['path']} is {why}")
        else:
            settings["NIXL_STORAGE_BASE"] = l3["path"]
            notes.append(f"NIXL tier on {l3['path']} ({l3['size_gb']:.0f} GB)")
    return settings, notes, ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--overlay", default=os.environ.get("PENNY_PLE_NVME_MODEL"),
                    help="where the NVMe overlay is, or would be prepared")
    ap.add_argument("--l3-dir", default=os.environ.get("NIXL_STORAGE_BASE"),
                    help="candidate directory for the NIXL disk tier")
    ap.add_argument("--json", action="store_true", help="print JSON, not exports")
    args = ap.parse_args(argv)

    overlay = describe_dir(args.overlay) if args.overlay else None
    l3 = describe_dir(args.l3_dir) if args.l3_dir else None
    table = (overlay_table_gb(args.overlay) if overlay and overlay["prepared"] else None) or TABLE_GB
    env = {k: v for k, v in os.environ.items()
           if k in ("PENNY_PLE_BACKEND", "HICACHE_SIZE", "NIXL_STORAGE_BASE")}
    settings, notes, ok = plan(mem_total_gb(), vram_gb(), overlay, l3, env, table)

    for note in notes:
        print(f"instance_env: {note}", file=sys.stderr)
    if args.json:
        print(json.dumps({"settings": settings, "notes": notes, "ok": ok}, indent=2))
    else:
        for key, value in settings.items():
            print(f"export {key}={shlex.quote(str(value))}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
