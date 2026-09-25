#!/usr/bin/env python3
"""Re-pin the NVMe PLE reader's source guard to this checkout.

The reader registers its hooks only when every SGLang module named in
ssd_stream/src/sglang_ssd_stream/pennyroyal-source.json is byte-identical to
the copy the adapter was reviewed against. After a rebase or edit changes one
of those modules, review the adapter (qwen4.py, graph.py, config.py,
offload.py, backend.py) against the new code, then re-pin, bump the package
version and reinstall the reader with install.sh.

    python tools/ple_nvme/refresh_source_guard.py --check
    python tools/ple_nvme/refresh_source_guard.py --source madlabs/systemone-prod@<sha>
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[1]
GUARD = TOOL_DIR / "ssd_stream" / "src" / "sglang_ssd_stream" / "pennyroyal-source.json"


def module_path(module: str) -> Path:
    base = REPO_ROOT / "python" / Path(*module.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    raise FileNotFoundError(f"guarded module {module} is not in {REPO_ROOT / 'python'}")


def module_hashes(modules) -> dict[str, str]:
    return {
        module: hashlib.sha256(module_path(module).read_bytes()).hexdigest()
        for module in modules
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report stale entries")
    mode.add_argument("--source", help="label for the reviewed source, e.g. a commit")
    args = parser.parse_args()

    guard = json.loads(GUARD.read_text())
    actual = module_hashes(guard["modules"])
    stale = sorted(m for m, digest in actual.items() if guard["modules"][m] != digest)
    if args.check:
        for module in stale:
            print(f"stale: {module}", file=sys.stderr)
        return 1 if stale else 0

    GUARD.write_text(
        json.dumps({"source": args.source, "modules": actual}, indent=2) + "\n"
    )
    print(f"re-pinned {len(actual)} modules ({len(stale)} changed) to {args.source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
