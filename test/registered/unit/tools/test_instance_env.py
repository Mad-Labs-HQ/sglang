"""Unit tests for tools/instance_env.py, the per-instance PLE and cache plan."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

_PATH = Path(__file__).resolve().parents[4] / "tools" / "instance_env.py"
_spec = importlib.util.spec_from_file_location("instance_env", _PATH)
instance_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(instance_env)
plan = instance_env.plan

VRAM_96 = 102.6  # RTX PRO 6000: 97,887 MiB


def _dir(path, prepared=False, local_nvme=True, is_mount=True, size_gb=422.0,
         free_gb=180.0, fstype="ext4", writable=True):
    return {"path": path, "mount": path, "is_mount": is_mount, "fstype": fstype,
            "local_nvme": local_nvme, "free_gb": free_gb, "size_gb": size_gb,
            "writable": writable, "prepared": prepared}


class TestInstanceEnvPlan(unittest.TestCase):
    def test_andrews_box_streams_the_table_from_nvme(self):
        """100.3 GB of RAM: the table in RAM would leave 3 GB of host cache."""
        settings, _, ok = plan(
            100.3, VRAM_96, _dir("/data/ple", prepared=True), _dir("/mnt/nixl-l3"), {}
        )
        self.assertTrue(ok)
        self.assertEqual(settings, {
            "PENNY_PLE_BACKEND": "nvme",
            "PENNY_PLE_NVME_MODEL": "/data/ple",
            "HICACHE_SIZE": "50",
            "NIXL_STORAGE_BASE": "/mnt/nixl-l3",
        })

    def test_mad_llm_1_keeps_the_table_in_ram(self):
        """218 GB of RAM and only a network disk: RAM mode, the 75 GB cap."""
        settings, _, ok = plan(218.0, VRAM_96, None, None, {})
        self.assertTrue(ok)
        self.assertEqual(settings, {"PENNY_PLE_BACKEND": "ram", "HICACHE_SIZE": "75"})

    def test_ram_wins_when_both_fit(self):
        settings, _, _ = plan(218.0, VRAM_96, _dir("/data/ple", prepared=True), None, {})
        self.assertEqual(settings["PENNY_PLE_BACKEND"], "ram")

    def test_no_placement_is_an_error(self):
        for overlay in (None, _dir("/nfs/ple", prepared=True, local_nvme=False, fstype="nfs4"),
                        _dir("/data/ple", free_gb=20.0)):
            with self.subTest(overlay=overlay and overlay["path"]):
                settings, notes, ok = plan(64.0, VRAM_96, overlay, None, {})
                self.assertFalse(ok)
                self.assertEqual(settings, {})
                self.assertTrue(notes[-1].startswith("ERROR"))

    def test_unprepared_overlay_names_the_preparer(self):
        settings, notes, ok = plan(100.3, VRAM_96, _dir("/data/ple"), None, {})
        self.assertFalse(ok)
        self.assertEqual(settings["PENNY_PLE_BACKEND"], "nvme")
        self.assertIn("prepare_ple_nvme.py", "\n".join(notes))

    def test_environment_wins_but_is_checked(self):
        settings, notes, ok = plan(
            100.3, VRAM_96, None, None, {"PENNY_PLE_BACKEND": "ram"}
        )
        self.assertEqual(settings["PENNY_PLE_BACKEND"], "ram")
        self.assertFalse(ok)  # 3 GB of room: no usable host cache
        self.assertIn("WARNING: PENNY_PLE_BACKEND=ram", "\n".join(notes))

        settings, notes, ok = plan(
            218.0, VRAM_96, None, None, {"HICACHE_SIZE": "120", "PENNY_PLE_BACKEND": "auto"}
        )
        self.assertTrue(ok)
        self.assertEqual(settings["HICACHE_SIZE"], "120")
        self.assertIn("WARNING: HICACHE_SIZE=120 exceeds", "\n".join(notes))

    def test_nixl_tier_needs_a_local_nvme_mount_twice_the_cache(self):
        for name, l3 in (
            ("not a mount point", _dir("/data/l3", is_mount=False)),
            ("on nfs4", _dir("/nfs/l3", local_nvme=False, fstype="nfs4")),
            ("not writable", _dir("/mnt/l3", writable=False)),
            ("under twice the host cache", _dir("/mnt/l3", size_gb=90.0)),
        ):
            with self.subTest(name):
                settings, notes, _ = plan(
                    100.3, VRAM_96, _dir("/data/ple", prepared=True), l3, {}
                )
                self.assertNotIn("NIXL_STORAGE_BASE", settings)
                self.assertIn(name, "\n".join(notes))

    def test_cache_size_rounds_down_to_5_gb_under_the_cap(self):
        self.assertEqual(instance_env.cache_size_gb(54.3), 50)
        self.assertEqual(instance_env.cache_size_gb(54.99), 50)
        self.assertEqual(instance_env.cache_size_gb(115.0), 75)

    def test_describe_dir_reads_a_real_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            facts = instance_env.describe_dir(os.path.join(tmp, "not-yet"))
        self.assertFalse(facts["prepared"])
        self.assertGreater(facts["size_gb"], 0)
        self.assertIsInstance(facts["local_nvme"], bool)


if __name__ == "__main__":
    unittest.main()
