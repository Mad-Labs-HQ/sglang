"""
The server's API keys must not appear in the startup log line, /server_info or
the launch command; every server start printed them before.

Pure CPU; no server, no CUDA.
Run with:
    python3 -m pytest test/registered/unit/entrypoints/test_server_secret_redaction.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import unittest

from sglang.srt.sampling.watermarking.core import (
    redact_watermark_command_line,
    redact_watermark_secrets,
)

KEY = "sk-live-0123456789"
ADMIN = "adm-9876543210"


class TestServerSecretRedaction(unittest.TestCase):
    def test_server_args_and_server_info(self):
        args = {
            "api_key": KEY,
            "admin_api_key": ADMIN,
            "watermark_key": "wk",
            "port": 8000,
        }
        out = redact_watermark_secrets({"server_args": args, "internal_states": [{}]})
        self.assertNotIn(KEY, repr(out))
        self.assertNotIn(ADMIN, repr(out))
        self.assertEqual(out["server_args"]["watermark_key"], "<redacted>")
        self.assertEqual(out["server_args"]["port"], 8000)

    def test_launch_command_both_flag_forms(self):
        argv = ["--model-path", "m", "--api-key", KEY, f"--admin-api-key={ADMIN}"]
        line = redact_watermark_command_line(argv)
        self.assertEqual(
            line, "--model-path m --api-key <redacted> --admin-api-key=<redacted>"
        )


if __name__ == "__main__":
    unittest.main()
