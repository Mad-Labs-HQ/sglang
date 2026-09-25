import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GUARD = (
    REPO
    / "tools"
    / "ple_nvme"
    / "ssd_stream"
    / "src"
    / "sglang_ssd_stream"
    / "pennyroyal-source.json"
)


def test_source_guard_matches_every_hooked_publication_module():
    guard = json.loads(GUARD.read_text())
    assert guard["source"].startswith("madlabs/systemone-prod@")
    assert "sglang.srt.models.qwen4_exp" in guard["modules"]

    for module, expected in guard["modules"].items():
        path = REPO / "python" / Path(*module.split("."))
        if path.with_suffix(".py").is_file():
            path = path.with_suffix(".py")
        else:
            path = path / "__init__.py"
        assert path.is_file(), module
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, (
            f"{module} changed; review the NVMe PLE adapter against it, then run "
            "tools/ple_nvme/refresh_source_guard.py and reinstall the reader"
        )
