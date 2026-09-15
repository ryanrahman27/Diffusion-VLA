"""Entry point for HIL rollout conversion (see examples/piper_real/)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/piper_real/convert_piperx_rollouts_to_lerobot.py"
)

if __name__ == "__main__":
    sys.argv[0] = str(_SCRIPT)
    runpy.run_path(str(_SCRIPT), run_name="__main__")
