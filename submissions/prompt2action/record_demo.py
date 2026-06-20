from __future__ import annotations

import os
import sys
from pathlib import Path


SUBMISSION_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SUBMISSION_DIR / "demo_video"


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    if __package__ in {None, ""}:
        sys.path.insert(0, str(SUBMISSION_DIR))
        from simulator import main as run_demo
    else:
        from submissions.prompt2action.simulator import main as run_demo

    return run_demo(
        [
            "--control-mode", "kinematic",
            "--no-llm",
            "--headless",
            "--batch-file", str(SUBMISSION_DIR / "video_demo_commands.txt"),
            "--record-video",
            "--output-dir", str(OUTPUT_DIR),
            "--fps", "24",
            "--width", "640",
            "--height", "480",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
