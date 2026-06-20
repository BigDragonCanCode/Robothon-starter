from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from simulator import main
else:
    from submissions.prompt2action.simulator import main


if __name__ == "__main__":
    sys.exit(main())
