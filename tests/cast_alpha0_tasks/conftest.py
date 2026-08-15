"""Load healed panic main.py from PANIC_TASK_DIR for cast_alpha0 task tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_task_dir = os.environ.get("PANIC_TASK_DIR")
if _task_dir:
    sys.path.insert(0, str(Path(_task_dir).resolve()))
