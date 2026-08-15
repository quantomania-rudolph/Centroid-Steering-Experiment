#!/usr/bin/env python
"""Convenience wrapper: python scripts/run_gates.py --through 7 [--from-artifacts]."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.gates.run_all import main

if __name__ == "__main__":
    raise SystemExit(main())
