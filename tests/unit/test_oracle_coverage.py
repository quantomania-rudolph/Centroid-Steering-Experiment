"""Oracle coverage — 66 oracles, brief pair_ids subset of oracle keys."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from build_demo_oracles import verify  # noqa: E402
from build_curator_briefs import build_all_briefs  # noqa: E402
from unicaa01_demo_oracles import ORACLES  # noqa: E402


def test_66_oracles():
    report = verify()
    assert report["pass"] is True
    assert report["n_oracles"] == 66


def test_brief_pair_ids_subset_of_oracles():
    briefs = build_all_briefs()
    pair_ids = {b.pair_id for b in briefs}
    missing = pair_ids - set(ORACLES.keys())
    assert not missing, f"missing oracles: {missing}"
