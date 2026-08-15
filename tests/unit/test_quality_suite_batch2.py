"""Suite coverage for the five new pairs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from calm_curator_demo_check import run_quality_suite  # noqa: E402
from unicaa01_quality_suite import configured_pair_ids  # noqa: E402

FX_NEG = ROOT / "tests" / "fixtures" / "oracle_negative"
FX_POS = ROOT / "tests" / "fixtures" / "oracle_positive"

NEW_PAIRS = (
    "p07_csv_agg",
    "p17_deque_queue",
    "p25_edit_distance",
    "p62_binary_search",
    "p66_rot13",
)


@pytest.mark.parametrize(
    ("pair_id", "neg", "pos"),
    [
        ("p07_csv_agg", "p07_hardcoded_sum.py", "p07_good_csv.py"),
        ("p17_deque_queue", "p17_hardcoded_demo.py", "p17_good_deque.py"),
        ("p25_edit_distance", "p25_hardcoded_edit.py", "p25_good_edit.py"),
        ("p62_binary_search", "p62_hardcoded_bisect.py", "p62_good_bisect.py"),
        ("p66_rot13", "p66_identity_rot13.py", "p66_good_rot13.py"),
    ],
)
def test_new_suite_rejects_hack_accepts_good(pair_id: str, neg: str, pos: str):
    bad = run_quality_suite(main_py=(FX_NEG / neg).read_text(encoding="utf-8"), pair_id=pair_id)
    good = run_quality_suite(main_py=(FX_POS / pos).read_text(encoding="utf-8"), pair_id=pair_id)
    assert bad.pass_ is False, bad.detail
    assert good.pass_ is True, good.detail


def test_new_pairs_configured():
    assert set(NEW_PAIRS) <= set(configured_pair_ids())
