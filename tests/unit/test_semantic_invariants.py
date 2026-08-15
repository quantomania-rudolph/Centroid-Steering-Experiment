"""Semantic invariant specs — 66/66 coverage."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from calm_curator_demo_check import check_invariants  # noqa: E402
from unicaa01_oracle_invariants import all_invariant_pair_ids, verify_invariant_coverage  # noqa: E402


def test_66_invariant_coverage():
    report = verify_invariant_coverage()
    assert report["pass"] is True
    assert report["n_pairs"] == 66


@pytest.mark.parametrize("pair_id", ["p06_json_parse", "p11_palindrome", "p18_trie_prefix", "p66_rot13"])
def test_tier_a_spot_check(pair_id: str):
    main_py = 'def demo():\n    return 1\n'
    r = check_invariants(main_py=main_py, pair_id=pair_id)
    assert r.pass_ is True


def test_p01_no_lock_fixture():
    fx = ROOT / "tests" / "fixtures" / "oracle_negative" / "p01_no_lock.py"
    r = check_invariants(main_py=fx.read_text(encoding="utf-8"), pair_id="p01_priority_queue")
    assert r.pass_ is False


def test_p06_import_json_fixture():
    fx = ROOT / "tests" / "fixtures" / "oracle_negative" / "p06_import_json.py"
    r = check_invariants(main_py=fx.read_text(encoding="utf-8"), pair_id="p06_json_parse")
    assert r.pass_ is False
