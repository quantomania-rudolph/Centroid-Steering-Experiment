"""v4 quality suite — p06 + solid suites for p12/p18/p26."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from calm_curator_demo_check import run_quality_suite  # noqa: E402
from unicaa01_quality_suite import configured_pair_ids  # noqa: E402
from validate_narrative_purity import validate_narrative_purity  # noqa: E402

FX_NEG = ROOT / "tests" / "fixtures" / "oracle_negative"
FX_POS = ROOT / "tests" / "fixtures" / "oracle_positive"

GOOD_PARSER = """
def parse_json(s: str):
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError("array only")
    inner = s[1:-1].strip()
    if not inner:
        return []
    return [int(x.strip()) for x in inner.split(",")]

def demo():
    return parse_json("[1,2]")
"""


def test_configured_pairs_include_next_three():
    ids = set(configured_pair_ids())
    assert {"p06_json_parse", "p12_fibonacci", "p18_trie_prefix", "p26_rle_compress"} <= ids
    assert {"p07_csv_agg", "p17_deque_queue", "p25_edit_distance", "p62_binary_search", "p66_rot13"} <= ids


def test_thin_parser_fails_quality_suite():
    result = run_quality_suite(
        main_py=(FX_NEG / "p06_thin_parser.py").read_text(encoding="utf-8"),
        pair_id="p06_json_parse",
    )
    assert result.pass_ is False


def test_good_parser_passes_quality_suite():
    result = run_quality_suite(main_py=GOOD_PARSER, pair_id="p06_json_parse")
    assert result.pass_ is True


def test_p12_hardcoded_fails_suite():
    result = run_quality_suite(
        main_py=(FX_NEG / "p12_hardcoded_fib.py").read_text(encoding="utf-8"),
        pair_id="p12_fibonacci",
    )
    assert result.pass_ is False


def test_p12_good_passes_suite():
    result = run_quality_suite(
        main_py=(FX_POS / "p12_good_fib.py").read_text(encoding="utf-8"),
        pair_id="p12_fibonacci",
    )
    assert result.pass_ is True
    assert "9 cases" in result.detail


def test_p18_always_true_fails_suite():
    result = run_quality_suite(
        main_py=(FX_NEG / "p18_always_true_prefix.py").read_text(encoding="utf-8"),
        pair_id="p18_trie_prefix",
    )
    assert result.pass_ is False


def test_p18_good_passes_suite():
    result = run_quality_suite(
        main_py=(FX_POS / "p18_good_trie.py").read_text(encoding="utf-8"),
        pair_id="p18_trie_prefix",
    )
    assert result.pass_ is True


def test_p26_hardcoded_fails_suite():
    result = run_quality_suite(
        main_py=(FX_NEG / "p26_hardcoded_rle.py").read_text(encoding="utf-8"),
        pair_id="p26_rle_compress",
    )
    assert result.pass_ is False


def test_p26_good_passes_suite():
    result = run_quality_suite(
        main_py=(FX_POS / "p26_good_rle.py").read_text(encoding="utf-8"),
        pair_id="p26_rle_compress",
    )
    assert result.pass_ is True


def test_narrative_continuation_grammar_rejects_implement():
    full = "Fixing `Implement minimal JSON parser\nunicaa01_p06_panic/main.py\n```python\npass\n```"
    results = validate_narrative_purity(full_text=full, prefix_text="Fixing `", pair_id="p06_json_parse")
    grammar = next(r for r in results if r.gate_id == "NARR_continuation_grammar")
    assert grammar.pass_ is False


def test_narrative_continuation_grammar_accepts_natural():
    full = "Fixing `broken bracket validation` in hand-rolled parser\nunicaa01_p06_panic/main.py\n```python\npass\n```"
    results = validate_narrative_purity(full_text=full, prefix_text="Fixing `", pair_id="p06_json_parse")
    grammar = next(r for r in results if r.gate_id == "NARR_continuation_grammar")
    ticks = next(r for r in results if r.gate_id == "NARR_balanced_backticks")
    assert grammar.pass_ is True
    assert ticks.pass_ is True
