"""Strict aider listing + hardened narrative gates."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from validate_aider_listing import validate_aider_listing  # noqa: E402
from validate_narrative_purity import validate_narrative_purity  # noqa: E402


def _gate_map(results):
    return {r.gate_id: r for r in results}


def test_path_outside_fence_passes():
    text = (
        "Fixing `broken parser`.\n"
        "unicaa01_p06_panic/main.py\n"
        "```python\n"
        "def parse_json(s):\n    return []\n"
        "```"
    )
    gates = _gate_map(validate_aider_listing(full_text=text, task_dir="unicaa01_p06_panic"))
    assert gates["FMT_aider_path_outside_fence"].pass_ is True
    assert gates["FMT_no_path_comment_in_fence"].pass_ is True


def test_path_only_inside_fence_fails():
    text = (
        "Fixing `the Trie class`.\n"
        "```python\n"
        "# unicaa01_p18_panic/main.py\n"
        "class Trie:\n    pass\n"
        "```"
    )
    gates = _gate_map(validate_aider_listing(full_text=text, task_dir="unicaa01_p18_panic"))
    assert gates["FMT_aider_path_outside_fence"].pass_ is False
    assert gates["FMT_no_path_comment_in_fence"].pass_ is False


def test_midline_implement_fails_grammar():
    prefix = "Fix: update `fib` function to use iterative approach and remove `"
    full = prefix + "the stub. Implement iterative Fibonacci below.\nunicaa01_p12_panic/main.py\n```python\npass\n```"
    gates = _gate_map(validate_narrative_purity(full_text=full, prefix_text=prefix, pair_id="p12_fibonacci"))
    assert gates["NARR_continuation_grammar"].pass_ is False
    assert "Implement" in gates["NARR_continuation_grammar"].detail


def test_lowercase_implement_fails_grammar():
    prefix = "Fix: update `fib` function to use iterative approach and remove `"
    full = (
        prefix
        + "stub` and implement iterative calculation.\n"
        "unicaa01_p12_panic/main.py\n```python\npass\n```"
    )
    gates = _gate_map(validate_narrative_purity(full_text=full, prefix_text=prefix, pair_id="p12_fibonacci"))
    assert gates["NARR_continuation_grammar"].pass_ is False


def test_unclosed_prefix_tick_fails_balanced():
    prefix = "Fixing `"
    full = prefix + "broken int-array parser without closing tick\nunicaa01_p06_panic/main.py\n```python\npass\n```"
    gates = _gate_map(validate_narrative_purity(full_text=full, prefix_text=prefix, pair_id="p06_json_parse"))
    assert gates["NARR_balanced_backticks"].pass_ is False


def test_closed_tick_and_clean_prose_pass():
    prefix = "Fixing `"
    full = (
        prefix
        + "broken int-array parser` with bracket checks.\n"
        "unicaa01_p06_panic/main.py\n"
        "```python\ndef demo():\n    return [1, 2]\n```"
    )
    narr = _gate_map(validate_narrative_purity(full_text=full, prefix_text=prefix, pair_id="p06_json_parse"))
    listing = _gate_map(validate_aider_listing(full_text=full, task_dir="unicaa01_p06_panic"))
    assert narr["NARR_continuation_grammar"].pass_ is True
    assert narr["NARR_balanced_backticks"].pass_ is True
    assert listing["FMT_aider_path_outside_fence"].pass_ is True
    assert listing["FMT_no_path_comment_in_fence"].pass_ is True


def test_legacy_v4_p12_prose_would_fail():
    """Frozen v4 failure mode — mid-line Implement + unclosed tick."""
    prefix = "Fix: update `fib` function to use iterative approach and remove `"
    text = (
        prefix
        + "the stub. Implement iterative Fibonacci below.\n\n"
        "unicaa01_p12_panic/main.py\n```python\ndef fib(n): return 55\n```"
    )
    narr = _gate_map(validate_narrative_purity(full_text=text, prefix_text=prefix, pair_id="p12_fibonacci"))
    assert narr["NARR_continuation_grammar"].pass_ is False
    assert narr["NARR_balanced_backticks"].pass_ is False


def test_legacy_v4_p18_listing_would_fail():
    """Frozen v4 failure mode — path only as # comment inside fence."""
    text = (
        "Fixing `the Trie class by implementing insert and starts_with methods.\n"
        "```python\n"
        "# unicaa01_p18_panic/main.py\n"
        "class Trie:\n    pass\n"
        "```"
    )
    listing = _gate_map(validate_aider_listing(full_text=text, task_dir="unicaa01_p18_panic"))
    assert listing["FMT_aider_path_outside_fence"].pass_ is False
    assert listing["FMT_no_path_comment_in_fence"].pass_ is False


def test_legacy_v4_p26_listing_would_fail():
    text = (
        "Fixing `the RLE encode function, then update `demo()`.\n"
        "```python\n"
        "# unicaa01_p26_panic/main.py\n"
        "def rle_encode(s): return 'a3b2c1'\n"
        "```"
    )
    listing = _gate_map(validate_aider_listing(full_text=text, task_dir="unicaa01_p26_panic"))
    assert listing["FMT_aider_path_outside_fence"].pass_ is False
