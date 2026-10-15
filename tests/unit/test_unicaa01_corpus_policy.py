"""Unit tests for UNICAA01 corpus CAST / p15 policy."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from unicaa01_corpus_policy import (  # noqa: E402
    corpus_pair_tier,
    filter_pairs_by_tier,
    load_cast_alpha_0,
)
from unicaa01_registry import get_pair  # noqa: E402


def test_load_cast_alpha_0() -> None:
    assert load_cast_alpha_0() == 0.85


def test_p15_http_retry_is_p15_tier() -> None:
    assert corpus_pair_tier(get_pair("p15_http_retry")) == "p15"


def test_astar_pair_is_hard_tier() -> None:
    assert corpus_pair_tier(get_pair("p22_astar_grid")) == "hard"


def test_filter_p15_excludes_hard() -> None:
    p15 = get_pair("p15_http_retry")
    hard = get_pair("p22_astar_grid")
    out = filter_pairs_by_tier([p15, hard], "p15")
    assert [p.pair_id for p in out] == ["p15_http_retry"]


def test_corpus_aider_timeouts_panic_longer_than_neutral() -> None:
    from unicaa01_corpus_policy import corpus_aider_timeouts

    panic_wall, panic_per, panic_sub = corpus_aider_timeouts("panic")
    neutral_wall, neutral_per, neutral_sub = corpus_aider_timeouts("neutral")
    assert panic_wall > neutral_wall
    assert panic_per > neutral_per
    assert panic_sub > neutral_sub
