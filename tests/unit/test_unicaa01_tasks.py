"""Tests for UNICAA01 corpus task registry (wave 1 + wave 2)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from unicaa01_batch2_tasks import PAIRS_BATCH2  # noqa: E402
from unicaa01_registry import PAIRS_ALL, get_pair, neutral_capture_specs, pairs_for_wave  # noqa: E402
from unicaa01_tasks import PAIRS as PAIRS_WAVE1  # noqa: E402


def test_wave1_sixteen_pairs():
    assert len(PAIRS_WAVE1) == 16


def test_wave2_fifty_pairs():
    assert len(PAIRS_BATCH2) == 50
    ids = [p.pair_id for p in PAIRS_BATCH2]
    assert len(ids) == len(set(ids))
    assert ids[0].startswith("p17_")


def test_all_pairs_sixty_six_unique():
    assert len(PAIRS_ALL) == 66
    assert len({p.pair_id for p in PAIRS_ALL}) == 66
    assert len({p.panic_slug for p in PAIRS_ALL}) == 66


def test_panic_starters_have_not_implemented():
    for p in PAIRS_BATCH2:
        assert "NotImplementedError" in p.panic_starter, p.pair_id


def test_neutral_capture_specs_batch2():
    specs = neutral_capture_specs(["p17_deque_queue", "p66_rot13"])
    assert len(specs) == 2


def test_get_pair_wave2():
    p = get_pair("p17_deque_queue")
    assert p.wave == 2
    assert p.domain == "data_structures"


def test_pairs_for_wave():
    assert len(pairs_for_wave(1)) == 16
    assert len(pairs_for_wave(2)) == 50
    assert len(pairs_for_wave(None)) == 66
