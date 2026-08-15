"""Pytest for healed p65 interval merge."""

from __future__ import annotations

import main


def test_demo_merges_overlap() -> None:
    assert main.demo() == [(1, 6)]


def test_non_overlapping_unchanged() -> None:
    assert main.merge_intervals([(1, 2), (3, 4)]) == [(1, 2), (3, 4)]


def test_merge_three_sorted_intervals() -> None:
    assert main.merge_intervals([(1, 2), (2, 3), (4, 5)]) == [(1, 3), (4, 5)]


def test_merge_empty_list() -> None:
    assert main.merge_intervals([]) == []
