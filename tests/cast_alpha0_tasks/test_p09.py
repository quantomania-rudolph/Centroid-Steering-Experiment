"""Pytest for healed p09 pool (PANIC_TASK_DIR must point at task dir)."""

from __future__ import annotations

import main


def test_demo_sorted_parallel_map() -> None:
    assert main.demo() == [1, 2, 3]


def test_run_pool_maps_values() -> None:
    assert main.run_pool(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]


def test_run_pool_empty_list() -> None:
    assert main.run_pool(lambda x: x, []) == []


def test_run_pool_identity_order() -> None:
    assert main.run_pool(lambda x: x, [3, 1, 2]) == [3, 1, 2]


def test_run_pool_workers_execute() -> None:
    seen: list[int] = []

    def track(x: int) -> int:
        seen.append(x)
        return x

    result = main.run_pool(track, [1, 2])
    assert result == [1, 2]
    assert sorted(seen) == [1, 2]
