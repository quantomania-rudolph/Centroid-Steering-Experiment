"""Tests for cast_oneoff_task contract."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from cast_oneoff_task import TASKS, get_task, panic_message, restore_starter  # noqa: E402


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_starter_has_not_implemented_error(task_id: str) -> None:
    task = get_task(task_id)
    assert "NotImplementedError" in task.starter


@pytest.mark.parametrize("task_id", sorted(TASKS))
def test_panic_message_uses_qwen_code_path(task_id: str) -> None:
    task = get_task(task_id)
    msg = panic_message(task)
    assert task.listing_path in msg
    assert "NotImplementedError" in msg


def test_restore_starter_writes_broken_main() -> None:
    path = restore_starter(get_task("panic"))
    assert "raise NotImplementedError" in path.read_text(encoding="utf-8")


def test_cast_retry_suffix() -> None:
    assert "CAST retry" in panic_message(get_task("b02"), cast_retry=True)
