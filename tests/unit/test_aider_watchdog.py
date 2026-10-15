"""Tests for Aider watchdog stall logic."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from run_aider_agent import _should_kill_on_invocation_stall  # noqa: E402


def test_no_stall_kill_during_first_call() -> None:
    assert not _should_kill_on_invocation_stall(
        inv=0,
        stall_s=600.0,
        per_invocation_timeout_s=360.0,
        generation_in_flight=False,
    )


def test_no_stall_kill_while_generation_in_flight() -> None:
    assert not _should_kill_on_invocation_stall(
        inv=1,
        stall_s=600.0,
        per_invocation_timeout_s=360.0,
        generation_in_flight=True,
    )


def test_stall_kill_between_invocations() -> None:
    assert _should_kill_on_invocation_stall(
        inv=2,
        stall_s=400.0,
        per_invocation_timeout_s=360.0,
        generation_in_flight=False,
    )
