"""Stress oracle runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

from calm_curator_demo_check import run_demo_stress  # noqa: E402


def test_stress_skipped_for_untagged_pair():
    main_py = "def demo():\n    return 1\n"
    r = run_demo_stress(main_py=main_py, pair_id="p06_json_parse")
    assert r.pass_ is True
    assert "skipped" in r.detail.lower()


def test_p01_lock_fixture_runs_stress_gate():
    fx = ROOT / "tests" / "fixtures" / "oracle_negative" / "p01_no_lock.py"
    r = run_demo_stress(main_py=fx.read_text(encoding="utf-8"), pair_id="p01_priority_queue")
    assert r.gate_id == "RUN_demo_stress"
