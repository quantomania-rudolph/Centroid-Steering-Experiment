"""GATE chaos-A: distribution-shape checker sign-off artifact."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "artifacts" / "gates" / "chaos_checker_A.json"


@pytest.mark.skipif(os.environ.get("CHAOS_LIVE") != "1", reason="set CHAOS_LIVE=1 after prove_chaos_live.py")
def test_gate_chaos_A():
    assert PATH.is_file(), f"missing {PATH} — run experiments/prove_chaos_live.py"
    data = json.loads(PATH.read_text(encoding="utf-8"))
    assert data["checker"] == "A"
    assert data["signed_off"] is True
    assert data.get("used_force_alarm") is False
    assert "speed" in data and "usefulness" in data and "verifiability" in data
    assert data["usefulness"].get("pass") is True
