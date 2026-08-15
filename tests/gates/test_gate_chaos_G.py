"""GATE chaos-G: sparse residual sign-off (may be unsigned if dual-flag never fired)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "artifacts" / "gates" / "chaos_checker_G.json"


@pytest.mark.skipif(os.environ.get("CHAOS_LIVE") != "1", reason="set CHAOS_LIVE=1 after prove_chaos_live.py")
def test_gate_chaos_G():
    assert PATH.is_file(), f"missing {PATH}"
    data = json.loads(PATH.read_text(encoding="utf-8"))
    assert data["checker"] == "G"
    assert data.get("used_force_alarm") is False
    assert "speed" in data and "usefulness" in data and "verifiability" in data
    # Natural dual-flag may not fire every run; artifact must still exist.
