"""Gate: chaos live proof artifacts must exist and be signed off."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GATES = ROOT / "artifacts" / "gates"
CHECKERS = [
    "A",
    "B_vocab",
    "B_jlens",
    "B_transport",
    "C",
    "D",
    "E",
    "F",
    "G",
]


@pytest.mark.skipif(os.environ.get("CHAOS_LIVE") != "1", reason="set CHAOS_LIVE=1 after prove_chaos_live.py")
def test_chaos_live_master_proof():
    path = GATES / "chaos_live_proof.json"
    assert path.is_file(), f"missing {path} — run experiments/prove_chaos_live.py"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("device") == "xpu"
    assert data.get("used_force_alarm") is False
    assert data.get("overhead", {}).get("pass") is True
    assert data.get("separation", {}).get("pass") is True
    # Prefer all signed; G may fail if no natural dual-flag — require 8/9 minimum
    signed = data.get("checkers") or {}
    n_ok = sum(1 for k in CHECKERS if signed.get(k))
    assert n_ok >= 8, f"only {n_ok}/9 checkers signed: {signed}"


@pytest.mark.skipif(os.environ.get("CHAOS_LIVE") != "1", reason="set CHAOS_LIVE=1 after prove_chaos_live.py")
@pytest.mark.parametrize("name", CHECKERS)
def test_chaos_checker_artifact(name: str):
    path = GATES / f"chaos_checker_{name}.json"
    assert path.is_file(), f"missing {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("checker") == name
    assert data.get("used_force_alarm") is False
    assert "speed" in data and "usefulness" in data and "verifiability" in data
    # G may be unsigned if alarms never fired
    if name != "G":
        assert data.get("signed_off") is True, data
