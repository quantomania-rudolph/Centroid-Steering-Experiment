"""GATE chaos-E: layer curve shape sign-off."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "artifacts" / "gates" / "chaos_checker_E.json"


@pytest.mark.skipif(os.environ.get("CHAOS_LIVE") != "1", reason="set CHAOS_LIVE=1 after prove_chaos_live.py")
def test_gate_chaos_E():
    assert PATH.is_file(), f"missing {PATH}"
    data = json.loads(PATH.read_text(encoding="utf-8"))
    assert data["checker"] == "E"
    assert data["signed_off"] is True
    assert data.get("used_force_alarm") is False
    cliff = (data.get("verifiability") or {}).get("cliff_clean")
    assert cliff is not None and 8 <= int(cliff) <= 26
