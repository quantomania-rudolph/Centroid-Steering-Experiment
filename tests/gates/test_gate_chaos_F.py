"""GATE chaos-F: J-lens vs logit-lens disagreement sign-off."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "artifacts" / "gates" / "chaos_checker_F.json"


@pytest.mark.skipif(os.environ.get("CHAOS_LIVE") != "1", reason="set CHAOS_LIVE=1 after prove_chaos_live.py")
def test_gate_chaos_F():
    assert PATH.is_file(), f"missing {PATH}"
    data = json.loads(PATH.read_text(encoding="utf-8"))
    assert data["checker"] == "F"
    assert data["signed_off"] is True
    assert data.get("used_force_alarm") is False
