"""Admitted calm tail invariant checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CALM_TAILS = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "calm_tails"


def test_pilot_meta_invariants():
    if not CALM_TAILS.is_dir():
        pytest.skip("no calm tails")
    metas = list(CALM_TAILS.glob("*.meta.json"))
    if not metas:
        pytest.skip("no meta files")
    for meta_path in metas:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        gate = meta.get("gate") or {}
        checks = {c["id"]: c.get("pass") for c in gate.get("checks") or []}
        if "RUN_demo_oracle" in checks:
            assert checks["RUN_demo_oracle"] is True
