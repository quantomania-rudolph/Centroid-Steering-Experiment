"""Gate: clean-tail FPR proof artifact."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "artifacts" / "gates" / "caa_clean_tail_gate.json"


@pytest.mark.skipif(
    os.environ.get("CAA_CLEAN_TAIL_GATE") != "1",
    reason="set CAA_CLEAN_TAIL_GATE=1 after prove_clean_tail.py",
)
def test_clean_tail_gate_passes() -> None:
    assert PATH.is_file(), f"missing {PATH}"
    data = json.loads(PATH.read_text(encoding="utf-8"))
    assert data.get("pass"), json.dumps(data, indent=2)
    checks = {c["id"]: c for c in data.get("checks") or []}
    assert checks["CT-4"]["pass"], "p61 must pass dedicated cap"
    assert checks["CT-3"]["pass"], "per-task FPR cap"
