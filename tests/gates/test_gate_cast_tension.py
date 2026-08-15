"""Gate: CAST tension proof artifact."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "artifacts" / "gates" / "caa_cast_tension_gate.json"


@pytest.mark.skipif(
    os.environ.get("CAA_TENSION_GATE") != "1",
    reason="set CAA_TENSION_GATE=1 after prove_cast_tension.py",
)
def test_cast_tension_gate_passes() -> None:
    assert PATH.is_file(), f"missing {PATH}"
    data = json.loads(PATH.read_text(encoding="utf-8"))
    assert data.get("pass"), json.dumps(data, indent=2)
