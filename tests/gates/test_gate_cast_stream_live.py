"""Gate: live CAST shadow stream proof artifact."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "artifacts" / "gates" / "cast_stream_live_gate.json"


@pytest.mark.skipif(os.environ.get("CAA_CAST_LIVE") != "1", reason="set CAA_CAST_LIVE=1 after prove_cast_stream_live.py")
def test_cast_stream_live_gate_passes() -> None:
    assert PATH.is_file(), f"missing {PATH} — run research/prove_cast_stream_live.py"
    data = json.loads(PATH.read_text(encoding="utf-8"))
    assert data.get("pass"), json.dumps(data, indent=2)
    assert any(s.get("pass") for s in (data.get("sabotage") or [])), "sabotage required"
