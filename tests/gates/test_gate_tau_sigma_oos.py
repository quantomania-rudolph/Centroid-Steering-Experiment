"""Gate: CAST τ/σ calibration v2 with OOS episode proof."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "artifacts" / "gates" / "caa_tau_sigma_calibration_gate.json"


@pytest.mark.skipif(
    os.environ.get("CAA_CALIB_LIVE") != "1",
    reason="set CAA_CALIB_LIVE=1 after prove_tau_sigma_oos.py",
)
def test_tau_sigma_oos_gate_v2_passes() -> None:
    assert PATH.is_file(), f"missing {PATH} — run research/prove_tau_sigma_oos.py"
    data = json.loads(PATH.read_text(encoding="utf-8"))
    assert int(data.get("version") or 0) >= 2, "expected gate version >= 2"
    assert data.get("pass"), json.dumps(data, indent=2)
    assert any(s.get("pass") for s in (data.get("sabotage") or [])), "sabotage required"
    verify = data.get("verify") or {}
    assert verify.get("n_verify_episodes_pass") == 3
