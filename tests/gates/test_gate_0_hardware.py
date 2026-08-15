"""GATE-0: Hardware readiness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jlens_qwen.hardware import evaluate_gate0, save_hardware_report
from tests.gates import save_gate_report, summarize_checks

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "artifacts" / "gates" / "hardware_report.json"


@pytest.mark.gate0
def test_gate0_hardware():
    report = save_hardware_report(REPORT_PATH)
    result = evaluate_gate0(report)
    summary = summarize_checks(result["checks"])
    path = save_gate_report(
        "GATE-0",
        {**summary, "hardware_report_path": str(REPORT_PATH)},
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {path}")
    failed = [c for c in summary["checks"] if not c["pass"]]
    assert summary["overall_pass"], f"GATE-0 FAIL: {failed}"
