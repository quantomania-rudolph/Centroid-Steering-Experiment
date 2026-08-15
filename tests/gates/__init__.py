"""Shared gate reporting utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "gates"


def save_gate_report(gate: str, payload: dict[str, Any]) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ARTIFACTS / f"{gate}_{ts}.json"
    payload = {
        **payload,
        "gate": gate,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    # Also write latest symlink-style copy
    latest = ARTIFACTS / f"{gate}_latest.json"
    latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def summarize_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    overall = all(bool(c.get("pass")) for c in checks)
    return {
        "overall_pass": overall,
        "n_pass": sum(1 for c in checks if c.get("pass")),
        "n_fail": sum(1 for c in checks if not c.get("pass")),
        "checks": checks,
    }
