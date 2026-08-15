"""Standardized proof gate JSON writer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_proof_gate(
    *,
    gate_id: str,
    pass_: bool,
    evidence: dict[str, Any],
    out: Path,
) -> dict[str, Any]:
    payload = {
        "gate_id": gate_id,
        "pass": pass_,
        "evidence": evidence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
