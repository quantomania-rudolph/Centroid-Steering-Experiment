"""Shared gate JSON writer for CAA proof pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_gate(
    path: Path | str,
    *,
    gate: str,
    pass_: bool,
    checks: list[dict[str, Any]],
    sabotage: list[dict[str, Any]] | None = None,
    environment: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "gate": gate,
        "version": 1,
        "timestamp": utc_now_iso(),
        "pass": bool(pass_),
        "checks": checks,
        "sabotage": sabotage or [],
    }
    if environment:
        payload["environment"] = environment
    if extra:
        payload.update(extra)
    # Sabotage mandatory: gate cannot pass without ≥1 sabotage check that passed.
    sab = payload["sabotage"]
    if pass_ and sab:
        if not any(s.get("pass") for s in sab):
            payload["pass"] = False
            payload["notes"] = ["sabotage checks required but none passed"]
    elif pass_ and not sab:
        payload["pass"] = False
        payload["notes"] = ["sabotage checks mandatory — empty sabotage list"]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
