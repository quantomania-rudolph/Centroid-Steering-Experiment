"""KV live gate stamping for natural capture manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_KV_GATE = Path("artifacts/gates/kv_equivalence_live_gate.json")


def read_global_kv_live_gate(gate_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(gate_path or DEFAULT_KV_GATE)
    if not path.is_file():
        return {"pass": False, "detail": f"missing {path}"}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "pass": bool(data.get("pass")),
        "timestamp": data.get("timestamp"),
        "environment": data.get("environment") or {},
        "path": str(path),
    }


def kv_live_gate_pass(gate_path: Path | str | None = None) -> bool:
    return bool(read_global_kv_live_gate(gate_path)["pass"])
