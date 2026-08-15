#!/usr/bin/env python
"""Proof gate: clean-tail FPR caps (p61 / per-task / corpus)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.clean_tail_audit import (  # noqa: E402
    audit_clean_tail,
    clean_tail_gate_passes,
    load_audit,
)
from jlens_qwen.caa.gate_io import write_gate

CALIB_DIR = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "calibration"
DEFAULT_META = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "metadata.json"
GATES = ROOT / "artifacts" / "gates"
AUDIT_JSON = CALIB_DIR / "clean_tail_audit.json"
OUT_GATE = GATES / "caa_clean_tail_gate.json"
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def _sabotage_p61_must_improve(audit: dict) -> dict:
    """CT-S1: p61 FPR must be at or below cap (not merely excluded)."""
    p61 = audit.get("p61")
    cap = float((audit.get("thresholds") or {}).get("p61_fpr_cap") or 0.08)
    if not p61:
        return {"id": "CT-S1", "pass": False, "detail": "p61 missing"}
    fpr = float(p61.get("fpr_gamma_high") or 1.0)
    return {
        "id": "CT-S1",
        "pass": fpr <= cap + 1e-4,
        "detail": f"p61_fpr_gamma_high={fpr:.4f} cap={cap}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, default=DEFAULT_META)
    ap.add_argument("--skip-audit", action="store_true")
    args = ap.parse_args()

    if not args.skip_audit:
        subprocess.call(
            [
                str(PY),
                str(ROOT / "research" / "audit_clean_tail.py"),
                "--metadata",
                str(args.metadata),
            ],
            cwd=str(ROOT),
        )

    audit = (
        load_audit(AUDIT_JSON)
        if AUDIT_JSON.is_file()
        else audit_clean_tail(CALIB_DIR, args.metadata)
    )
    gate_pass, checks = clean_tail_gate_passes(audit)
    sabotage = [_sabotage_p61_must_improve(audit)]

    payload = write_gate(
        OUT_GATE,
        gate="caa_clean_tail",
        pass_=gate_pass and sabotage[0]["pass"],
        checks=checks,
        sabotage=sabotage,
        extra={
            "n_clean_episodes": audit.get("n_clean_episodes"),
            "corpus_fpr_gamma_high": audit.get("corpus_fpr_gamma_high"),
            "p61": audit.get("p61"),
            "tau_excluded_noisy": audit.get("tau_excluded_noisy"),
        },
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
