#!/usr/bin/env python
"""Offline chaos vs û_cond tension audit (Phase 0)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.tension_audit import (  # noqa: E402
    audit_corpus,
    tension_gate_passes,
    write_audit_tsv,
)

CALIB_DIR = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "calibration"
DEFAULT_META = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "metadata.json"
OUT_JSON = CALIB_DIR / "tension_audit.json"
OUT_TSV = CALIB_DIR / "tension_audit.tsv"
DECISION_PATH = CALIB_DIR / "tension_decision.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="CAST chaos vs û_cond tension audit")
    ap.add_argument("--metadata", type=Path, default=DEFAULT_META)
    ap.add_argument("--calib-dir", type=Path, default=CALIB_DIR)
    ap.add_argument("--include-thin", action="store_true")
    args = ap.parse_args()

    audit = audit_corpus(
        args.calib_dir,
        args.metadata,
        thick_only=not args.include_thin,
    )
    gate_pass, checks = tension_gate_passes(audit)
    audit["gate_checks"] = checks
    audit["pass"] = gate_pass

    decision = {
        "retrain_recommended": bool(audit.get("retrain_recommended")),
        "reason": (
            "systematic C+/G- or many low-gamma thick episodes"
            if audit.get("retrain_recommended")
            else "tension within caps — label tighten may suffice"
        ),
        "corpus_c_plus_g_minus_rate": audit.get("corpus_c_plus_g_minus_rate"),
        "low_gamma_thick_episodes": audit.get("low_gamma_thick_episodes"),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    write_audit_tsv(audit, OUT_TSV)
    DECISION_PATH.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"pass": gate_pass, "decision": decision, "checks": checks}, indent=2))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
