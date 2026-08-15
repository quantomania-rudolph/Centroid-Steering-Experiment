#!/usr/bin/env python
"""CAST chaos vs u_cond tension proof gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.caa.gate_io import write_gate
from jlens_qwen.caa.tension_audit import audit_corpus, load_audit, sabotage_shuffled_ucond_on_corpus, tension_gate_passes

CALIB_DIR = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "calibration"
DEFAULT_META = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "metadata.json"
GATES = ROOT / "artifacts" / "gates"
AUDIT_JSON = CALIB_DIR / "tension_audit.json"
OUT_GATE = GATES / "caa_cast_tension_gate.json"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
TAU_GATE = GATES / "caa_tau_sigma_calibration_gate.json"
CLEAN_TAIL_GATE = GATES / "caa_clean_tail_gate.json"
STREAM_GATE = GATES / "cast_stream_live_gate.json"
INJECT_GATE = GATES / "cast_inject_live_gate.json"


def _load_gate(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, default=DEFAULT_META)
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--offline-only", action="store_true")
    args = ap.parse_args()

    if not args.skip_audit:
        subprocess.call(
            [
                str(PY),
                str(ROOT / "research" / "audit_cast_tension.py"),
                "--metadata",
                str(args.metadata),
            ],
            cwd=str(ROOT),
        )

    audit = (
        load_audit(AUDIT_JSON)
        if AUDIT_JSON.is_file()
        else audit_corpus(CALIB_DIR, args.metadata)
    )
    offline_pass, offline_checks = tension_gate_passes(audit)

    checks = list(offline_checks)
    sabotage = [sabotage_shuffled_ucond_on_corpus(CALIB_DIR, args.metadata)]
    checks.append(
        {
            "id": "TEN-S1",
            "pass": sabotage[0]["pass"],
            "detail": sabotage[0]["detail"],
        }
    )
    tau_gate = _load_gate(TAU_GATE)
    checks.append(
        {
            "id": "TAU-SIGMA",
            "pass": bool(tau_gate and tau_gate.get("pass")),
            "detail": str(TAU_GATE),
        }
    )
    clean_tail_gate = _load_gate(CLEAN_TAIL_GATE)
    checks.append(
        {
            "id": "CLEAN-TAIL",
            "pass": bool(clean_tail_gate and clean_tail_gate.get("pass")),
            "detail": str(CLEAN_TAIL_GATE),
        }
    )

    stream_gate = _load_gate(STREAM_GATE)
    inject_gate = _load_gate(INJECT_GATE)
    if not args.offline_only:
        checks.append(
            {
                "id": "LIVE-STREAM",
                "pass": bool(stream_gate and stream_gate.get("pass")),
                "detail": "stream",
            }
        )
        checks.append(
            {
                "id": "LIVE-INJECT",
                "pass": bool(inject_gate and inject_gate.get("pass")),
                "detail": "inject",
            }
        )
        alpha0_gate = _load_gate(GATES / "caa_alpha0_live_gate.json")
        checks.append(
            {
                "id": "LIVE-ALPHA0",
                "pass": bool(alpha0_gate and alpha0_gate.get("pass")),
                "detail": "alpha0",
            }
        )
        for chk in (stream_gate or {}).get("checks") or []:
            if chk.get("id") in ("B4", "B5"):
                checks.append(chk)

    sabotage_pass = sabotage[0]["pass"]
    if stream_gate:
        sabotage_pass = sabotage_pass or any(
            s.get("pass") for s in (stream_gate.get("sabotage") or [])
        )
    if tau_gate:
        sabotage_pass = sabotage_pass or any(
            s.get("pass") for s in (tau_gate.get("sabotage") or [])
        )
    checks.append({"id": "SABOTAGE", "pass": sabotage_pass, "detail": "sabotage"})

    gate_pass = (
        offline_pass
        and bool(tau_gate and tau_gate.get("pass"))
        and bool(clean_tail_gate and clean_tail_gate.get("pass"))
        and sabotage_pass
    )
    if not args.offline_only:
        live_checks = [c for c in checks if str(c.get("id", "")).startswith("LIVE")]
        gate_pass = gate_pass and all(c["pass"] for c in live_checks)

    payload = write_gate(
        OUT_GATE,
        gate="caa_cast_tension",
        pass_=gate_pass,
        checks=checks,
        sabotage=sabotage,
        extra={
            "audit_summary": {
                "corpus_c_plus_g_minus_rate": audit.get("corpus_c_plus_g_minus_rate"),
                "low_gamma_thick_episodes": audit.get("low_gamma_thick_episodes"),
                "retrain_recommended": audit.get("retrain_recommended"),
            },
        },
    )
    print(json.dumps(payload, indent=2))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
