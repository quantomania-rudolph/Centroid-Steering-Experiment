#!/usr/bin/env python
"""Offline clean-tail FPR audit (per-task + corpus)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.clean_tail_audit import (  # noqa: E402
    audit_clean_tail,
    clean_tail_gate_passes,
    write_clean_tail_tsv,
)

CALIB_DIR = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "calibration"
DEFAULT_META = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "metadata.json"
OUT_JSON = CALIB_DIR / "clean_tail_audit.json"
OUT_TSV = CALIB_DIR / "clean_tail_audit.tsv"


def main() -> int:
    ap = argparse.ArgumentParser(description="CAST clean-tail per-task FPR audit")
    ap.add_argument("--metadata", type=Path, default=DEFAULT_META)
    ap.add_argument("--calib-dir", type=Path, default=CALIB_DIR)
    args = ap.parse_args()

    audit = audit_clean_tail(args.calib_dir, args.metadata)
    gate_pass, checks = clean_tail_gate_passes(audit)
    audit["gate_checks"] = checks
    audit["pass"] = gate_pass

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    write_clean_tail_tsv(audit, OUT_TSV)

    print(json.dumps({"pass": gate_pass, "checks": checks, "p61": audit.get("p61")}, indent=2))
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
