#!/usr/bin/env python
"""Probe hardware and write hardware_report.json (GATE-0)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.hardware import evaluate_gate0, save_hardware_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "gates" / "hardware_report.json",
    )
    args = parser.parse_args()
    report = save_hardware_report(args.out)
    result = evaluate_gate0(report)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
