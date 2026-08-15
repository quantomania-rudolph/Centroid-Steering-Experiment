#!/usr/bin/env python
"""Verify dashboard data: multiple experiments, all layers present."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "dashboard" / "data.json"
OUT = ROOT / "experiments" / "out"


def main() -> int:
    if not DATA.is_file():
        print("FAIL: missing", DATA)
        return 1

    runs = json.loads(DATA.read_text(encoding="utf-8"))
    if len(runs) < 2:
        print(f"FAIL: expected >=2 experiments, got {len(runs)}")
        return 1

    names = sorted(r.get("experiment", "?") for r in runs)
    print(f"OK: {len(runs)} experiments: {names}")

    for r in runs:
        layers = r.get("layers") or []
        nums = [row["layer"] for row in layers]
        expected = list(range(8, 27))
        if nums != expected:
            print(f"FAIL: {r.get('experiment')} layers {nums} != {expected}")
            return 1
        if not all("j_lens_top" in row and "logit_lens_top" in row for row in layers):
            print(f"FAIL: {r.get('experiment')} missing token data")
            return 1
        print(
            f"  {r['experiment']}: L{nums[0]}-L{nums[-1]} "
            f"({len(layers)} rows), J-lens top L8={layers[0]['j_lens_top'][0]['token']!r}"
        )

    # Cross-experiment distinguishability: L8 top token should differ or prompts differ
    a, b = runs[0], runs[1]
    if a.get("prompt") == b.get("prompt"):
        print("FAIL: prompts identical")
        return 1
    print("OK: prompts differ between experiments")

    print("\nPROOF PASS — open with: python experiments/serve_dashboard.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
