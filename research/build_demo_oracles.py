#!/usr/bin/env python
"""Verify demo oracle coverage for all UNI_CAA01 pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from unicaa01_demo_oracles import ORACLES, oracle_to_dict  # noqa: E402
from unicaa01_registry import PAIRS_ALL  # noqa: E402

EXPECTED_N = 66


def verify() -> dict[str, object]:
    registry_ids = sorted(p.pair_id for p in PAIRS_ALL)
    oracle_ids = sorted(ORACLES.keys())
    missing = [pid for pid in registry_ids if pid not in ORACLES]
    extra = [pid for pid in oracle_ids if pid not in registry_ids]
    empty = [pid for pid, o in ORACLES.items() if not o.expect_repr.strip()]
    return {
        "n_oracles": len(ORACLES),
        "n_pairs": len(registry_ids),
        "missing": missing,
        "extra": extra,
        "empty_expect_repr": empty,
        "pass": len(missing) == 0 and len(extra) == 0 and len(empty) == 0 and len(ORACLES) == EXPECTED_N,
        "oracles": {pid: oracle_to_dict(o) for pid, o in sorted(ORACLES.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="Fail if coverage incomplete")
    ap.add_argument("--out", type=Path, help="Write coverage JSON")
    args = ap.parse_args()

    report = verify()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "oracles"}, indent=2))

    if args.verify and not report["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
