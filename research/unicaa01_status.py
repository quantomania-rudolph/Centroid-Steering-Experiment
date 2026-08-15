#!/usr/bin/env python
"""Quick UNICAA01 corpus progress snapshot (no long waits)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from unicaa01_post_batch import quality_report  # noqa: E402
from unicaa01_registry import PAIRS_ALL, pairs_for_wave  # noqa: E402

CORPUS_OUT = ROOT / "experiments" / "unicaa01_corpus" / "out"


def main() -> int:
    files = sorted(CORPUS_OUT.glob("*_latest.json"))
    report = quality_report()
    done_pairs = {r.get("pair_id") for r in report.get("rows") or [] if r.get("pair_id")}
    wave2_ids = {p.pair_id for p in pairs_for_wave(2)}
    wave2_done = sum(1 for pid in done_pairs if pid in wave2_ids)
    snap = {
        "n_corpus_files": len(files),
        "expected_runs": len(PAIRS_ALL) * 2,
        "wave1_pairs": 16,
        "wave2_pairs": 50,
        "wave2_files_done": wave2_done,
        "n_panic_ok": report.get("n_panic_ok"),
        "n_neutral_ok": report.get("n_neutral_ok"),
        "n_dense_panic": report.get("n_dense_panic"),
        "pairs_with_any_file": len(done_pairs),
        "failed_panic": [
            r["pair_id"]
            for r in report.get("rows") or []
            if r.get("arm") == "panic" and r.get("dense_token_ids") and not r.get("eligible")
        ],
        "latest_files": [f.name for f in sorted(files, key=lambda p: p.stat().st_mtime)[-4:]],
    }
    print(json.dumps(snap, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
