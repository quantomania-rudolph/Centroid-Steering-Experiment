#!/usr/bin/env python
"""After each UNICAA01 batch: quality report, miner scan, curation pack export."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.caa.unicaa01_corpus import CORPUS_GLOB, CORPUS_OUT, CURATION_PACKS, ensure_dirs  # noqa: E402
from prefix_divergence_tracer import trace_invocation  # noqa: E402

PY = ROOT / ".venv" / "Scripts" / "python.exe"
SCAN_REPORT = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus" / "scan_report.json"


def quality_report() -> dict[str, Any]:
    ensure_dirs()
    rows: list[dict[str, Any]] = []
    for path in sorted(CORPUS_OUT.glob("*_latest.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rows.append({"path": str(path), "ok": False, "error": "json_decode"})
            continue
        arm = data.get("unicaa01_arm", "?")
        invs = data.get("invocations") or []
        if not invs:
            rows.append({"path": str(path), "ok": False, "error": "no_invocations"})
            continue
        inv = invs[0]
        chaos = inv.get("chaos") or {}
        token_ids = chaos.get("token_ids") or []
        ct = int(inv.get("completion_tokens") or 0)
        dense = isinstance(token_ids, list) and ct > 0 and len(token_ids) == ct
        trace = trace_invocation(
            inv,
            experiment=str(data.get("experiment") or ""),
            task_slug=str(data.get("task_slug") or ""),
            experiment_data=data,
            use_tokenizer=False,
        )
        row = {
            "path": str(path),
            "experiment": data.get("experiment"),
            "pair_id": data.get("unicaa01_pair_id"),
            "arm": arm,
            "ok": dense and bool(data.get("session_path")),
            "dense_token_ids": dense,
            "completion_tokens": ct,
            "token_ids_len": len(token_ids) if isinstance(token_ids, list) else 0,
            "eligible": trace.eligible,
            "d_consensus": trace.d_consensus,
            "token_ids_match": trace.token_ids_match,
            "chaos_version": chaos.get("version"),
        }
        if arm == "neutral":
            row["ok"] = row["ok"] and not trace.eligible
        elif arm == "panic":
            row["ok"] = row["ok"] and trace.eligible and trace.token_ids_match
        rows.append(row)

    panic_ok = [r for r in rows if r.get("arm") == "panic" and r.get("ok")]
    neutral_ok = [r for r in rows if r.get("arm") == "neutral" and r.get("ok")]
    dense_panic = [r for r in rows if r.get("arm") == "panic" and r.get("token_ids_match")]
    return {
        "n_files": len(rows),
        "n_panic_ok": len(panic_ok),
        "n_neutral_ok": len(neutral_ok),
        "n_dense_panic": len(dense_panic),
        "rows": rows,
    }


def run_miner_scan() -> int:
    proc = subprocess.run(
        [str(PY), str(ROOT / "research" / "caa_mine_natural_pairs.py"), "--scan", "--unicaa01-corpus"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return proc.returncode


def run_export_packs() -> int:
    proc = subprocess.run(
        [str(PY), str(ROOT / "research" / "export_calm_curation_packs.py"), "--glob", CORPUS_GLOB],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality-only", action="store_true")
    ap.add_argument("--skip-scan", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    report = quality_report()
    SCAN_REPORT.parent.mkdir(parents=True, exist_ok=True)
    SCAN_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"quality_report": str(SCAN_REPORT), **{k: report[k] for k in report if k != "rows"}}, indent=2))

    if args.quality_only:
        return 0

    rc = 0
    if not args.skip_scan:
        rc |= run_miner_scan()
    if not args.skip_export:
        rc |= run_export_packs()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
