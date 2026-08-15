#!/usr/bin/env python
"""Run calm-tail curation batch with checkpointing and pilot mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from build_curator_briefs import BRIEFS_ROOT, build_all_briefs  # noqa: E402
from calm_curator_orchestrator import curate_batch  # noqa: E402
from calm_curator_schemas import CuratorBrief  # noqa: E402
from validate_calm_curator_output import brief_from_dict, write_gate_aggregate  # noqa: E402

ARTIFACTS = ROOT / "artifacts" / "caa" / "UNI_CAA01_corpus"
CHECKPOINT = ARTIFACTS / "curator_checkpoint.json"
GATE_OUT = ARTIFACTS / "caa_calm_corpus_gate.json"

_CONF_ORDER = {"high": 0, "medium": 1, "low": 2}


def _load_briefs(briefs_dir: Path) -> list[CuratorBrief]:
    index = briefs_dir / "index.json"
    if not index.is_file():
        build_all_briefs()
    paths = json.loads(index.read_text(encoding="utf-8")).get("files") or []
    out: list[CuratorBrief] = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        out.append(brief_from_dict(data))
    return out


def _select_pilot(briefs: list[CuratorBrief], n: int, pair_ids: list[str] | None) -> list[CuratorBrief]:
    if pair_ids:
        # Prefer one brief per pair_id (best confidence), then allow explicit brief_id hits.
        by_pair: dict[str, list[CuratorBrief]] = {}
        explicit: list[CuratorBrief] = []
        for b in briefs:
            if b.brief_id in pair_ids:
                explicit.append(b)
            elif b.pair_id in pair_ids:
                by_pair.setdefault(b.pair_id, []).append(b)
        picked: list[CuratorBrief] = []
        seen: set[str] = set()
        for b in explicit:
            if b.brief_id not in seen:
                picked.append(b)
                seen.add(b.brief_id)
        for pid in pair_ids:
            cands = by_pair.get(pid) or []
            cands = sorted(cands, key=lambda b: (_CONF_ORDER.get(b.tracer_confidence, 9), b.brief_id))
            for b in cands[:1]:
                if b.brief_id not in seen:
                    picked.append(b)
                    seen.add(b.brief_id)
        return picked[:n] if n > 0 else picked
    sorted_briefs = sorted(briefs, key=lambda b: (_CONF_ORDER.get(b.tracer_confidence, 9), b.brief_id))
    # Ensure p01 in pilot if present
    p01 = [b for b in sorted_briefs if b.pair_id == "p01_priority_queue"]
    rest = [b for b in sorted_briefs if b.pair_id != "p01_priority_queue"]
    ordered = (p01[:1] + rest)[:n]
    return ordered


def _load_checkpoint() -> set[str]:
    if not CHECKPOINT.is_file():
        return set()
    data = json.loads(CHECKPOINT.read_text(encoding="utf-8-sig"))
    return set(data.get("completed") or [])


def _save_checkpoint(completed: set[str], results: list[dict[str, Any]]) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(
        json.dumps({"completed": sorted(completed), "results": results}, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="UNI_CAA01 calm-tail curation batch")
    ap.add_argument("--briefs-dir", type=Path, default=BRIEFS_ROOT)
    ap.add_argument("--pilot", type=int, default=0, help="Run only N briefs (pilot mode)")
    ap.add_argument("--pair-id", action="append", dest="pair_ids", help="Restrict to pair/brief id")
    ap.add_argument("--allow-skip-qa", action="store_true", help="Dev only — skip DeepSeek QA")
    ap.add_argument("--dry-run", action="store_true", help="Gates only; do not write calm_tails")
    ap.add_argument("--rebuild-briefs", action="store_true")
    ap.add_argument(
        "--force-redo",
        action="store_true",
        help="Ignore checkpoint completed set (re-curate selected briefs)",
    )
    args = ap.parse_args()

    if args.rebuild_briefs or not (args.briefs_dir / "index.json").is_file():
        build_all_briefs()

    briefs = _load_briefs(args.briefs_dir)
    done = set() if args.force_redo else _load_checkpoint()
    pending = [b for b in briefs if b.brief_id not in done]

    if args.pilot > 0:
        pending = _select_pilot(pending or briefs, args.pilot, args.pair_ids)
    elif args.pair_ids:
        pending = _select_pilot(pending or briefs, len(briefs), args.pair_ids)

    if not pending:
        print(json.dumps({"status": "nothing_to_do", "completed": len(done)}))
        return 0

    print(f"Curating {len(pending)} brief(s): {[b.brief_id for b in pending]}")
    results = curate_batch(
        pending,
        allow_skip_qa=args.allow_skip_qa,
        dry_run=args.dry_run,
    )

    result_dicts = [r.to_dict() for r in results]
    for r in results:
        if r.status == "admitted":
            done.add(r.brief_id)
    _save_checkpoint(done, result_dicts)

    from calm_curator_schemas import CuratorGateReport, GateCheck  # noqa: E402

    reports = []
    for r in results:
        gr = r.gate_report or {}
        checks = [GateCheck(c["id"], c["pass"], c.get("detail", "")) for c in gr.get("checks", [])]
        reports.append(
            CuratorGateReport(r.brief_id, r.model, bool(gr.get("pass")), checks, False, r.calm_tail_path)
        )
    summary = write_gate_aggregate(reports, GATE_OUT)
    summary["batch_results"] = result_dicts
    GATE_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if all(r.status == "admitted" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
