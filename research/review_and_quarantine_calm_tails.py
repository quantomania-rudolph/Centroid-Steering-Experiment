#!/usr/bin/env python
"""Senior review + quarantine for UNI_CAA01 calm tails.

Downstream miner only consumes meta.status == 'admitted' (pipeline_version >= 5).
Quarantine flips status and records reason; files stay on disk for audit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from assemble_calm_completion import extract_main_py  # noqa: E402
from verify_admitted_tail import verify_one  # noqa: E402

TAILS = ROOT / "artifacts/caa/UNI_CAA01_corpus/calm_tails"
BRIEFS = ROOT / "artifacts/caa/UNI_CAA01_corpus/curator_briefs"
PROOF = ROOT / "artifacts/caa/UNI_CAA01_corpus/curator_hardening_proof"
MANIFEST = TAILS / "_quarantine_manifest.json"

# Locked-prefix defects that cannot be edited without breaking BPE / miner d-key.
PREFIX_HARD_QUARANTINE: list[tuple[str, re.Pattern[str]]] = [
    ("prefix_first_person_i_will", re.compile(r"\bI will\b")),
    ("prefix_first_person_i_asked", re.compile(r"\bI asked\b", re.I)),
    ("prefix_stub_word", re.compile(r"\bstub\b", re.I)),
    ("prefix_fix_colon_imperative", re.compile(r"^Fix:\s", re.I)),
    ("prefix_prompt_paste_imperative", re.compile(r"\b(?:Implement|Write|Create|Edit|Build)\b")),
]

NARR_SOFT_FLAGS: list[tuple[str, re.Pattern[str]]] = [
    ("narr_stub_word", re.compile(r"\bstub\b", re.I)),
    ("narr_first_person", re.compile(r"\bI (?:will|asked|am going)\b", re.I)),
    ("narr_placeholder", re.compile(r"NotImplemented|TODO|FIXME|placeholder", re.I)),
    ("narr_very_short", re.compile(r"^.{0,35}$")),
    ("narr_broken_doc_leak", re.compile(r"Broken ", re.I)),
]


def _narrative(text: str) -> str:
    m = re.search(r"^```", text, re.M)
    block = text[: m.start()] if m else text
    # drop path line
    lines = []
    for line in block.splitlines():
        s = line.strip()
        if s.endswith("/main.py") or s.endswith("\\main.py"):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def review_one(txt_path: Path) -> dict:
    meta_path = txt_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    brief_id = str(meta.get("brief_id") or txt_path.stem)
    brief_path = BRIEFS / f"{brief_id}.json"
    brief = json.loads(brief_path.read_text(encoding="utf-8")) if brief_path.is_file() else {}
    prefix = str((brief.get("prefix_lock") or {}).get("prefix_text") or "")
    text = txt_path.read_text(encoding="utf-8")
    narr = _narrative(text)
    code = extract_main_py(text)

    prefix_hits = [name for name, rx in PREFIX_HARD_QUARANTINE if rx.search(prefix)]
    narr_hits = [name for name, rx in NARR_SOFT_FLAGS if rx.search(narr)]
    code_hits: list[str] = []
    if re.search(r'"""Broken|NotImplementedError|TODO|FIXME', code):
        code_hits.append("code_stub_or_broken")
    if len(code.strip()) < 40:
        code_hits.append("code_too_short")

    verify = verify_one(txt_path)
    scores = (meta.get("qa") or {}).get("scores") or (meta.get("judge") or {}).get("scores") or {}
    low_scores = {
        k: v
        for k, v in scores.items()
        if isinstance(v, (int, float)) and v < 4
    }

    hard_q = bool(prefix_hits) or bool(code_hits) or not verify.get("pass")
    soft_q = bool(narr_hits) or bool(low_scores)
    # If only soft narr flags that are entirely inside the locked prefix, escalate.
    if narr_hits and prefix_hits:
        hard_q = True

    verdict = "quarantine" if hard_q else ("watch" if soft_q else "keep")
    return {
        "brief_id": brief_id,
        "pair_id": brief.get("pair_id"),
        "file": txt_path.name,
        "status_now": meta.get("status"),
        "verdict": verdict,
        "prefix_hits": prefix_hits,
        "narr_hits": narr_hits,
        "code_hits": code_hits,
        "verify_pass": bool(verify.get("pass")),
        "failed_checks": [k for k, v in (verify.get("checks") or {}).items() if not v],
        "low_scores": low_scores,
        "prefix_text": prefix[:180],
        "narrative": narr.replace("\n", " | ")[:220],
        "pipeline_version": meta.get("pipeline_version"),
        "editorial_hand_admit": bool(meta.get("editorial_hand_admit")),
        "editorial_rewrite": bool(meta.get("editorial_rewrite")),
    }


def apply_quarantine(rows: list[dict], *, dry_run: bool) -> list[dict]:
    applied = []
    for row in rows:
        if row["verdict"] != "quarantine":
            continue
        meta_path = TAILS / row["file"].replace(".txt", ".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        reason = ",".join(row["prefix_hits"] or row["code_hits"] or ["review_fail"])
        entry = {
            "file": row["file"],
            "brief_id": row["brief_id"],
            "pair_id": row["pair_id"],
            "reason": reason,
            "prefix_hits": row["prefix_hits"],
            "code_hits": row["code_hits"],
            "prefix_text": row["prefix_text"],
        }
        if not dry_run:
            meta["status"] = "quarantined"
            meta["quarantine"] = {
                "reason": reason,
                "prefix_hits": row["prefix_hits"],
                "code_hits": row["code_hits"],
                "review": "senior_slam_dunk_pass",
            }
            # Keep pipeline_version so history is clear; miner requires admitted.
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        applied.append(entry)
    return applied


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write quarantined status to meta")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run or not args.apply

    rows = [review_one(p) for p in sorted(TAILS.glob("unicaa01_*.txt"))]
    keep = [r for r in rows if r["verdict"] == "keep"]
    watch = [r for r in rows if r["verdict"] == "watch"]
    quar = [r for r in rows if r["verdict"] == "quarantine"]

    applied = apply_quarantine(rows, dry_run=dry) if (args.apply or args.dry_run) else []

    # Downstream-ready set = currently admitted after optional apply
    ready = []
    for p in sorted(TAILS.glob("unicaa01_*.txt")):
        meta = json.loads(p.with_suffix(".meta.json").read_text(encoding="utf-8"))
        if meta.get("status") == "admitted" and int(meta.get("pipeline_version") or 0) >= 5:
            ready.append(p.name)

    payload = {
        "dry_run": dry,
        "n_reviewed": len(rows),
        "n_keep": len(keep),
        "n_watch": len(watch),
        "n_quarantine": len(quar),
        "n_downstream_ready": len(ready),
        "quarantine": quar,
        "watch": watch,
        "keep": [{"brief_id": r["brief_id"], "pair_id": r["pair_id"], "file": r["file"]} for r in keep],
        "downstream_ready_files": ready,
        "applied": applied,
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    (PROOF / "G21_slam_dunk_review.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.apply and not dry:
        MANIFEST.write_text(
            json.dumps(
                {
                    "n_quarantined": len(applied),
                    "note": "Miner ignores status!=admitted. Files retained for audit.",
                    "quarantined": applied,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "dry_run": dry,
                "n_reviewed": len(rows),
                "n_keep": len(keep),
                "n_watch": len(watch),
                "n_quarantine": len(quar),
                "n_downstream_ready": len(ready),
                "quarantine_ids": [r["brief_id"] for r in quar],
                "watch_ids": [r["brief_id"] for r in watch],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
