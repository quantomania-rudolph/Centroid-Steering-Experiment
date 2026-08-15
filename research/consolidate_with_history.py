#!/usr/bin/env python
"""Merge CAA study rounds with historical stress runs; regenerate report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
RESULTS = RESEARCH / "results"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(RESEARCH))

from caa_error_combo_study import (  # noqa: E402
    QWEN,
    audit_code_file,
    audit_invocation,
    flag_rates,
    invocation_verdict,
    recipe_summary,
    score_text,
    synthesize_caa_candidates,
    temporal_resolution,
    write_report,
)

HISTORICAL = [
    ("H1_CALIB_STRESS", "aider_calib_stress_v2_latest.json", "live_panic_demo_scheduler"),
    ("H2_MONITOR_STRESS", "monitor_gate_stress_latest.json", "live_panic_demo_scheduler"),
    ("H3_PANIC_DEMO", "aider_live_panic_demo_scheduler_latest.json", "live_panic_demo_scheduler"),
]


def analyze_experiment(path: Path, round_id: str, task_slug: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    inv_rows = []
    for inv in data.get("invocations") or []:
        chaos = inv.get("chaos") or {}
        tokens = chaos.get("tokens") or []
        audit = audit_invocation(chaos)
        code_path = QWEN / task_slug / "main.py"
        code_audit = audit_code_file(code_path)
        assistant = inv.get("assistant_preview") or inv.get("assistant") or ""
        temporal = temporal_resolution(tokens)
        lint_ok = bool((data.get("lint") or {}).get("ok", True))
        inv_rows.append(
            {
                "invocation_index": inv.get("invocation_index"),
                "id": inv.get("id"),
                "completion_tokens": inv.get("completion_tokens"),
                "seconds": inv.get("seconds"),
                "verdict": invocation_verdict(
                    code=code_audit,
                    audit=audit,
                    temporal=temporal,
                    runtime_ok=True,
                    lint_ok=lint_ok,
                ),
                "code_audit": code_audit,
                "assistant_scores": score_text(assistant),
                "checker_audit": audit,
                "recipe_summary": recipe_summary(chaos),
                "temporal_resolution": temporal,
                "flag_rates": {
                    "A": flag_rates(tokens, "A_"),
                    "B_distress": flag_rates(tokens, "B_distress"),
                    "B_stub": flag_rates(tokens, "B_stub"),
                    "C": flag_rates(tokens, "C_"),
                    "D": flag_rates(tokens, "D_"),
                },
                "runtime_ok": True,
                "lint_ok": lint_ok,
            }
        )
    return {
        "round_id": round_id,
        "task_slug": task_slug,
        "experiment": data.get("experiment"),
        "source": str(path),
        "round_verdict": "PASS" if inv_rows else "NO_DATA",
        "invocations": inv_rows,
    }


def main() -> int:
    rounds: list[dict] = []
    for fname in ["r1_clean.json", "r2_stub.json", "r3_distress.json", "r4_combo.json", "r5_traj.json"]:
        p = RESULTS / fname
        if p.is_file():
            rounds.append(json.loads(p.read_text(encoding="utf-8")))

    for rid, fname, slug in HISTORICAL:
        p = ROOT / "experiments" / "out" / fname
        if p.is_file():
            rounds.append(analyze_experiment(p, rid, slug))

    # Fix verdict for empty rounds
    for r in rounds:
        if not r.get("invocations"):
            r["round_verdict"] = "NO_DATA"

    candidates = synthesize_caa_candidates(rounds)
    # Boost priority from historical stress evidence
    hist_recipe_totals: dict[str, int] = {}
    for r in rounds:
        if not r["round_id"].startswith("H"):
            continue
        for inv in r.get("invocations") or []:
            for rec, cnt in (inv.get("recipe_summary") or {}).get("recipe_counts", {}).items():
                hist_recipe_totals[rec] = hist_recipe_totals.get(rec, 0) + cnt

    for c in candidates:
        rid = c["caa_id"].replace("CAA_", "").replace("_ONLY", "")
        for rec, cnt in hist_recipe_totals.items():
            if rec.replace("_", "") in rid.replace("_", "") or rec in c["caa_id"]:
                c["fires_in_study"] = c.get("fires_in_study", 0) + cnt
                c.setdefault("historical_contexts", []).append(f"{rec}:{cnt}")
        if c.get("fires_in_study", 0) >= 5:
            c["priority"] = "HIGH"
        elif c.get("fires_in_study", 0) >= 1:
            c["priority"] = "MEDIUM"

    consolidated = {
        "stamp": "merged",
        "n_rounds": len(rounds),
        "rounds": rounds,
        "historical_recipe_totals": hist_recipe_totals,
        "caa_candidates": candidates,
    }
    (RESULTS / "consolidated.json").write_text(
        json.dumps(consolidated, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(rounds, candidates, RESEARCH / "CAA_STEERING_REPORT.md")
    print(json.dumps({"historical_recipe_totals": hist_recipe_totals, "priorities": [(c["caa_id"], c["priority"], c["fires_in_study"]) for c in candidates]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
