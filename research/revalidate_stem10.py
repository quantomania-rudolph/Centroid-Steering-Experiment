#!/usr/bin/env python
"""Re-validate all STEM10 task results and regenerate consolidated report."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
RESULTS = RESEARCH / "results" / "stem10"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
QWEN = ROOT / "qwen code"

sys.path.insert(0, str(RESEARCH))
from run_stem10_experiments import TASKS, analyze_experiment, synthesize_stem10_caa, write_stem10_report  # noqa: E402


def main() -> int:
    all_results = []
    for task in TASKS:
        path = RESULTS / f"{task.task_id}_{task.slug}.json"
        if not path.is_file():
            print(f"MISSING {task.task_id}")
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Re-load experiment json for chaos data
        exp = ROOT / "experiments" / "out" / f"stem10_{task.task_id.lower()}_{task.slug}_latest.json"
        if not exp.is_file():
            exp = ROOT / "experiments" / "out" / f"{raw.get('experiment', '')}_latest.json"
        if exp.is_file():
            data = json.loads(exp.read_text(encoding="utf-8"))
            analyzed = analyze_experiment(data, task)
            analyzed["task_id"] = task.task_id
            analyzed["domain"] = task.domain
            analyzed["slug"] = task.slug
            analyzed["success"] = (
                analyzed.get("n_invocations", 0) > 0
                and all(inv.get("runtime_ok") for inv in analyzed.get("invocations", []))
            )
            path.write_text(json.dumps(analyzed, indent=2, ensure_ascii=False), encoding="utf-8")
            all_results.append(analyzed)
            print(f"{task.task_id} success={analyzed['success']} inv={analyzed['n_invocations']}")
        else:
            all_results.append(raw)

    caa = synthesize_stem10_caa(all_results)
    consolidated = {
        "stamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "n_tasks": len(all_results),
        "n_success": sum(1 for t in all_results if t.get("success")),
        "tasks": all_results,
        "caa_recommendations": caa,
    }
    (RESULTS / "stem10_consolidated.json").write_text(
        json.dumps(consolidated, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_stem10_report(all_results, caa, RESEARCH / "STEM10_CHAOS_CAA_FINDINGS.md")
    print(json.dumps({"n_success": consolidated["n_success"], "n_tasks": consolidated["n_tasks"]}))
    return 0 if consolidated["n_success"] == len(TASKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
