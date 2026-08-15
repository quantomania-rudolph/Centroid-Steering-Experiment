#!/usr/bin/env python
"""Run 25 hard STEM25 Aider×Qwen experiments + cross-field chaos error mapping."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
STARTERS = RESEARCH / "stem25_starters"
RESULTS = RESEARCH / "results" / "stem25"
QWEN = ROOT / "qwen code"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SERVER = "http://127.0.0.1:8767"

sys.path.insert(0, str(ROOT / "src"))
from jlens_qwen.chaos.checker_audit import audit_invocation  # noqa: E402

sys.path.insert(0, str(RESEARCH))
from caa_error_combo_study import (  # noqa: E402
    audit_code_file,
    flag_rates,
    invocation_verdict,
    recipe_summary,
    score_text,
    temporal_resolution,
)
from stem25_tasks import TASKS, StemTask  # noqa: E402


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 7200.0) -> dict:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_server(timeout_s: float = 300.0) -> dict:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            h = _http_json("GET", f"{SERVER}/health", timeout=10)
            if h.get("ok"):
                return h
            last = h
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(3)
    raise RuntimeError(f"server not ready: {last}")


def deploy_starter(slug: str) -> None:
    src = STARTERS / slug / "main.py"
    dst_dir = QWEN / slug
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / "main.py")


def analyze_experiment(data: dict, task: StemTask) -> dict[str, Any]:
    inv_rows = []
    for inv in data.get("invocations") or []:
        chaos = inv.get("chaos") or {}
        tokens = chaos.get("tokens") or []
        audit = audit_invocation(chaos)
        code_audit = audit_code_file(QWEN / task.slug / "main.py")
        assistant = inv.get("assistant_preview") or inv.get("assistant") or ""
        temporal = temporal_resolution(tokens)
        lint_ok = bool((data.get("lint") or {}).get("ok"))
        rc = subprocess.run(
            task.runtime_cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        runtime_ok = rc.returncode == 0
        verdict = invocation_verdict(
            code=code_audit,
            audit=audit,
            temporal=temporal,
            runtime_ok=runtime_ok,
            lint_ok=lint_ok,
        )
        inv_rows.append(
            {
                "invocation_index": inv.get("invocation_index"),
                "id": inv.get("id"),
                "completion_tokens": inv.get("completion_tokens"),
                "seconds": inv.get("seconds"),
                "verdict": verdict,
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
                "monitor": chaos.get("monitor") or {},
                "runtime_ok": runtime_ok,
                "runtime_stderr": (rc.stderr or "")[-400:],
                "lint_ok": lint_ok,
            }
        )
    return {
        "task_id": task.task_id,
        "domain": task.domain,
        "slug": task.slug,
        "stem10_echo": task.stem10_echo,
        "experiment": data.get("experiment"),
        "n_invocations": len(inv_rows),
        "invocations": inv_rows,
        "evidence": data.get("evidence"),
        "lint": data.get("lint"),
        "aider": data.get("aider"),
    }


def _runtime_passes(task: StemTask) -> bool:
    rc = subprocess.run(
        task.runtime_cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return rc.returncode == 0


def run_one_task(task: StemTask, *, max_retries: int = 5) -> dict[str, Any]:
    out = RESULTS / f"{task.task_id}_{task.slug}.json"
    if out.is_file():
        prior = json.loads(out.read_text(encoding="utf-8"))
        if prior.get("success"):
            return prior
    if not _runtime_passes(task):
        deploy_starter(task.slug)
    experiment = f"stem25_{task.task_id.lower()}_{task.slug}"
    last_err = ""
    last_analyzed: dict[str, Any] | None = None
    for attempt in range(1, max_retries + 1):
        try:
            wait_server(timeout_s=120.0)
        except RuntimeError as exc:
            last_err = str(exc)
            time.sleep(20)
            continue
        if attempt > 1:
            rc_chk = subprocess.run(
                task.runtime_cmd,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if rc_chk.returncode != 0:
                deploy_starter(task.slug)
            time.sleep(15)
        cmd = [
            str(PY),
            str(ROOT / "experiments" / "run_aider_agent.py"),
            "--task-slug",
            task.slug,
            "--experiment",
            experiment,
            "--message",
            task.message,
            "--layers",
            "16",
            "--chaos-only",
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
        elapsed = time.perf_counter() - t0
        latest = ROOT / "experiments" / "out" / f"{experiment}_latest.json"
        if not latest.is_file():
            last_err = "no latest json"
            continue
        data = json.loads(latest.read_text(encoding="utf-8"))
        n_inv = len(data.get("invocations") or [])
        if n_inv == 0:
            last_err = (data.get("aider") or {}).get("stdout_tail", proc.stdout)[-500:]
            continue
        analyzed = analyze_experiment(data, task)
        analyzed["elapsed_s"] = elapsed
        aider_rc = (data.get("aider") or {}).get("returncode", proc.returncode)
        analyzed["aider_rc"] = aider_rc
        analyzed["attempt"] = attempt
        final_runtime_ok = _runtime_passes(task)
        analyzed["success"] = (
            n_inv > 0
            and aider_rc == 0
            and final_runtime_ok
        )
        last_analyzed = analyzed
        if analyzed["success"]:
            return analyzed
        last_err = analyzed["invocations"][-1].get("runtime_stderr", "runtime fail") if analyzed["invocations"] else last_err
    if last_analyzed is not None:
        last_analyzed["success"] = False
        last_analyzed["error"] = last_err
        return last_analyzed
    return {
        "task_id": task.task_id,
        "domain": task.domain,
        "slug": task.slug,
        "stem10_echo": task.stem10_echo,
        "success": False,
        "error": last_err,
        "attempt": max_retries,
        "n_invocations": 0,
        "invocations": [],
    }


def load_all_task_results() -> list[dict]:
    results: list[dict] = []
    for task in TASKS:
        out = RESULTS / f"{task.task_id}_{task.slug}.json"
        if out.is_file():
            results.append(json.loads(out.read_text(encoding="utf-8")))
    return results


def write_stem25_report(tasks: list[dict], mapper_md: str, path: Path) -> None:
    lines = [
        "# STEM25 Ultra-Hard Tasks — Results & Chaos Mapping",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "",
        f"**Success:** {sum(1 for t in tasks if t.get('success'))}/{len(tasks)} tasks",
        "",
        "## Task summary",
        "",
        "| ID | Domain | STEM10 echo | Inv | Runtime | Success | Top recipes |",
        "|----|--------|-------------|-----|---------|---------|-------------|",
    ]
    for t in tasks:
        invs = t.get("invocations") or []
        recs = set()
        for inv in invs:
            recs.update((inv.get("recipe_summary") or {}).get("unique_recipes") or [])
        rt = "ok" if t.get("success") else "FAIL"
        lines.append(
            f"| {t.get('task_id')} | {t.get('domain')} | {t.get('stem10_echo','')} | "
            f"{len(invs)} | {rt} | {'yes' if t.get('success') else 'no'} | {', '.join(list(recs)[:3]) or '—'} |"
        )
    lines += ["", "## Per-invocation A–G detail", ""]
    for t in tasks:
        lines.append(f"### {t.get('task_id')} — {t.get('slug')}")
        for inv in t.get("invocations") or []:
            chk = inv.get("checker_audit", {}).get("checkers") or {}
            fired = [k for k, v in chk.items() if v.get("fired")]
            fa = inv.get("flag_rates") or {}
            lines.append(
                f"- Inv#{inv.get('invocation_index')}: {inv.get('verdict')} | "
                f"checkers={fired} | B_stub={fa.get('B_stub',0):.1%} | "
                f"recipes={(inv.get('recipe_summary') or {}).get('unique_recipes')}"
            )
        if not t.get("invocations"):
            lines.append(f"- **FAILED:** {t.get('error', '')[:200]}")
        lines.append("")
    lines += ["", "---", "", "## Cross-field error mapper", "", mapper_md]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", help="T01..T25 or all")
    parser.add_argument("--retry-failed", action="store_true", help="Only rerun tasks without success")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    selected = TASKS
    if args.retry_failed:
        by_id = {}
        for t in TASKS:
            out = RESULTS / f"{t.task_id}_{t.slug}.json"
            if out.is_file():
                by_id[t.task_id] = json.loads(out.read_text(encoding="utf-8"))
        selected = [t for t in TASKS if not by_id.get(t.task_id, {}).get("success")]
        print(f"Retry-failed: {len(selected)} tasks", flush=True)
    elif args.task != "all":
        want = {x.strip().upper() for x in args.task.split(",")}
        selected = [t for t in TASKS if t.task_id in want]

    if not args.skip_run:
        print("Waiting for server...", flush=True)
        wait_server()

    run_mapper = args.task == "all" and not args.retry_failed

    for task in selected:
        out = RESULTS / f"{task.task_id}_{task.slug}.json"
        print(f"=== {task.task_id} {task.slug} ({task.domain}) ===", flush=True)
        if args.skip_run and out.is_file():
            result = json.loads(out.read_text(encoding="utf-8"))
        else:
            result = run_one_task(task)
            out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        status = "OK" if result.get("success") else "FAIL"
        print(f"  {status} inv={result.get('n_invocations', 0)}", flush=True)
        if not result.get("success"):
            print(f"  err: {str(result.get('error',''))[:200]}", flush=True)
        time.sleep(8)

    all_results = load_all_task_results()
    # merge selected into full task list
    by_id = {t["task_id"]: t for t in all_results}
    for t in selected:
        out = RESULTS / f"{t.task_id}_{t.slug}.json"
        if out.is_file():
            by_id[t.task_id] = json.loads(out.read_text(encoding="utf-8"))
    all_results = [by_id[t.task_id] for t in TASKS if t.task_id in by_id]

    consolidated = {
        "stamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "n_tasks": len(all_results),
        "n_success": sum(1 for t in all_results if t.get("success")),
        "tasks": all_results,
    }
    (RESULTS / "stem25_consolidated.json").write_text(
        json.dumps(consolidated, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    mapper_md = ""
    if run_mapper or args.skip_run:
        subprocess.run(
            [str(PY), str(RESEARCH / "chaos_error_mapper.py")],
            cwd=str(ROOT),
            check=True,
        )
        mapper_md = (RESEARCH / "CHAOS_ERROR_MAPPER_REPORT.md").read_text(encoding="utf-8")
    if run_mapper or args.skip_run:
        write_stem25_report(all_results, mapper_md, RESEARCH / "STEM25_RESULTS_REPORT.md")

    failed = [t["task_id"] for t in all_results if not t.get("success")]
    print(json.dumps({"success": len(all_results) - len(failed), "failed": failed}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
