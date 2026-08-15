#!/usr/bin/env python
"""Phase 4b: quality-gated alpha_0 fit + confirm (UNI_CAA01).

Fit grid on p09+p19+clean with post-hoc quality (garbled/ruff/pytest).
Confirm on hold-out p30+p65; control inject=0 on p09+p19; stability repeat p09.
Writes artifacts/gates/caa_alpha0_4b_gate.json (does not overwrite 4a gate).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.caa.alpha0_gate import (  # noqa: E402
    MAX_ALPHA_CAP_4B,
    classify_failure,
    inject_stats,
    sabotage_alpha_scales,
    score_sweep_row,
    select_alpha_0_4b,
)
from jlens_qwen.caa.cast_panic_stubs import (  # noqa: E402
    PANIC_SLUGS_CONFIRM,
    PANIC_SLUGS_FIT,
    panic_message,
    restore_panic_stub,
)
from jlens_qwen.caa.cast_stream_gate import chaos_tokens_from_session  # noqa: E402
from jlens_qwen.caa.gate_io import write_gate  # noqa: E402
from jlens_qwen.caa.task_quality import score_task_quality  # noqa: E402
from prove_cast_alpha0_live import (  # noqa: E402
    _load_gate,
    _run_aider,
    _run_demo,
    _stub_mean,
)
from unicaa01_restart_server import fetch_health, restart_server, stop_server  # noqa: E402

GATES = ROOT / "artifacts" / "gates"
OUT = ROOT / "experiments" / "out"
QWEN_CODE = ROOT / "qwen code"
OUT_PATH = GATES / "caa_alpha0_4b_gate.json"
SWEEP_JSON = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "calibration" / "alpha0_sweep_4b.json"
SNAPSHOT_ROOT = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "calibration" / "alpha0_4b_snapshots"
STREAM_GATE = GATES / "cast_stream_live_gate.json"
INJECT_GATE = GATES / "cast_inject_live_gate.json"
TENSION_GATE = GATES / "caa_cast_tension_gate.json"
ALPHA0_GATE = GATES / "caa_alpha0_live_gate.json"

DEFAULT_ALPHAS = [0.80, 0.85, 0.90, 0.95]

FIT_TASKS = [
    {"slug": "unicaa01_p09_panic", "suffix": "p09", "expected_demo": [1, 2, 3]},
    {"slug": "unicaa01_p19_panic", "suffix": "p19", "expected_demo": 1},
]
CONFIRM_TASKS = [
    {"slug": "unicaa01_p30_panic", "suffix": "p30", "expected_demo": "localhost"},
    {"slug": "unicaa01_p65_panic", "suffix": "p65", "expected_demo": [(1, 6)]},
]


def _alpha_tag(alpha_0: float) -> str:
    return f"a{int(round(alpha_0 * 100)):03d}"


def _experiment(alpha_0: float, suffix: str, *, phase: str = "fit") -> str:
    return f"cast_alpha0_4b_{phase}_{_alpha_tag(alpha_0)}_{suffix}"


def _load_checkpoint() -> dict:
    if not SWEEP_JSON.is_file():
        return {"alphas": [], "fit_rows": [], "selected": None}
    return json.loads(SWEEP_JSON.read_text(encoding="utf-8"))


def _save_checkpoint(payload: dict) -> None:
    SWEEP_JSON.parent.mkdir(parents=True, exist_ok=True)
    SWEEP_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_session_json(session_path: str | None) -> dict | None:
    if not session_path:
        return None
    path = Path(session_path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _session_diag_fields(session_json: dict | None) -> dict[str, Any]:
    invocations = (session_json or {}).get("invocations") or []
    if not invocations:
        return {"completion_tokens": None, "assistant_preview": None}
    inv = invocations[0]
    preview = str(inv.get("assistant_preview") or "")
    return {
        "completion_tokens": inv.get("completion_tokens"),
        "assistant_preview": preview[:200] if preview else None,
    }


def _annotate_failure_class(row: dict, *, alpha_0: float) -> None:
    row["alpha_0"] = alpha_0
    session_json = _load_session_json(row.get("session"))
    row["failure_class"] = classify_failure(row, session_json)


def _panic_row_diag(row: dict) -> dict[str, Any]:
    session_json = _load_session_json(row.get("session"))
    diag = _session_diag_fields(session_json)
    return {
        "slug": row.get("slug"),
        "pass": row.get("pass"),
        "failure_class": row.get("failure_class"),
        "demo_ok": row.get("demo_ok"),
        "quality_ok": row.get("quality_ok"),
        "stats": row.get("stats"),
        "completion_tokens": diag["completion_tokens"],
        "assistant_preview": diag["assistant_preview"],
    }


def _write_diag_out(
    path: Path,
    *,
    alpha_0: float,
    inject: bool,
    elicit: bool,
    confirm: dict,
) -> dict:
    payload = {
        "alpha_0": alpha_0,
        "inject": inject,
        "elicit": elicit,
        "confirm": {
            "pass": confirm.get("pass"),
            "rows": [_panic_row_diag(row) for row in confirm.get("rows", [])],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def detect_oversteer(fit_inject_rows: list[dict], control_rows: list[dict]) -> bool:
    """True when inject=1 failed quality but inject=0 control passed on same slug."""
    fit_by_slug = {r["slug"]: r for r in fit_inject_rows}
    for row in control_rows:
        fit_row = fit_by_slug.get(row["slug"])
        if fit_row and not fit_row.get("quality_ok") and row.get("quality_ok"):
            return True
    return False


def _snapshot_main(alpha_0: float, slug: str, task_dir: Path) -> Path:
    dest_dir = SNAPSHOT_ROOT / _alpha_tag(alpha_0) / slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "main.py"
    shutil.copy2(task_dir / "main.py", dest)
    return dest


def _quality_dict(slug: str, task_dir: Path) -> dict[str, Any]:
    result = score_task_quality(slug, task_dir)
    payload = asdict(result)
    payload["ok"] = result.ok
    return payload


def _verify_health_alpha(alpha_0: float, *, inject: bool, retries: int = 3) -> tuple[bool, dict]:
    last: dict = {}
    for _ in range(retries):
        health = fetch_health() or {}
        last = health
        ok = (
            health.get("ok")
            and health.get("cast_shadow")
            and bool(health.get("cast_inject")) == inject
            and str(health.get("device", "")).startswith("xpu")
            and abs(float(health.get("cast_alpha_0") or -1) - alpha_0) < 1e-6
        )
        if ok:
            return True, health
        time.sleep(3)
    return False, last


def _panic_needs_retry(
    stats: dict[str, Any],
    demo_ok: bool,
    quality_ok: bool,
    *,
    require_inject: bool,
) -> bool:
    if not demo_ok or not quality_ok:
        return True
    if not require_inject:
        return stats.get("inject_count", 0) != 0
    if stats.get("inject_count", 0) < 1:
        return True
    if stats.get("g_stream_count", 0) < 1:
        return True
    return False


def _run_panic(
    *,
    alpha_0: float,
    task: dict,
    phase: str,
    max_retries: int,
    aider_timeout_s: float,
    require_inject: bool,
    elicit: bool = True,
) -> dict:
    slug = task["slug"]
    experiment = _experiment(alpha_0, task["suffix"], phase=phase)
    task_dir = QWEN_CODE / slug
    last_row: dict | None = None
    for attempt in range(max_retries + 1):
        restore_panic_stub(task_dir, slug)
        session_path = _run_aider(
            slug, experiment, panic_message(slug, elicit=elicit), aider_timeout_s=aider_timeout_s
        )
        tokens = chaos_tokens_from_session(session_path)
        stats = inject_stats(tokens)
        snapshot = _snapshot_main(alpha_0, slug, task_dir)
        snapshot_dir = snapshot.parent
        quality = _quality_dict(slug, snapshot_dir)
        demo_ok, demo_value, demo_err = _run_demo(snapshot_dir, task["expected_demo"])
        last_row = {
            "slug": slug,
            "stats": stats,
            "demo_ok": demo_ok,
            "demo_value": demo_value,
            "demo_err": demo_err,
            "quality_ok": quality["ok"],
            "quality": quality,
            "session": str(session_path),
            "snapshot": str(snapshot),
            "attempt": attempt + 1,
            "phase": phase,
            "require_inject": require_inject,
        }
        if not _panic_needs_retry(stats, demo_ok, quality["ok"], require_inject=require_inject):
            break
        print(
            f"  {slug} retry {attempt + 1}/{max_retries}: inject={stats.get('inject_count')} "
            f"g_stream={stats.get('g_stream_count')} demo_ok={demo_ok} quality_ok={quality['ok']}",
            flush=True,
        )
    assert last_row is not None
    last_row["pass"] = _panic_passes(alpha_0, last_row, require_inject=require_inject)
    _annotate_failure_class(last_row, alpha_0=alpha_0)
    return last_row


def _run_fit_alpha_row(
    alpha_0: float,
    *,
    max_retries: int,
    aider_timeout_s: float,
    checkpoint: dict,
) -> dict:
    print(f"\n=== FIT alpha_0={alpha_0} ===", flush=True)
    if not restart_server(device="xpu", shadow=True, inject=True, alpha_0=alpha_0):
        raise RuntimeError(f"server restart failed for alpha_0={alpha_0}")
    health_ok, health = _verify_health_alpha(alpha_0, inject=True)
    if not health_ok:
        stop_server()
        raise RuntimeError(f"health mismatch for alpha_0={alpha_0}: {health}")

    panic_rows = []
    for task in FIT_TASKS:
        row = _run_panic(
            alpha_0=alpha_0,
            task=task,
            phase="fit",
            max_retries=max_retries,
            aider_timeout_s=aider_timeout_s,
            require_inject=True,
        )
        panic_rows.append(row)
        stats = row["stats"]
        print(
            f"  {row['slug']}: inject={stats.get('inject_count')} quality_ok={row['quality_ok']} "
            f"attempts={row.get('attempt')}",
            flush=True,
        )
        checkpoint["fit_rows"] = [
            r for r in checkpoint.get("fit_rows", []) if r.get("alpha_0") != alpha_0
        ] + [{"alpha_0": alpha_0, "partial": True, "panics": panic_rows}]
        _save_checkpoint(checkpoint)

    clean_experiment = _experiment(alpha_0, "clean", phase="fit")
    clean_path = _run_aider(
        "chaos_proof_clean",
        clean_experiment,
        "Edit chaos_proof_clean/main.py only. Ensure iterative fibonacci(10) prints 55.",
        aider_timeout_s=aider_timeout_s,
    )
    clean_tokens = chaos_tokens_from_session(clean_path)
    clean_stats = inject_stats(clean_tokens)
    clean_stub = _stub_mean(clean_tokens)
    print(
        f"  clean: inject={clean_stats.get('inject_count')} "
        f"mean_alpha={clean_stats.get('mean_alpha', 0):.4f}",
        flush=True,
    )

    row = score_sweep_row(
        alpha_0=alpha_0,
        panics=panic_rows,
        clean={"stats": clean_stats, "stub_mean": clean_stub, "session": str(clean_path)},
        require_panic_inject=True,
    )
    row["health"] = health
    row["complete"] = True
    row["phase"] = "fit"
    return row


def _binary_midpoint_alpha(fit_rows: list[dict], grid: list[float]) -> float | None:
    by_alpha = {float(r["alpha_0"]): bool(r.get("pass")) for r in fit_rows if r.get("complete")}
    ordered = sorted(grid)
    for lo, hi in zip(ordered, ordered[1:]):
        if by_alpha.get(lo) and not by_alpha.get(hi):
            return round((lo + hi) / 2, 4)
    return None


def _next_lower_passing_alpha(selected: float, fit_rows: list[dict]) -> float | None:
    passing = sorted(
        {
            float(r["alpha_0"])
            for r in fit_rows
            if r.get("pass") and float(r["alpha_0"]) < selected - 1e-9
        },
        reverse=True,
    )
    return passing[0] if passing else None


def _panic_passes(alpha_0: float, row: dict, *, require_inject: bool) -> bool:
    scored = score_sweep_row(
        alpha_0=alpha_0,
        panics=[row],
        clean={"stats": {"inject_count": 0, "mean_alpha": 0.0}, "stub_mean": 0.0},
        require_panic_inject=require_inject,
    )
    return bool(scored["panics"][0]["pass"])


def _tasks_by_suffix(suffixes: str, pool: list[dict]) -> list[dict]:
    if not suffixes.strip():
        return []
    wanted = {s.strip() for s in suffixes.split(",") if s.strip()}
    return [task for task in pool if task["suffix"] in wanted]


def _rescore_panic_row_from_snapshot(
    *,
    alpha_0: float,
    task: dict,
    phase: str,
    require_inject: bool,
) -> dict:
    slug = task["slug"]
    snapshot = SNAPSHOT_ROOT / _alpha_tag(alpha_0) / slug / "main.py"
    if not snapshot.is_file():
        raise FileNotFoundError(f"missing snapshot {snapshot}")
    snapshot_dir = snapshot.parent
    session_path = OUT / f"cast_alpha0_4b_{phase}_{_alpha_tag(alpha_0)}_{task['suffix']}_latest.json"
    tokens = chaos_tokens_from_session(session_path) if session_path.is_file() else []
    stats = inject_stats(tokens)
    quality = _quality_dict(slug, snapshot_dir)
    demo_ok, demo_value, demo_err = _run_demo(snapshot_dir, task["expected_demo"])
    row = {
        "slug": slug,
        "stats": stats,
        "demo_ok": demo_ok,
        "demo_value": demo_value,
        "demo_err": demo_err,
        "quality_ok": quality["ok"],
        "quality": quality,
        "session": str(session_path) if session_path.is_file() else None,
        "snapshot": str(snapshot),
        "attempt": 0,
        "phase": phase,
        "require_inject": require_inject,
        "rescore_only": True,
    }
    row["pass"] = _panic_passes(alpha_0, row, require_inject=require_inject)
    _annotate_failure_class(row, alpha_0=alpha_0)
    return row


def _run_confirm_at_alpha(
    alpha_0: float,
    *,
    max_retries: int,
    aider_timeout_s: float,
    run_tasks: list[dict] | None = None,
    rescore_tasks: list[dict] | None = None,
    inject: bool = True,
    elicit: bool = True,
) -> dict:
    run_tasks = list(run_tasks) if run_tasks is not None else list(CONFIRM_TASKS)
    rescore_tasks = list(rescore_tasks or [])
    rescore_slugs = {task["slug"] for task in rescore_tasks}
    active_tasks = list(rescore_tasks)
    for task in run_tasks:
        if task["slug"] not in rescore_slugs:
            active_tasks.append(task)
    if not active_tasks:
        raise ValueError("confirm requires at least one task")

    print(f"\n=== CONFIRM alpha_0={alpha_0} inject={inject} elicit={elicit} ===", flush=True)
    if not restart_server(device="xpu", shadow=True, inject=inject, alpha_0=alpha_0):
        raise RuntimeError(f"confirm restart failed for alpha_0={alpha_0}")
    health_ok, health = _verify_health_alpha(alpha_0, inject=inject)
    if not health_ok:
        stop_server()
        raise RuntimeError(f"confirm health mismatch: {health}")

    rows_by_slug: dict[str, dict] = {}
    for task in rescore_tasks:
        row = _rescore_panic_row_from_snapshot(
            alpha_0=alpha_0,
            task=task,
            phase="confirm",
            require_inject=inject,
        )
        rows_by_slug[task["slug"]] = row
        print(
            f"  {row['slug']} (rescore): pass={row['pass']} quality_ok={row['quality_ok']} "
            f"demo_ok={row['demo_ok']}",
            flush=True,
        )

    for task in run_tasks:
        if task["slug"] in rescore_slugs:
            continue
        row = _run_panic(
            alpha_0=alpha_0,
            task=task,
            phase="confirm",
            max_retries=max_retries,
            aider_timeout_s=aider_timeout_s,
            require_inject=inject,
            elicit=elicit,
        )
        rows_by_slug[task["slug"]] = row
        print(
            f"  {row['slug']}: pass={row['pass']} quality_ok={row['quality_ok']} "
            f"failure_class={row.get('failure_class')}",
            flush=True,
        )

    rows = [rows_by_slug[task["slug"]] for task in CONFIRM_TASKS if task["slug"] in rows_by_slug]
    return {"alpha_0": alpha_0, "health": health, "rows": rows, "pass": all(r["pass"] for r in rows)}


def _run_control_at_alpha(
    alpha_0: float,
    *,
    max_retries: int,
    aider_timeout_s: float,
    fit_inject_rows: list[dict],
) -> dict:
    print(f"\n=== CONTROL inject=0 alpha_0={alpha_0} ===", flush=True)
    if not restart_server(device="xpu", shadow=True, inject=False, alpha_0=alpha_0):
        raise RuntimeError(f"control restart failed for alpha_0={alpha_0}")
    health_ok, health = _verify_health_alpha(alpha_0, inject=False)
    if not health_ok:
        stop_server()
        raise RuntimeError(f"control health mismatch: {health}")

    rows = []
    for task in FIT_TASKS:
        row = _run_panic(
            alpha_0=alpha_0,
            task=task,
            phase="control",
            max_retries=max_retries,
            aider_timeout_s=aider_timeout_s,
            require_inject=False,
        )
        row["pass"] = _panic_passes(alpha_0, row, require_inject=False)
        rows.append(row)
    oversteer = detect_oversteer(fit_inject_rows, rows)
    if oversteer:
        fit_by_slug = {r["slug"]: r for r in fit_inject_rows}
        for row in rows:
            fit_row = fit_by_slug.get(row["slug"])
            if fit_row and not fit_row.get("quality_ok") and row.get("quality_ok"):
                row["oversteer"] = True
    for row in rows:
        print(
            f"  {row['slug']}: inject={row['stats'].get('inject_count')} "
            f"quality_ok={row['quality_ok']} pass={row['pass']}",
            flush=True,
        )
    return {
        "alpha_0": alpha_0,
        "health": health,
        "rows": rows,
        "pass": all(r["pass"] for r in rows),
        "oversteer": oversteer,
    }


def _run_stability_at_alpha(
    alpha_0: float,
    *,
    max_retries: int,
    aider_timeout_s: float,
) -> dict:
    print(f"\n=== STABILITY alpha_0={alpha_0} ===", flush=True)
    if not restart_server(device="xpu", shadow=True, inject=True, alpha_0=alpha_0):
        raise RuntimeError(f"stability restart failed for alpha_0={alpha_0}")
    health_ok, health = _verify_health_alpha(alpha_0, inject=True)
    if not health_ok:
        stop_server()
        raise RuntimeError(f"stability health mismatch: {health}")
    task = FIT_TASKS[0]
    row = _run_panic(
        alpha_0=alpha_0,
        task=task,
        phase="stability",
        max_retries=max_retries,
        aider_timeout_s=aider_timeout_s,
        require_inject=True,
    )
    row["pass"] = _panic_passes(alpha_0, row, require_inject=True)
    print(f"  {row['slug']}: pass={row['pass']}", flush=True)
    return {"alpha_0": alpha_0, "health": health, "row": row, "pass": row["pass"]}


def _pre_checks(alphas: list[float]) -> tuple[list[dict], list[dict]]:
    checks: list[dict] = []
    sabotage = [sabotage_alpha_scales()]
    stream = _load_gate(STREAM_GATE)
    inject = _load_gate(INJECT_GATE)
    tension = _load_gate(TENSION_GATE)
    alpha0 = _load_gate(ALPHA0_GATE)
    checks.append({"id": "PRE1", "pass": bool(stream and stream.get("pass")), "detail": str(STREAM_GATE)})
    checks.append({"id": "PRE2", "pass": bool(inject and inject.get("pass")), "detail": str(INJECT_GATE)})
    checks.append({"id": "PRE3", "pass": bool(tension and tension.get("pass")), "detail": str(TENSION_GATE)})
    checks.append({"id": "PRE4", "pass": bool(alpha0 and alpha0.get("pass")), "detail": str(ALPHA0_GATE)})
    checks.append({"id": "PRE5", "pass": len(alphas) >= 3, "detail": f"grid={alphas}"})
    for slug in (*PANIC_SLUGS_FIT, *PANIC_SLUGS_CONFIRM):
        stub = QWEN_CODE / slug / "main.py"
        text = stub.read_text(encoding="utf-8") if stub.is_file() else ""
        checks.append(
            {
                "id": f"PRE-STUB-{slug}",
                "pass": "NotImplementedError" in text,
                "detail": str(stub),
            }
        )
    return checks, sabotage


def _finalize_gate(
    *,
    alphas: list[float],
    fit_rows: list[dict],
    selected: float | None,
    confirm: dict | None,
    control: dict | None,
    stability: dict | None,
    checks: list[dict],
    sabotage: list[dict],
) -> int:
    checkpoint = _load_checkpoint()
    checkpoint.update(
        {
            "alphas": alphas,
            "fit_rows": fit_rows,
            "selected": selected,
            "confirm": confirm,
            "control": control,
            "stability": stability,
        }
    )
    _save_checkpoint(checkpoint)

    checks.append(
        {
            "id": "FIT-GRID",
            "pass": len([r for r in fit_rows if r.get("complete")]) >= 3,
            "detail": f"n={len(fit_rows)}",
        }
    )
    checks.append(
        {
            "id": "SEL1",
            "pass": selected is not None and (selected <= MAX_ALPHA_CAP_4B + 1e-9),
            "detail": f"selected_alpha_0={selected}",
        }
    )
    if confirm:
        for row in confirm.get("rows", []):
            slug_id = row["slug"].split("_")[1]
            checks.append(
                {
                    "id": f"CONFIRM-{slug_id}",
                    "pass": bool(row.get("pass")),
                    "detail": str(row.get("quality", {}).get("reasons", [])),
                }
            )
        checks.append(
            {
                "id": "CONFIRM-ALL",
                "pass": bool(confirm.get("pass")),
                "detail": f"alpha={confirm.get('alpha_0')}",
            }
        )
    if control:
        checks.append({"id": "CTRL-SILENT", "pass": bool(control.get("pass")), "detail": "inject=0"})
        checks.append(
            {
                "id": "CTRL-OVERSTEER",
                "pass": not bool(control.get("oversteer")),
                "detail": "inject=1 worse than inject=0 on quality",
            }
        )
    if stability:
        checks.append(
            {
                "id": "STAB-p09",
                "pass": bool(stability.get("pass")),
                "detail": str(stability.get("row", {}).get("session")),
            }
        )

    passed = all(c["pass"] for c in checks) and sabotage[0]["pass"]
    payload = write_gate(
        OUT_PATH,
        gate="caa_alpha0_4b",
        pass_=passed,
        checks=checks,
        sabotage=sabotage,
        extra={
            "alphas": alphas,
            "selected_alpha_0": selected,
            "sweep_json": str(SWEEP_JSON),
            "fit_rows": fit_rows,
            "confirm": confirm,
            "control": control,
            "stability": stability,
        },
    )
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Live CAST alpha_0 Phase 4b (quality-gated)")
    ap.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--confirm-only", action="store_true")
    ap.add_argument("--confirm-alpha", type=float, default=None)
    ap.add_argument(
        "--confirm-run-slugs",
        default="",
        help="Comma-separated confirm suffixes to run live (e.g. p30)",
    )
    ap.add_argument(
        "--rescore-slugs",
        default="",
        help="Comma-separated confirm suffixes to re-score from snapshot only (e.g. p65)",
    )
    ap.add_argument("--skip-midpoint", action="store_true")
    ap.add_argument("--no-elicit", action="store_true")
    ap.add_argument("--confirm-no-inject", action="store_true")
    ap.add_argument("--skip-control-stability", action="store_true")
    ap.add_argument("--diag-out", type=Path, default=None)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--aider-timeout-s", type=float, default=1200.0)
    args = ap.parse_args()

    alphas = sorted({float(x.strip()) for x in args.alphas.split(",") if x.strip()})
    if not alphas:
        raise SystemExit("empty --alphas grid")

    checks, sabotage = _pre_checks(alphas)
    if args.dry_run:
        passed = all(c["pass"] for c in checks) and sabotage[0]["pass"]
        payload = write_gate(
            OUT_PATH,
            gate="caa_alpha0_4b",
            pass_=passed,
            checks=checks + [{"id": "DRY", "pass": True, "detail": "dry-run only"}],
            sabotage=sabotage,
            extra={"alphas": alphas, "dry_run": True},
        )
        print(json.dumps(payload, indent=2))
        return 0 if passed else 1

    if not all(c["pass"] for c in checks):
        write_gate(
            OUT_PATH,
            gate="caa_alpha0_4b",
            pass_=False,
            checks=checks,
            sabotage=sabotage,
            extra={"alphas": alphas},
        )
        return 1

    checkpoint = _load_checkpoint()
    if args.resume or args.fit_only or args.confirm_only:
        fit_rows = [r for r in checkpoint.get("fit_rows", []) if r.get("complete")]
    else:
        fit_rows = []
    done = {float(r["alpha_0"]) for r in fit_rows if float(r["alpha_0"]) in alphas}

    if args.confirm_only:
        inject = not args.confirm_no_inject
        elicit = not args.no_elicit
        selected = (
            float(args.confirm_alpha)
            if args.confirm_alpha is not None
            else select_alpha_0_4b(fit_rows)
        )
        if selected is None:
            if args.diag_out:
                payload = _write_diag_out(
                    args.diag_out,
                    alpha_0=-1.0,
                    inject=inject,
                    elicit=elicit,
                    confirm={"pass": False, "rows": []},
                )
                print(json.dumps(payload, indent=2))
                return 0
            return _finalize_gate(
                alphas=alphas,
                fit_rows=fit_rows,
                selected=None,
                confirm=None,
                control=None,
                stability=None,
                checks=checks,
                sabotage=sabotage,
            )
        run_tasks = _tasks_by_suffix(args.confirm_run_slugs, CONFIRM_TASKS)
        rescore_tasks = _tasks_by_suffix(args.rescore_slugs, CONFIRM_TASKS)
        if not run_tasks and not rescore_tasks:
            raise SystemExit("confirm-only requires --confirm-run-slugs and/or --rescore-slugs")

        selected_row = next((r for r in fit_rows if float(r["alpha_0"]) == selected), None)
        fit_inject_rows: list[dict] = []
        if selected_row:
            fit_inject_rows = [
                {
                    "slug": panic["slug"],
                    "quality_ok": panic.get("quality_ok", True),
                    "pass": panic.get("pass"),
                }
                for panic in selected_row.get("panics", [])
            ]

        confirm = _run_confirm_at_alpha(
            selected,
            max_retries=args.max_retries,
            aider_timeout_s=args.aider_timeout_s,
            run_tasks=run_tasks,
            rescore_tasks=rescore_tasks,
            inject=inject,
            elicit=elicit,
        )
        control = None
        stability = None
        if not args.skip_control_stability:
            control = _run_control_at_alpha(
                selected,
                max_retries=args.max_retries,
                aider_timeout_s=args.aider_timeout_s,
                fit_inject_rows=fit_inject_rows,
            )
            stability = _run_stability_at_alpha(
                selected,
                max_retries=args.max_retries,
                aider_timeout_s=args.aider_timeout_s,
            )
        if args.diag_out:
            payload = _write_diag_out(
                args.diag_out,
                alpha_0=selected,
                inject=inject,
                elicit=elicit,
                confirm=confirm,
            )
            print(json.dumps(payload, indent=2))
            return 0
        return _finalize_gate(
            alphas=alphas,
            fit_rows=fit_rows,
            selected=selected,
            confirm=confirm,
            control=control,
            stability=stability,
            checks=checks,
            sabotage=sabotage,
        )

    for alpha_0 in alphas:
        if args.resume and alpha_0 in done:
            continue
        try:
            row = _run_fit_alpha_row(
                alpha_0,
                max_retries=args.max_retries,
                aider_timeout_s=args.aider_timeout_s,
                checkpoint=checkpoint,
            )
            fit_rows = [r for r in fit_rows if float(r.get("alpha_0", -1)) != alpha_0] + [row]
            checkpoint["fit_rows"] = fit_rows
            _save_checkpoint(checkpoint)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR fit alpha_0={alpha_0}: {exc}", flush=True)
            traceback.print_exc()
            fit_rows.append({"alpha_0": alpha_0, "pass": False, "complete": False, "error": str(exc)})
            checkpoint["fit_rows"] = fit_rows
            _save_checkpoint(checkpoint)
            stop_server(settle_s=3.0)

    if not args.skip_midpoint:
        mid = _binary_midpoint_alpha(fit_rows, alphas)
        if mid is not None and (not args.resume or mid not in done):
            print(f"\n=== MIDPOINT alpha_0={mid} ===", flush=True)
            try:
                row = _run_fit_alpha_row(
                    mid,
                    max_retries=args.max_retries,
                    aider_timeout_s=args.aider_timeout_s,
                    checkpoint=checkpoint,
                )
                fit_rows = [r for r in fit_rows if float(r.get("alpha_0", -1)) != mid] + [row]
                checkpoint["fit_rows"] = fit_rows
                _save_checkpoint(checkpoint)
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR midpoint alpha_0={mid}: {exc}", flush=True)
                traceback.print_exc()

    selected = select_alpha_0_4b(fit_rows)
    print(f"Selected alpha_0={selected}", flush=True)
    complete_fit = [r for r in fit_rows if r.get("complete")]

    if args.fit_only:
        return _finalize_gate(
            alphas=alphas,
            fit_rows=complete_fit,
            selected=selected,
            confirm=None,
            control=None,
            stability=None,
            checks=checks,
            sabotage=sabotage,
        )

    if selected is None:
        return _finalize_gate(
            alphas=alphas,
            fit_rows=complete_fit,
            selected=None,
            confirm=None,
            control=None,
            stability=None,
            checks=checks,
            sabotage=sabotage,
        )

    selected_row = next((r for r in fit_rows if float(r["alpha_0"]) == selected), None)
    fit_inject_rows: list[dict] = []
    if selected_row:
        fit_inject_rows = [
            {
                "slug": p["slug"],
                "quality_ok": p.get("quality_ok", True),
                "pass": p.get("pass"),
            }
            for p in selected_row.get("panics", [])
        ]

    confirm = _run_confirm_at_alpha(
        selected,
        max_retries=args.max_retries,
        aider_timeout_s=args.aider_timeout_s,
    )
    if not confirm["pass"]:
        lower = _next_lower_passing_alpha(selected, fit_rows)
        if lower is not None:
            print(f"Confirm failed at {selected}; step-down to {lower}", flush=True)
            selected = lower
            confirm = _run_confirm_at_alpha(
                lower,
                max_retries=args.max_retries,
                aider_timeout_s=args.aider_timeout_s,
            )

    control = _run_control_at_alpha(
        selected,
        max_retries=args.max_retries,
        aider_timeout_s=args.aider_timeout_s,
        fit_inject_rows=fit_inject_rows,
    )
    stability = _run_stability_at_alpha(
        selected,
        max_retries=args.max_retries,
        aider_timeout_s=args.aider_timeout_s,
    )

    return _finalize_gate(
        alphas=alphas,
        fit_rows=complete_fit,
        selected=selected,
        confirm=confirm,
        control=control,
        stability=stability,
        checks=checks,
        sabotage=sabotage,
    )


if __name__ == "__main__":
    raise SystemExit(main())
