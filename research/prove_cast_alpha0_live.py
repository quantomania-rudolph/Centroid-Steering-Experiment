#!/usr/bin/env python
"""Phase 4: live alpha_0 sweep with inject on (UNI_CAA01).

Prerequisites (fail-closed):
  - cast_stream_live_gate.json pass=true
  - cast_inject_live_gate.json pass=true
  - caa_cast_tension_gate.json pass=true

For each alpha in the sweep grid, restarts the server with CAA_ALPHA_0 set,
runs p09 + p19 panic Aider sessions and one clean session, scores inject/demo,
selects the highest passing alpha_0, and writes artifacts/gates/caa_alpha0_live_gate.json.

Does not modify metadata.json unless --apply is passed and the gate passes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.caa.alpha0_gate import (  # noqa: E402
    demo_matches,
    inject_stats,
    sabotage_alpha_scales,
    score_sweep_row,
    select_alpha_0,
)
from jlens_qwen.caa.cast_panic_stubs import panic_message, restore_panic_stub  # noqa: E402
from jlens_qwen.caa.cast_stream_gate import chaos_tokens_from_session  # noqa: E402
from jlens_qwen.caa.gate_io import write_gate  # noqa: E402
from unicaa01_restart_server import fetch_health, restart_server, stop_server  # noqa: E402

GATES = ROOT / "artifacts" / "gates"
OUT = ROOT / "experiments" / "out"
QWEN_CODE = ROOT / "qwen code"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
META = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "metadata.json"
OUT_PATH = GATES / "caa_alpha0_live_gate.json"
SWEEP_JSON = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "calibration" / "alpha0_sweep_live.json"
STREAM_GATE = GATES / "cast_stream_live_gate.json"
INJECT_GATE = GATES / "cast_inject_live_gate.json"
TENSION_GATE = GATES / "caa_cast_tension_gate.json"
DEFAULT_ALPHAS = [0.35, 0.55, 0.7, 0.85]

PANIC_TASKS = [
    {
        "slug": "unicaa01_p09_panic",
        "suffix": "p09",
        "expected_demo": [1, 2, 3],
    },
    {
        "slug": "unicaa01_p19_panic",
        "suffix": "p19",
        "expected_demo": 1,
    },
]


def _alpha_tag(alpha_0: float) -> str:
    return f"a{int(round(alpha_0 * 100)):03d}"


def _experiment(alpha_0: float, suffix: str) -> str:
    return f"cast_alpha0_{_alpha_tag(alpha_0)}_{suffix}"


def _load_gate(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_checkpoint() -> dict:
    if not SWEEP_JSON.is_file():
        return {"alphas": [], "rows": [], "selected": None}
    return json.loads(SWEEP_JSON.read_text(encoding="utf-8"))


def _save_checkpoint(payload: dict) -> None:
    SWEEP_JSON.parent.mkdir(parents=True, exist_ok=True)
    SWEEP_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_aider(slug: str, experiment: str, message: str, *, aider_timeout_s: float) -> Path:
    cmd = [
        str(PY),
        str(ROOT / "experiments" / "run_aider_agent.py"),
        "--task-slug",
        slug,
        "--experiment",
        experiment,
        "--message",
        message,
        "--layers",
        "16",
        "--chaos-only",
        "--no-auto-lint",
        "--aider-timeout-s",
        str(aider_timeout_s),
    ]
    rc = subprocess.call(cmd, cwd=str(ROOT), env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    latest = OUT / f"{experiment}_latest.json"
    if not latest.is_file():
        raise RuntimeError(f"missing {latest} (aider rc={rc})")
    return latest


def _run_demo(task_dir: Path, expected: object) -> tuple[bool, object | None, str | None]:
    code = (
        "import json\n"
        "from main import demo\n"
        "v = demo()\n"
        "print(json.dumps({'ok': True, 'value': v}))\n"
    )
    proc = subprocess.run(
        [str(PY), "-c", code],
        cwd=str(task_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        return False, None, (proc.stderr or proc.stdout or "demo subprocess failed").strip()
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        value = payload.get("value")
    except (json.JSONDecodeError, IndexError):
        return False, None, f"unparseable demo output: {proc.stdout!r}"
    return demo_matches(value, expected), value, None


def _stub_mean(tokens: list[dict]) -> float:
    rates = [float((t.get("B_vocab") or {}).get("stub_mass") or 0) for t in tokens]
    return statistics.mean(rates) if rates else 0.0


def _restore_panics() -> None:
    for task in PANIC_TASKS:
        restore_panic_stub(QWEN_CODE / task["slug"], task["slug"])


def _verify_health_alpha(alpha_0: float, *, retries: int = 3) -> tuple[bool, dict]:
    last: dict = {}
    for attempt in range(retries):
        health = fetch_health() or {}
        last = health
        ok = (
            health.get("ok")
            and health.get("cast_shadow")
            and health.get("cast_inject")
            and str(health.get("device", "")).startswith("xpu")
            and abs(float(health.get("cast_alpha_0") or -1) - alpha_0) < 1e-6
        )
        if ok:
            return True, health
        time.sleep(3)
    return False, last


def _panic_needs_retry(stats: dict[str, Any], demo_ok: bool) -> bool:
    if not demo_ok:
        return True
    if stats.get("inject_count", 0) < 1:
        return True
    if stats.get("g_stream_count", 0) < 1:
        return True
    return False


def _run_panic_with_retries(
    *,
    alpha_0: float,
    task: dict,
    max_retries: int,
    aider_timeout_s: float,
) -> dict:
    slug = task["slug"]
    experiment = _experiment(alpha_0, task["suffix"])
    last_row: dict | None = None
    for attempt in range(max_retries + 1):
        restore_panic_stub(QWEN_CODE / slug, slug)
        session_path = _run_aider(slug, experiment, panic_message(slug), aider_timeout_s=aider_timeout_s)
        tokens = chaos_tokens_from_session(session_path)
        stats = inject_stats(tokens)
        demo_ok, demo_value, demo_err = _run_demo(QWEN_CODE / slug, task["expected_demo"])
        last_row = {
            "slug": slug,
            "stats": stats,
            "demo_ok": demo_ok,
            "demo_value": demo_value,
            "demo_err": demo_err,
            "session": str(session_path),
            "attempt": attempt + 1,
        }
        if not _panic_needs_retry(stats, demo_ok):
            break
        print(
            f"  {slug} retry {attempt + 1}/{max_retries}: "
            f"inject={stats.get('inject_count')} g_stream={stats.get('g_stream_count')} demo_ok={demo_ok}",
            flush=True,
        )
    assert last_row is not None
    return last_row


def _run_alpha_row(
    alpha_0: float,
    *,
    max_retries: int,
    aider_timeout_s: float,
    checkpoint: dict,
) -> dict:
    print(f"\n=== alpha_0={alpha_0} ===", flush=True)
    if not restart_server(device="xpu", shadow=True, inject=True, alpha_0=alpha_0):
        raise RuntimeError(f"server restart failed for alpha_0={alpha_0}")
    health_ok, health = _verify_health_alpha(alpha_0)
    if not health_ok:
        stop_server()
        raise RuntimeError(f"health mismatch for alpha_0={alpha_0}: {health}")

    panic_rows = []
    for task in PANIC_TASKS:
        row = _run_panic_with_retries(
            alpha_0=alpha_0,
            task=task,
            max_retries=max_retries,
            aider_timeout_s=aider_timeout_s,
        )
        panic_rows.append(row)
        stats = row["stats"]
        print(
            f"  {row['slug']}: inject={stats.get('inject_count')} "
            f"g_stream={stats.get('g_stream_count')} demo_ok={row['demo_ok']} "
            f"attempts={row.get('attempt')}",
            flush=True,
        )
        checkpoint["rows"] = [
            r for r in checkpoint.get("rows", []) if r.get("alpha_0") != alpha_0
        ] + [{"alpha_0": alpha_0, "partial": True, "panics": panic_rows}]
        _save_checkpoint(checkpoint)

    clean_experiment = _experiment(alpha_0, "clean")
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
        f"mean_alpha={clean_stats.get('mean_alpha'):.4f}",
        flush=True,
    )

    row = score_sweep_row(
        alpha_0=alpha_0,
        panics=panic_rows,
        clean={
            "stats": clean_stats,
            "stub_mean": clean_stub,
            "session": str(clean_path),
        },
    )
    row["health"] = health
    row["complete"] = True
    return row


def _apply_alpha_0(selected: float, *, sweep_rows: list[dict]) -> None:
    backup = META.with_suffix(".alpha0_backup.json")
    shutil.copy2(META, backup)
    meta = json.loads(META.read_text(encoding="utf-8"))
    meta["alpha_0"] = selected
    meta["alpha_0_provenance"] = {
        "source": "prove_cast_alpha0_live",
        "selected_alpha_0": selected,
        "sweep_gate": str(OUT_PATH),
        "sweep_rows": sweep_rows,
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Applied alpha_0={selected} to {META} (backup {backup})", flush=True)


def _finalize_gate(
    *,
    alphas: list[float],
    sweep_rows: list[dict],
    checks: list[dict],
    sabotage: list[dict],
    apply: bool,
) -> int:
    selected = select_alpha_0(sweep_rows)
    checkpoint = {
        "alphas": alphas,
        "rows": sweep_rows,
        "selected": selected,
    }
    _save_checkpoint(checkpoint)

    checks.append(
        {
            "id": "SEL1",
            "pass": selected is not None,
            "detail": f"selected_alpha_0={selected}",
        }
    )
    if selected is not None:
        selected_row = next(r for r in sweep_rows if r["alpha_0"] == selected)
        for panic in selected_row["panics"]:
            slug = panic["slug"]
            checks.append(
                {
                    "id": f"PANIC-{slug}",
                    "pass": panic["pass"],
                    "detail": (
                        f"inject={panic['stats'].get('inject_count')} "
                        f"demo_ok={panic['demo_ok']}"
                    ),
                }
            )
        checks.append(
            {
                "id": "CLEAN",
                "pass": selected_row["clean"]["pass"],
                "detail": str(selected_row["clean"]["stats"]),
            }
        )

    passed = all(c["pass"] for c in checks) and sabotage[0]["pass"]
    payload = write_gate(
        OUT_PATH,
        gate="caa_alpha0_live",
        pass_=passed,
        checks=checks,
        sabotage=sabotage,
        extra={
            "alphas": alphas,
            "selected_alpha_0": selected,
            "sweep_json": str(SWEEP_JSON),
            "sweep_rows": sweep_rows,
        },
    )
    print(json.dumps(payload, indent=2))

    if passed and apply and selected is not None:
        _apply_alpha_0(selected, sweep_rows=sweep_rows)

    return 0 if passed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Live CAST alpha_0 sweep (Phase 4)")
    ap.add_argument(
        "--alphas",
        default=",".join(str(a) for a in DEFAULT_ALPHAS),
        help="Comma-separated alpha_0 grid (default: 0.35,0.55,0.7,0.85)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate prerequisites and grid only; do not restart or run Aider",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip alphas already marked complete in alpha0_sweep_live.json",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Write selected alpha_0 to metadata.json when gate passes",
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per panic task when inject/demo fails (default 2)",
    )
    ap.add_argument(
        "--aider-timeout-s",
        type=float,
        default=1200.0,
        help="Per-Aider timeout in seconds (default 1200)",
    )
    args = ap.parse_args()

    alphas = sorted({float(x.strip()) for x in args.alphas.split(",") if x.strip()})
    if not alphas:
        raise SystemExit("empty --alphas grid")

    checks: list[dict] = []
    sabotage = [sabotage_alpha_scales()]

    stream = _load_gate(STREAM_GATE)
    inject = _load_gate(INJECT_GATE)
    tension = _load_gate(TENSION_GATE)
    checks.append({"id": "PRE1", "pass": bool(stream and stream.get("pass")), "detail": str(STREAM_GATE)})
    checks.append({"id": "PRE2", "pass": bool(inject and inject.get("pass")), "detail": str(INJECT_GATE)})
    checks.append({"id": "PRE3", "pass": bool(tension and tension.get("pass")), "detail": str(TENSION_GATE)})
    checks.append({"id": "PRE4", "pass": len(alphas) >= 1, "detail": f"grid={alphas}"})

    if args.dry_run:
        passed = all(c["pass"] for c in checks) and sabotage[0]["pass"]
        payload = write_gate(
            OUT_PATH,
            gate="caa_alpha0_live",
            pass_=passed,
            checks=checks + [{"id": "DRY", "pass": True, "detail": "dry-run only"}],
            sabotage=sabotage,
            extra={"alphas": alphas, "dry_run": True},
        )
        print(json.dumps(payload, indent=2))
        return 0 if passed else 1

    if not all(c["pass"] for c in checks):
        payload = write_gate(
            OUT_PATH,
            gate="caa_alpha0_live",
            pass_=False,
            checks=checks,
            sabotage=sabotage,
            extra={"alphas": alphas},
        )
        print(json.dumps(payload, indent=2))
        return 1

    checkpoint = _load_checkpoint()
    existing_rows = [
        r for r in checkpoint.get("rows", []) if r.get("complete") and r.get("alpha_0") in alphas
    ]
    done_alphas = {float(r["alpha_0"]) for r in existing_rows}

    sweep_rows: list[dict] = list(existing_rows) if args.resume else []
    if args.resume:
        print(f"Resume: skipping complete alphas {sorted(done_alphas)}", flush=True)

    for alpha_0 in alphas:
        if args.resume and alpha_0 in done_alphas:
            continue
        try:
            row = _run_alpha_row(
                alpha_0,
                max_retries=args.max_retries,
                aider_timeout_s=args.aider_timeout_s,
                checkpoint=checkpoint,
            )
            sweep_rows = [r for r in sweep_rows if r.get("alpha_0") != alpha_0] + [row]
            checkpoint["rows"] = sweep_rows
            checkpoint["alphas"] = alphas
            _save_checkpoint(checkpoint)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR alpha_0={alpha_0}: {exc}", flush=True)
            traceback.print_exc()
            sweep_rows.append(
                {
                    "alpha_0": alpha_0,
                    "pass": False,
                    "complete": False,
                    "error": str(exc),
                }
            )
            checkpoint["rows"] = sweep_rows
            _save_checkpoint(checkpoint)
            stop_server(settle_s=3.0)

    complete_rows = [r for r in sweep_rows if r.get("complete")]
    return _finalize_gate(
        alphas=alphas,
        sweep_rows=complete_rows,
        checks=checks,
        sabotage=sabotage,
        apply=args.apply,
    )


if __name__ == "__main__":
    raise SystemExit(main())
