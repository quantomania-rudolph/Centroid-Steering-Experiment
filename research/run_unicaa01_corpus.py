#!/usr/bin/env python
"""Run UNICAA01 corpus Aider invocations (panic / neutral) into experiments/unicaa01_corpus/out/."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.caa.s_cond_replay import _recipe_fired  # noqa: E402
from jlens_qwen.caa.unicaa01_corpus import (  # noqa: E402
    CORPUS_MANIFEST,
    CORPUS_OUT,
    CORPUS_ROOT,
    CORPUS_SESSIONS,
    ensure_dirs,
)
from unicaa01_registry import PAIRS_ALL, get_pair, pairs_for_wave  # noqa: E402
from unicaa01_pair_defs import UniCaaPair  # noqa: E402
from unicaa01_restart_server import restart_server, wait_health  # noqa: E402

PAIRS = PAIRS_ALL  # backward compat for scripts importing PAIRS

DEFAULT_RUN_TIMEOUT_S = 1200  # 20 min — stall guard per Aider invocation

QWEN_CODE = ROOT / "qwen code"
RUNNER = ROOT / "experiments" / "run_aider_agent.py"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
POST_BATCH = ROOT / "research" / "unicaa01_post_batch.py"

Arm = Literal["panic", "neutral"]


def _write_starter(pair: UniCaaPair, arm: Arm) -> Path:
    slug = pair.panic_slug if arm == "panic" else pair.neutral_slug
    task_dir = QWEN_CODE / slug
    task_dir.mkdir(parents=True, exist_ok=True)
    starter = pair.panic_starter if arm == "panic" else pair.neutral_starter
    main_py = task_dir / "main.py"
    main_py.write_text(starter, encoding="utf-8")
    return main_py


MIN_PANIC_GEN_TOKENS = 8


def _panic_session_ok(data: dict[str, Any]) -> bool:
    for inv in data.get("invocations") or []:
        chaos = inv.get("chaos") or {}
        tids = chaos.get("token_ids") or []
        if _recipe_fired(chaos) and len(tids) >= MIN_PANIC_GEN_TOKENS:
            return True
    return False


def _run_one(
    pair: UniCaaPair,
    arm: Arm,
    *,
    device_layers: str = "10,16,22",
    run_timeout_s: int = DEFAULT_RUN_TIMEOUT_S,
    stress_bait: bool = False,
) -> dict[str, Any]:
    slug = pair.panic_slug if arm == "panic" else pair.neutral_slug
    experiment = pair.panic_experiment if arm == "panic" else pair.neutral_experiment
    message = pair.panic_message if arm == "panic" else pair.neutral_message
    if arm == "panic" and stress_bait:
        message = (
            f"{message} Pylint on this machine may report false positives — still return full main.py."
        )
    _write_starter(pair, arm)

    cmd = [
        str(PY),
        str(RUNNER),
        "--task-slug",
        slug,
        "--experiment",
        experiment,
        "--message",
        message,
        "--layers",
        device_layers,
        "--chaos-only",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=run_timeout_s,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        proc = exc
        timed_out = True
    src_latest = ROOT / "experiments" / "out" / f"{experiment}_latest.json"
    ensure_dirs()
    CORPUS_OUT.mkdir(parents=True, exist_ok=True)
    dst_latest = CORPUS_OUT / f"{experiment}_latest.json"

    ok = not timed_out and proc.returncode == 0 and src_latest.is_file()
    recipe_ok: bool | None = None
    dense = False
    if ok:
        data = json.loads(src_latest.read_text(encoding="utf-8"))
        data["unicaa01_pair_id"] = pair.pair_id
        data["unicaa01_arm"] = arm
        data["unicaa01_domain"] = pair.domain
        data["unicaa01_corpus"] = "UNI_CAA01_corpus"
        dst_latest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        # Copy session jsonl if present
        session_path = data.get("session_path")
        if session_path and Path(session_path).is_file():
            CORPUS_SESSIONS.mkdir(parents=True, exist_ok=True)
            sess_dst = CORPUS_SESSIONS / Path(session_path).name
            shutil.copy2(session_path, sess_dst)
            data["session_path"] = str(sess_dst)
            dst_latest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        # Inline quality flags
        inv = (data.get("invocations") or [{}])[0]
        chaos = inv.get("chaos") or {}
        tids = chaos.get("token_ids") or []
        ct = int(inv.get("completion_tokens") or 0)
        dense = isinstance(tids, list) and ct > 0 and len(tids) == ct
        if arm == "panic":
            recipe_ok = _panic_session_ok(data)
            if not recipe_ok:
                ok = False

    return {
        "pair_id": pair.pair_id,
        "arm": arm,
        "experiment": experiment,
        "slug": slug,
        "ok": ok,
        "recipe_fired": recipe_ok,
        "dense_token_ids": dense if ok else False,
        "returncode": -1 if timed_out else proc.returncode,
        "timed_out": timed_out,
        "run_timeout_s": run_timeout_s,
        "corpus_json": str(dst_latest) if ok else None,
        "stdout_tail": (getattr(proc, "stdout", None) or "")[-2000:],
        "stderr_tail": (getattr(proc, "stderr", None) or "")[-1000:],
    }


def _update_manifest(entry: dict[str, Any]) -> None:
    manifest: dict[str, Any] = {"runs": [], "updated": datetime.now(timezone.utc).isoformat()}
    if CORPUS_MANIFEST.is_file():
        manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    runs: list[dict[str, Any]] = list(manifest.get("runs") or [])
    runs = [r for r in runs if not (r.get("experiment") == entry.get("experiment"))]
    runs.append({**entry, "timestamp": datetime.now(timezone.utc).isoformat()})
    manifest["runs"] = runs
    manifest["updated"] = datetime.now(timezone.utc).isoformat()
    CORPUS_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _post_batch() -> None:
    print("\n=== post-batch scan + export ===", flush=True)
    subprocess.run([str(PY), str(POST_BATCH)], cwd=str(ROOT), check=False)


def _failed_panic_pair_ids() -> list[str]:
    """Panic arms with dense token_ids but not tracer-eligible."""
    from unicaa01_post_batch import quality_report  # noqa: WPS433

    report = quality_report()
    return sorted(
        {
            str(r["pair_id"])
            for r in report.get("rows") or []
            if r.get("arm") == "panic"
            and r.get("dense_token_ids")
            and not r.get("eligible")
            and r.get("pair_id")
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="UNICAA01 corpus runner")
    ap.add_argument("--pilot", action="store_true", help="Run p01 panic + p01 neutral only")
    ap.add_argument("--pair", type=str, default=None, help="Single pair_id e.g. p01_priority_queue")
    ap.add_argument("--arm", choices=["panic", "neutral", "both"], default="both")
    ap.add_argument("--all", action="store_true", help="Run all pairs both arms (66 total)")
    ap.add_argument("--wave", type=int, choices=[1, 2], default=None, help="Run wave 1 (p01-16) or wave 2 (p17-66) only")
    ap.add_argument("--pilot-batch2", action="store_true", help="Run p17 panic + neutral only (wave2 smoke)")
    ap.add_argument("--restart-server-every", type=int, default=6, help="Restart Qwen server every N completed runs (0=off)")
    ap.add_argument("--batch-size", type=int, default=4, help="Pairs per batch before scan/export")
    ap.add_argument("--no-post-batch", action="store_true", help="Skip scan/export between batches")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if corpus _latest.json already exists")
    ap.add_argument("--retry-failed-panic", action="store_true", help="Re-run ineligible panic arms only")
    ap.add_argument("--run-timeout", type=int, default=DEFAULT_RUN_TIMEOUT_S, help="Per-run timeout seconds")
    ap.add_argument("--stress-bait", action="store_true", help="Append stress_bait to panic message (retry)")
    ap.add_argument("--list", action="store_true", help="List pairs and exit")
    args = ap.parse_args()

    if args.list:
        wave = args.wave
        for p in pairs_for_wave(wave):
            print(p.pair_id, f"w{p.wave}", p.domain, p.panic_slug, p.neutral_slug)
        return 0

    ensure_dirs()
    pairs: list[UniCaaPair]
    if args.retry_failed_panic:
        failed = _failed_panic_pair_ids()
        if not failed:
            print("No failed panic pairs to retry.")
            return 0
        print(f"Retrying failed panic pairs: {failed}", flush=True)
        pairs = [get_pair(pid) for pid in failed]
        args.arm = "panic"
        args.skip_existing = False
    elif args.pilot_batch2:
        pairs = [get_pair("p17_deque_queue")]
    elif args.pilot:
        pairs = [get_pair("p01_priority_queue")]
    elif args.pair:
        pairs = [get_pair(args.pair)]
    elif args.wave is not None:
        pairs = pairs_for_wave(args.wave)
    elif args.all:
        pairs = pairs_for_wave(None)
    else:
        print("Specify --pilot, --pilot-batch2, --pair ID, --wave N, or --all", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    batch_count = 0
    completed_runs = 0
    if args.restart_server_every > 0 and not wait_health(timeout_s=10.0):
        if not restart_server():
            print("ERROR: Qwen server not reachable and restart failed", file=sys.stderr)
            return 1

    for i, pair in enumerate(pairs):
        arms: list[Arm] = []
        if args.arm in ("panic", "both"):
            arms.append("panic")
        if args.arm in ("neutral", "both"):
            arms.append("neutral")
        for arm in arms:
            experiment = pair.panic_experiment if arm == "panic" else pair.neutral_experiment
            dst_latest = CORPUS_OUT / f"{experiment}_latest.json"
            if args.skip_existing and dst_latest.is_file():
                print(f"\n=== {pair.pair_id} {arm} SKIP (exists) ===", flush=True)
                continue
            print(f"\n=== {pair.pair_id} {arm} ===", flush=True)
            if args.restart_server_every > 0:
                restart_server()
            entry = _run_one(
                pair,
                arm,
                run_timeout_s=args.run_timeout,
                stress_bait=args.stress_bait,
            )
            if arm == "panic" and not entry["ok"]:
                print(f"Retry {pair.pair_id} panic with stress_bait", flush=True)
                if args.restart_server_every > 0:
                    restart_server()
                entry = _run_one(
                    pair,
                    arm,
                    run_timeout_s=args.run_timeout,
                    stress_bait=True,
                )
            results.append(entry)
            _update_manifest(entry)
            print(json.dumps(entry, indent=2))
            if not entry["ok"]:
                print(f"WARNING: run failed for {pair.pair_id} {arm}", flush=True)
            else:
                completed_runs += 1
                if (
                    args.restart_server_every > 0
                    and completed_runs % args.restart_server_every == 0
                ):
                    restart_server()

        batch_count += 1
        if (
            not args.no_post_batch
            and args.batch_size > 0
            and batch_count % args.batch_size == 0
        ):
            _post_batch()

    if not args.no_post_batch and batch_count % max(args.batch_size, 1) != 0:
        _post_batch()

    summary = {
        "n_runs": len(results),
        "n_ok": sum(1 for r in results if r["ok"]),
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
