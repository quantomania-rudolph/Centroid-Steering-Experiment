#!/usr/bin/env python
"""Phase A: build clean + held-out panic s_cond corpora (teacher-forced L16 replay)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.caa.calibration_deny import is_denied_panic_task, load_deny_list  # noqa: E402
from jlens_qwen.caa.s_cond_replay import (  # noqa: E402
    default_calibration_experiment_dirs,
    discover_clean_sessions,
    discover_held_out_panic_sessions,
    load_u_cond,
    relabel_panic_agree_windows,
    relabel_panic_windows,
    replay_invocation,
    write_replay_jsonl,
)
from jlens_qwen.caa.cast_math import load_cast_thresholds  # noqa: E402
from jlens_qwen.caa.stream_dynamics import non_resolving_from_temporal  # noqa: E402
from jlens_qwen.model_loader import load_qwen_coder  # noqa: E402
from unicaa01_restart_server import stop_server  # noqa: E402

DEFAULT_META = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "metadata.json"
DEFAULT_COND = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "condition_L16.pt"
CALIB_DIR = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "calibration"
CORPUS_OUT = ROOT / "experiments" / "unicaa01_corpus" / "out"
EXPERIMENTS = ROOT / "experiments" / "out"
G42_MIN_PANIC_TARGET = 15  # aspirational inventory for full 12+3 split
G42_MIN_THICK_PANIC = 12  # 9 fit + 3 verify after weak-geometry exclusion
G42_MIN_VERIFY_ELIGIBLE = 3
G42_MIN_VERIFY_WINDOW = 2
G42_MIN_CLEAN_RUNS = 20


def _dedupe_clean_by_task(
    candidates: list[tuple[Path, int]],
) -> list[tuple[Path, int]]:
    """Keep one invocation per neutral task file (longest already ranked first)."""
    seen: set[str] = set()
    out: list[tuple[Path, int]] = []
    for path, inv_idx in candidates:
        slug = path.stem
        if slug in seen:
            continue
        seen.add(slug)
        out.append((path, inv_idx))
    return out


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _inv_by_index(data: dict, idx: int) -> dict:
    for inv in data.get("invocations") or []:
        if int(inv.get("invocation_index") or 0) == idx:
            return inv
    invs = data.get("invocations") or []
    if not invs:
        raise ValueError("no invocations")
    return invs[min(idx - 1, len(invs) - 1)]


def _panic_episode_stats(path: Path) -> dict[str, int]:
    n_window = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("type") == "token" and obj.get("in_panic_window"):
            n_window += 1
    return {"n_window_tokens": n_window}


def _finalize_panic_corpus(
    panic_paths: list[Path],
    *,
    metadata_path: Path = DEFAULT_META,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Relabel chaos windows then agree-intersect; classify thick by chaos window size."""
    thresholds = load_cast_thresholds(metadata_path)
    thick: list[Path] = []
    thin: list[Path] = []
    weak: list[Path] = []
    for path in panic_paths:
        if not path.is_file():
            continue
        relabel_panic_windows(path)
        n_chaos = _panic_episode_stats(path)["n_window_tokens"]
        relabel_panic_agree_windows(
            path,
            tau=thresholds["tau"],
            sigma=thresholds["sigma"],
            tau_hard=thresholds["tau_hard"],
            gamma_floor=0.5,
        )
        n_agree = _panic_episode_stats(path)["n_window_tokens"]
        if n_chaos >= G42_MIN_VERIFY_WINDOW:
            if n_agree == 0:
                weak.append(path)
            else:
                thick.append(path)
        elif n_chaos > 0:
            thin.append(path)
        else:
            thin.append(path)
    return thick, thin, weak


def _gate(
    *,
    deny: dict,
    clean_files: list[Path],
    panic_files: list[Path],
    errors: list[str],
    n_clean_tokens: int,
    n_panic_episodes: int,
    n_non_resolving: int,
    n_verify_eligible: int,
    excluded_thin_panic: list[str] | None = None,
    excluded_weak_geometry: list[str] | None = None,
) -> dict:
    leaked: list[str] = []
    for p in panic_files:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("type") != "meta":
                continue
            if is_denied_panic_task(str(obj.get("task_slug") or ""), deny):
                leaked.append(obj.get("run_id", str(p)))
    pass_gate = (
        not errors
        and len(clean_files) >= G42_MIN_CLEAN_RUNS
        and n_clean_tokens >= 2000
        and n_panic_episodes >= G42_MIN_THICK_PANIC
        and n_non_resolving >= 1
        and n_verify_eligible >= G42_MIN_VERIFY_ELIGIBLE
        and not leaked
    )
    return {
        "gate": "G42_scond_corpus",
        "version": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pass": pass_gate,
        "n_clean_runs": len(clean_files),
        "n_clean_tokens": n_clean_tokens,
        "n_panic_episodes": n_panic_episodes,
        "n_verify_eligible": n_verify_eligible,
        "n_non_resolving": n_non_resolving,
        "errors": errors,
        "leaked_train_panic": leaked,
        "deny_n_pairs": deny.get("n_pairs"),
        "clean_paths": [str(p) for p in clean_files],
        "panic_paths": [str(p) for p in panic_files],
        "excluded_thin_panic": excluded_thin_panic or [],
        "excluded_weak_geometry": excluded_weak_geometry or [],
        "thresholds": {
            "min_clean_runs": G42_MIN_CLEAN_RUNS,
            "min_panic_episodes_target": G42_MIN_PANIC_TARGET,
            "min_thick_panic_episodes": G42_MIN_THICK_PANIC,
            "min_verify_eligible": G42_MIN_VERIFY_ELIGIBLE,
            "min_verify_window_tokens": G42_MIN_VERIFY_WINDOW,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build s_cond calibration corpora")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--metadata", type=Path, default=DEFAULT_META)
    ap.add_argument("--condition", type=Path, default=DEFAULT_COND)
    ap.add_argument("--experiments-dir", type=Path, default=None)
    ap.add_argument("--max-clean", type=int, default=30)
    ap.add_argument("--max-panic", type=int, default=25)
    ap.add_argument("--max-gen-tokens", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Skip runs with existing JSONL")
    ap.add_argument(
        "--rescore-panic",
        action="store_true",
        help="Re-replay panic JSONL even when --resume (after u_cond update)",
    )
    ap.add_argument(
        "--job",
        default=None,
        help="Single job: clean|panic,<session_json>,<inv_index>",
    )
    ap.add_argument("--finalize-only", action="store_true", help="Write G42 from on-disk JSONL")
    args = ap.parse_args()

    deny = load_deny_list(args.metadata)
    deny_path = CALIB_DIR / "deny_list.json"
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    deny_path.write_text(json.dumps(deny, indent=2) + "\n", encoding="utf-8")

    clean_dir = CALIB_DIR / "clean"
    panic_dir = CALIB_DIR / "panic"
    clean_dir.mkdir(parents=True, exist_ok=True)
    panic_dir.mkdir(parents=True, exist_ok=True)

    if args.experiments_dir is not None:
        exp_dirs = [args.experiments_dir]
    else:
        exp_dirs = default_calibration_experiment_dirs(ROOT)

    if args.finalize_only:
        clean_written = sorted(clean_dir.glob("*.jsonl"))
        panic_all = sorted(panic_dir.glob("*.jsonl"))
        panic_written, excluded_thin, excluded_weak = _finalize_panic_corpus(
            panic_all, metadata_path=args.metadata
        )
        n_verify_eligible = sum(
            1
            for p in panic_written
            if _panic_episode_stats(p)["n_window_tokens"] >= G42_MIN_VERIFY_WINDOW
        )
        n_clean_tokens = 0
        n_non_resolving = 0
        for p in clean_written:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip() and json.loads(line).get("type") == "token":
                    n_clean_tokens += 1
        for p in panic_written:
            meta = {}
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("type") == "meta":
                    meta = obj
            if meta.get("non_resolving") or non_resolving_from_temporal(
                meta.get("temporal") or {}
            ):
                n_non_resolving += 1
        gate = _gate(
            deny=deny,
            clean_files=clean_written,
            panic_files=panic_written,
            errors=[],
            n_clean_tokens=n_clean_tokens,
            n_panic_episodes=len(panic_written),
            n_non_resolving=n_non_resolving,
            n_verify_eligible=n_verify_eligible,
            excluded_thin_panic=[str(p) for p in excluded_thin],
            excluded_weak_geometry=[str(p) for p in excluded_weak],
        )
        gate["n_panic_files"] = len(panic_all)
        gate["n_panic_with_windows"] = len(panic_written)
        gate_path = CALIB_DIR / "G42_scond_corpus_gate.json"
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(gate, indent=2))
        return 0 if gate["pass"] else 1

    if args.job:
        kind, path_s, inv_s = args.job.split(",", 2)
        path = Path(path_s)
        inv_idx = int(inv_s)
        run_id = f"{kind}__{path.stem}__inv{inv_idx}"
        out = (clean_dir if kind == "clean" else panic_dir) / f"{run_id}.jsonl"
        if args.resume and out.is_file() and not (kind == "panic" and args.rescore_panic):
            print(json.dumps({"skipped": True, "path": str(out)}))
            return 0
        device = _pick_device(args.device)
        if str(device).startswith("xpu"):
            torch.xpu.empty_cache()
        bundle = load_qwen_coder(device=device, dtype=torch.float16, prefer_local=True)
        u_cond = load_u_cond(args.condition)
        data = json.loads(path.read_text(encoding="utf-8"))
        inv = _inv_by_index(data, inv_idx)
        run = replay_invocation(
            bundle,
            u_cond=u_cond,
            experiment_data=data,
            inv=inv,
            kind=kind,  # type: ignore[arg-type]
            run_id=run_id,
            max_gen_tokens=args.max_gen_tokens,
        )
        write_replay_jsonl(out, run, split="pool")
        print(json.dumps({"pass": True, "path": str(out), "n_tokens": len(run.records)}))
        return 0

    clean_candidates = discover_clean_sessions(exp_dirs[0], deny=deny)
    if len(exp_dirs) > 1 and exp_dirs[1].is_dir():
        extra_clean = discover_clean_sessions(exp_dirs[1], deny=deny)
        seen_clean = {p for p, _ in clean_candidates}
        for p, idx in extra_clean:
            if p not in seen_clean:
                clean_candidates.append((p, idx))
    clean_candidates = _dedupe_clean_by_task(clean_candidates)
    panic_candidates = discover_held_out_panic_sessions(exp_dirs, deny)
    clean_candidates = clean_candidates[: args.max_clean]
    panic_candidates = panic_candidates[: args.max_panic]

    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "n_clean_candidates": len(clean_candidates),
                "n_panic_candidates": len(panic_candidates),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return 0

    stop_server(settle_s=8.0)

    device = _pick_device(args.device)
    errors: list[str] = []
    clean_written: list[Path] = []
    panic_written: list[Path] = []
    n_clean_tokens = 0
    n_non_resolving = 0

    import gc
    import subprocess

    py = sys.executable
    script = str(Path(__file__).resolve())
    jobs: list[tuple[str, Path, int]] = []
    for path, inv_idx in clean_candidates:
        jobs.append(("clean", path, inv_idx))
    for path, inv_idx in panic_candidates:
        jobs.append(("panic", path, inv_idx))

    for kind, path, inv_idx in jobs:
        run_id = f"{kind}__{path.stem}__inv{inv_idx}"
        out = (clean_dir if kind == "clean" else panic_dir) / f"{run_id}.jsonl"
        if args.resume and out.is_file() and not (kind == "panic" and args.rescore_panic):
            if kind == "clean":
                clean_written.append(out)
                for line in out.read_text(encoding="utf-8").splitlines():
                    if line.strip() and json.loads(line).get("type") == "token":
                        n_clean_tokens += 1
            else:
                panic_written.append(out)
                for line in out.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if obj.get("type") == "meta" and (
                        obj.get("non_resolving")
                        or non_resolving_from_temporal(obj.get("temporal") or {})
                    ):
                        n_non_resolving += 1
            continue
        job_arg = f"{kind},{path},{inv_idx}"
        cmd = [
            py,
            script,
            "--device",
            device,
            "--job",
            job_arg,
            "--max-gen-tokens",
            str(args.max_gen_tokens),
            "--metadata",
            str(args.metadata),
            "--condition",
            str(args.condition),
        ]
        try:
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
            if proc.returncode != 0:
                errors.append(f"{run_id}: rc={proc.returncode} stderr={proc.stderr[-500:]}")
                continue
            if out.is_file():
                if kind == "clean":
                    clean_written.append(out)
                    for line in out.read_text(encoding="utf-8").splitlines():
                        if line.strip() and json.loads(line).get("type") == "token":
                            n_clean_tokens += 1
                else:
                    panic_written.append(out)
                    for line in out.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        if obj.get("type") == "meta" and (
                            obj.get("non_resolving")
                            or non_resolving_from_temporal(obj.get("temporal") or {})
                        ):
                            n_non_resolving += 1
            else:
                errors.append(f"{run_id}: no output file after subprocess")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{run_id}: {exc}")
        gc.collect()
        if str(device).startswith("xpu") and hasattr(torch, "xpu"):
            try:
                torch.xpu.empty_cache()
            except Exception:
                pass

    n_verify_eligible = 0
    panic_written, excluded_thin, excluded_weak = _finalize_panic_corpus(
        panic_written, metadata_path=args.metadata
    )
    for p in panic_written:
        stats = _panic_episode_stats(p)
        if stats["n_window_tokens"] >= G42_MIN_VERIFY_WINDOW:
            n_verify_eligible += 1

    gate = _gate(
        deny=deny,
        clean_files=clean_written,
        panic_files=panic_written,
        errors=errors,
        n_clean_tokens=n_clean_tokens,
        n_panic_episodes=len(panic_written),
        n_non_resolving=n_non_resolving,
        n_verify_eligible=n_verify_eligible,
        excluded_thin_panic=[str(p) for p in excluded_thin],
        excluded_weak_geometry=[str(p) for p in excluded_weak],
    )
    gate_path = CALIB_DIR / "G42_scond_corpus_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
