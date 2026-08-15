#!/usr/bin/env python
"""Run held-out panic Aider invocations to fill calibration corpus gaps."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.caa.calibration_deny import is_denied_panic_task, load_deny_list  # noqa: E402
from jlens_qwen.caa.s_cond_replay import (  # noqa: E402
    _recipe_fired,
    default_calibration_experiment_dirs,
    discover_held_out_panic_sessions,
    extract_task_num,
)
from calibration_panic_inventory import stage_0a_pre_replay  # noqa: E402
from unicaa01_registry import PAIRS_ALL, get_pair  # noqa: E402
from unicaa01_restart_server import restart_server, stop_server  # noqa: E402

INVENTORY = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "calibration" / "panic_inventory.json"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
RUNNER = ROOT / "research" / "run_unicaa01_corpus.py"
DEFAULT_META = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "metadata.json"
MIN_GEN_TOKENS = 8
TARGET_SESSIONS = 15
SETTLE_S = 8.0
DEFAULT_RUN_TIMEOUT_S = 900  # 15 min — fail fast vs 20 min stall on bad pairs

_PANIC_SLUG_TO_PAIR: dict[str, str] = {p.panic_slug: p.pair_id for p in PAIRS_ALL}


def _resolve_pair_id(entry: dict) -> str | None:
    slug = str(entry.get("task_slug") or "")
    if slug in _PANIC_SLUG_TO_PAIR:
        return _PANIC_SLUG_TO_PAIR[slug]
    exp = str(entry.get("experiment") or "")
    if exp.startswith("unicaa01_corpus_") and exp.endswith("_panic"):
        return exp[len("unicaa01_corpus_") : -len("_panic")]
    if slug.startswith("unicaa01_") and slug.endswith("_panic"):
        return slug[len("unicaa01_") : -len("_panic")]
    return None


def _session_recipe_acceptable(corpus_json: str | None) -> bool:
    if not corpus_json:
        return False
    path = Path(corpus_json)
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    for inv in data.get("invocations") or []:
        chaos = inv.get("chaos") or {}
        tids = chaos.get("token_ids") or []
        if _recipe_fired(chaos) and len(tids) >= MIN_GEN_TOKENS:
            return True
    return False


def _eligible_task_nums(deny: dict) -> set[str]:
    nums: set[str] = set()
    for path, _inv in discover_held_out_panic_sessions(
        default_calibration_experiment_dirs(ROOT), deny
    ):
        data = json.loads(path.read_text(encoding="utf-8"))
        num = extract_task_num(str(data.get("task_slug") or ""))
        if num:
            nums.add(num)
    return nums


def _prioritize_diverse(entries: list[dict], *, max_runs: int) -> list[dict]:
    """Round-robin across domains; dedupe by task_num."""
    by_domain: dict[str, list[dict]] = defaultdict(list)
    seen_nums: set[str] = set()
    for entry in entries:
        pair_id = _resolve_pair_id(entry)
        if not pair_id:
            continue
        try:
            pair = get_pair(pair_id)
        except KeyError:
            continue
        num = str(entry.get("task_num") or "")
        if num in seen_nums:
            continue
        seen_nums.add(num)
        by_domain[pair.domain].append({**entry, "pair_id": pair_id, "domain": pair.domain})

    domains = sorted(by_domain)
    out: list[dict] = []
    idx = 0
    while len(out) < max_runs and any(by_domain[d] for d in domains):
        domain = domains[idx % len(domains)]
        if by_domain[domain]:
            out.append(by_domain[domain].pop(0))
        idx += 1
        if idx > max_runs * len(domains) * 2:
            break
    return out


def _run_one_pair(pair_id: str, *, run_timeout_s: int) -> dict:
    if not restart_server():
        return {"pair_id": pair_id, "returncode": 1, "recipe_accepted": False, "error": "server_start"}
    time.sleep(2.0)
    cmd = [
        str(PY),
        str(RUNNER),
        "--pair",
        pair_id,
        "--arm",
        "panic",
        "--no-post-batch",
        "--restart-server-every",
        "0",
        "--run-timeout",
        str(run_timeout_s),
    ]
    print("+", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT))
    stop_server(settle_s=SETTLE_S)
    pair = get_pair(pair_id)
    corpus_json = str(
        ROOT / "experiments" / "unicaa01_corpus" / "out" / f"{pair.panic_experiment}_latest.json"
    )
    return {
        "pair_id": pair_id,
        "returncode": rc,
        "corpus_json": corpus_json,
        "recipe_accepted": _session_recipe_acceptable(corpus_json),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run panic gap fills for CAST calibration")
    ap.add_argument("--metadata", type=Path, default=DEFAULT_META)
    ap.add_argument("--max-runs", type=int, default=10)
    ap.add_argument("--target-sessions", type=int, default=TARGET_SESSIONS)
    ap.add_argument("--inventory", type=Path, default=INVENTORY)
    ap.add_argument(
        "--run-timeout",
        type=int,
        default=DEFAULT_RUN_TIMEOUT_S,
        help="Per-pair Aider timeout seconds (default 900)",
    )
    args = ap.parse_args()

    deny = load_deny_list(args.metadata)
    inv = json.loads(args.inventory.read_text(encoding="utf-8")) if args.inventory.is_file() else {}
    pre = inv.get("pre_replay") or stage_0a_pre_replay(deny)

    results: list[dict] = []
    attempted_pairs: set[str] = set()
    attempts = 0
    while attempts < args.max_runs:
        have_nums = _eligible_task_nums(deny)
        if len(have_nums) >= args.target_sessions:
            break
        candidates: list[dict] = []
        for e in pre.get("needs_rerun") or []:
            if is_denied_panic_task(str(e.get("task_slug") or ""), deny):
                continue
            if str(e.get("task_num")) in have_nums:
                continue
            pair_id = e.get("pair_id") or _resolve_pair_id(e)
            if pair_id and pair_id in attempted_pairs:
                continue
            candidates.append(e)
        if not candidates:
            break
        targets = _prioritize_diverse(candidates, max_runs=1)
        if not targets:
            break
        entry = targets[0]
        pair_id = entry.get("pair_id") or _resolve_pair_id(entry)
        if not pair_id:
            results.append({**entry, "skipped": True, "reason": "no pair_id"})
            attempts += 1
            continue
        try:
            get_pair(pair_id)
        except KeyError:
            results.append({**entry, "skipped": True, "reason": f"unknown pair {pair_id}"})
            attempts += 1
            continue

        attempted_pairs.add(pair_id)
        result = _run_one_pair(pair_id, run_timeout_s=args.run_timeout)
        results.append({**entry, **result})
        attempts += 1
        if result.get("recipe_accepted"):
            have_nums = _eligible_task_nums(deny)
            print(
                json.dumps(
                    {"progress": f"{len(have_nums)}/{args.target_sessions}", "pair_id": pair_id}
                ),
                flush=True,
            )

    stop_server(settle_s=SETTLE_S)
    have_nums = _eligible_task_nums(deny)
    out = {
        "n_attempted": len(results),
        "n_session_eligible": len(have_nums),
        "target_sessions": args.target_sessions,
        "ready_for_harvest": len(have_nums) >= args.target_sessions,
        "results": results,
    }
    print(json.dumps(out, indent=2))
    ok_runs = [r for r in results if r.get("recipe_accepted")]
    need = max(0, args.target_sessions - len(have_nums))
    if need > 0:
        return 1
    return 0 if ok_runs or not results else 1


if __name__ == "__main__":
    raise SystemExit(main())
