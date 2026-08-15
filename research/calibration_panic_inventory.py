#!/usr/bin/env python
"""Two-stage held-out panic inventory for CAST τ/σ calibration."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jlens_qwen.caa.calibration_deny import is_denied_panic_task, load_deny_list  # noqa: E402
from jlens_qwen.caa.s_cond_replay import (  # noqa: E402
    default_calibration_experiment_dirs,
    discover_held_out_panic_sessions,
    extract_task_num,
    load_replay_jsonl,
    relabel_panic_windows,
)

DEFAULT_META = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "metadata.json"
CALIB_DIR = ROOT / "artifacts" / "caa" / "UNI_CAA01" / "calibration"
OUT_PATH = CALIB_DIR / "panic_inventory.json"
PANIC_DIR = CALIB_DIR / "panic"
MIN_GEN_TOKENS = 8
MIN_VERIFY_WINDOW = 2
TARGET_REPLAY = 15
TARGET_VERIFY = 3


def _recipe_fired_in_session(data: dict, inv_idx: int) -> bool:
    for inv in data.get("invocations") or []:
        if int(inv.get("invocation_index") or 0) != inv_idx:
            continue
        chaos = inv.get("chaos") or {}
        timeline = (chaos.get("monitor") or {}).get("recipe_timeline") or []
        return any(
            x.get("recipe_id") == "B_STUB_DISTRESS"
            and x.get("severity") in ("warn", "critical")
            for x in timeline
        )
    return False


def stage_0a_pre_replay(deny: dict[str, Any]) -> dict[str, Any]:
    dirs = default_calibration_experiment_dirs(ROOT)
    session_eligible: list[dict[str, Any]] = []
    needs_rerun: list[dict[str, Any]] = []
    denied_train_leak: list[dict[str, Any]] = []

    for root in dirs:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("unicaa01_corpus_p*_panic_latest.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            task_slug = str(data.get("task_slug") or "")
            task_num = extract_task_num(task_slug) or ""
            entry = {
                "path": str(path),
                "task_slug": task_slug,
                "task_num": task_num,
                "experiment": data.get("experiment"),
            }
            if is_denied_panic_task(task_slug, deny):
                denied_train_leak.append(entry)
                continue
            candidates = discover_held_out_panic_sessions([path.parent], deny)
            matched = [c for c in candidates if c[0] == path]
            if matched:
                path_m, inv_idx = matched[0]
                session_eligible.append({**entry, "invocation_index": inv_idx})
            else:
                needs_rerun.append(entry)

    by_task: dict[str, dict] = {}
    for e in session_eligible:
        slug = e["task_slug"]
        if slug not in by_task:
            by_task[slug] = e
    session_eligible = sorted(by_task.values(), key=lambda x: x["task_slug"])

    return {
        "session_eligible": session_eligible,
        "needs_rerun": sorted(needs_rerun, key=lambda x: x["task_slug"]),
        "denied_train_leak": denied_train_leak,
        "n_session_eligible": len(session_eligible),
        "n_needs_rerun": len(needs_rerun),
    }


def _window_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_window = sum(1 for r in rows if r.get("in_panic_window"))
    return {
        "n_window_tokens": n_window,
        "thin_episode": n_window > 0 and n_window < MIN_VERIFY_WINDOW,
        "verify_eligible": n_window >= MIN_VERIFY_WINDOW,
    }


def stage_0b_post_replay() -> dict[str, Any]:
    replay_eligible: list[dict[str, Any]] = []
    verify_eligible: list[dict[str, Any]] = []
    no_window: list[dict[str, Any]] = []

    for path in sorted(PANIC_DIR.glob("*.jsonl")):
        relabel_panic_windows(path)
        meta, rows = load_replay_jsonl(path)
        stats = _window_stats(rows)
        entry = {
            "run_id": meta.get("run_id"),
            "path": str(path),
            "task_slug": meta.get("task_slug"),
            "task_num": extract_task_num(str(meta.get("task_slug") or "")) or "",
            **stats,
        }
        if stats["n_window_tokens"] < 1:
            no_window.append(entry)
        else:
            replay_eligible.append(entry)
            if stats["verify_eligible"]:
                verify_eligible.append(entry)

    return {
        "replay_eligible": sorted(replay_eligible, key=lambda x: str(x.get("run_id"))),
        "verify_eligible": sorted(verify_eligible, key=lambda x: str(x.get("run_id"))),
        "no_window": no_window,
        "n_replay_eligible": len(replay_eligible),
        "n_verify_eligible": len(verify_eligible),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="CAST calibration panic inventory")
    ap.add_argument("--metadata", type=Path, default=DEFAULT_META)
    ap.add_argument(
        "--stage",
        choices=["pre", "post", "both"],
        default="both",
    )
    args = ap.parse_args()

    deny = load_deny_list(args.metadata)
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "targets": {
            "replay_eligible": TARGET_REPLAY,
            "verify_eligible": TARGET_VERIFY,
        },
    }

    if args.stage in ("pre", "both"):
        payload["pre_replay"] = stage_0a_pre_replay(deny)
    if args.stage in ("post", "both"):
        if args.stage == "post" and OUT_PATH.is_file():
            try:
                existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
                if existing.get("pre_replay"):
                    payload["pre_replay"] = existing["pre_replay"]
            except json.JSONDecodeError:
                pass
        payload["post_replay"] = stage_0b_post_replay()

    payload["ready_for_split"] = (
        payload.get("post_replay", {}).get("n_replay_eligible", 0) >= TARGET_REPLAY
        and payload.get("post_replay", {}).get("n_verify_eligible", 0) >= TARGET_VERIFY
    )

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
