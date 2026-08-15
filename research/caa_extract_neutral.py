#!/usr/bin/env python
"""Extract û_neutral — real N1–N10 purity gate on live capture."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from caa_error_combo_study import score_text  # noqa: E402
from jlens_qwen.caa.gate_io import write_gate  # noqa: E402
from jlens_qwen.caa.neutral_capture import NEUTRAL_TASKS, capture_neutral_gen_start  # noqa: E402
from jlens_qwen.caa.neutral_purity import run_neutral_purity_checks  # noqa: E402
from jlens_qwen.caa.unicaa01_corpus import NEUTRAL_ARTIFACTS as CORPUS_NEUTRAL  # noqa: E402
from prefix_divergence_tracer import eligible_for_panic_trace  # noqa: E402
from unicaa01_registry import neutral_capture_specs, pairs_for_wave  # noqa: E402

GATES = ROOT / "artifacts" / "gates"
OUT = ROOT / "artifacts" / "caa" / "UNI_CAA01"
NEUTRAL_ARTIFACT_SCHEMA = {
    "vector_id": "UNI_CAA01",
    "kind": "neutral",
    "source": "neutral_capture",
}


def save_neutral_artifacts(result, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for L, vec in result.neutral.items():
        path = out_dir / f"neutral_L{L}.pt"
        torch.save(
            {
                **NEUTRAL_ARTIFACT_SCHEMA,
                "layer": int(L),
                "u_hat": torch.tensor(vec, dtype=torch.float32),
                "n_tasks": result.n_tasks,
                "n_pairs": result.n_tasks,
            },
            path,
        )
        written.append(str(path))
    capture_path = out_dir / "neutral_capture.json"
    capture_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    written.append(str(capture_path))
    return written


def build_gate(
    result,
    *,
    mode: str,
    device: str | None = None,
    bundle=None,
    tasks: list[dict[str, str]] | None = None,
) -> dict:
    task_list = tasks or NEUTRAL_TASKS
    checks, sabotage = run_neutral_purity_checks(
        result,
        caa_root=OUT,
        score_text_fn=score_text,
        eligible_for_panic_trace_fn=eligible_for_panic_trace,
        capture_neutral_gen_start_fn=capture_neutral_gen_start,
        bundle=bundle,
        unit_only=(mode == "unit"),
        tasks=task_list,
    )
    env = {"mode": mode}
    if device:
        env["device"] = device
    if result is not None:
        env["n_tasks"] = result.n_tasks
    return write_gate(
        GATES / "caa_neutral_purity_gate.json",
        gate="caa_neutral_purity",
        pass_=all(c["pass"] for c in checks) and all(s["pass"] for s in sabotage),
        checks=checks,
        sabotage=sabotage,
        environment=env,
        extra={"neutral_tasks": [t["slug"] for t in task_list]},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--unit-only", action="store_true")
    ap.add_argument(
        "--corpus-pilot",
        action="store_true",
        help="Use UNICAA01 corpus p01+p02 neutral specs; write to UNI_CAA01_corpus/neutral/",
    )
    ap.add_argument(
        "--corpus-batch",
        action="store_true",
        help="Use first 4 UNICAA01 neutral pair specs (p01-p04)",
    )
    ap.add_argument(
        "--corpus-all",
        action="store_true",
        help="Use all 66 UNICAA01 corpus neutral specs (wave1+wave2)",
    )
    ap.add_argument("--corpus-pairs", nargs="*", default=None, help="Pair IDs for corpus neutral tasks")
    args = ap.parse_args()

    out_dir = CORPUS_NEUTRAL if args.corpus_pilot or args.corpus_pairs or args.corpus_batch or args.corpus_all else OUT
    tasks = NEUTRAL_TASKS
    if args.corpus_pilot:
        tasks = neutral_capture_specs(["p01_priority_queue", "p02_lru_cache"])
    elif args.corpus_all:
        tasks = neutral_capture_specs([p.pair_id for p in pairs_for_wave(None)])
    elif args.corpus_batch:
        tasks = neutral_capture_specs(
            ["p01_priority_queue", "p02_lru_cache", "p03_bst_insert", "p04_graph_bfs"]
        )
    elif args.corpus_pairs:
        tasks = neutral_capture_specs(args.corpus_pairs)

    if args.unit_only or os.environ.get("CAA_NEUTRAL_LIVE") != "1":
        payload = build_gate(None, mode="unit")
        print(json.dumps(payload, indent=2))
        return 0 if payload["pass"] else 1

    from jlens_qwen.model_loader import load_qwen_coder

    bundle = load_qwen_coder(device=args.device)
    result = capture_neutral_gen_start(bundle, tasks=tasks)
    artifacts = save_neutral_artifacts(result, out_dir)
    payload = build_gate(result, mode="live", device=args.device, bundle=bundle, tasks=tasks)
    payload["artifacts"] = artifacts
    print(json.dumps({"pass": payload["pass"], "n_tasks": result.n_tasks, "artifacts": artifacts, "out_dir": str(out_dir)}, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
