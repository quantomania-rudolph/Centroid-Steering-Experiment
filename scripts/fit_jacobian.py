#!/usr/bin/env python
"""Fit Jacobian lens on CPU with machine-safe max parallelism."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fit_jacobian")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=ROOT / "data" / "prompts" / "corpus.txt")
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--device", default="cpu", help="Device for fit (cpu recommended)")
    parser.add_argument("--dim-batch", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--source-layers", default="8-24", help="e.g. 8-24 or 0-26")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "lenses" / "qwen_jacobian_lens.pt",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts" / "checkpoints" / "fit_ckpt.pt",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Prompt shards (default: 1; use >1 for resume-friendly chunks)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max parallel model workers (RAM-limited; default auto)",
    )
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=None,
        help="BLAS/PyTorch threads per worker (default: logical_cpus - 1)",
    )
    parser.add_argument("--reserve-cores", type=int, default=1)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile each decoder block via jlens (default on)",
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Allow 2 concurrent model workers if total RAM permits",
    )
    parser.add_argument(
        "--plan-out",
        type=Path,
        default=None,
        help="Write JSON CPU fit plan (default: artifacts/gates/cpu_fit_plan.json)",
    )
    args = parser.parse_args()

    from jlens_qwen.cpu_tuning import build_fit_plan, configure_cpu_threads, log_fit_plan
    from jlens_qwen.jacobian_fit import fit_jacobians, fit_jacobians_sharded, load_prompts
    from jlens_qwen.model_loader import load_qwen_coder

    plan = build_fit_plan(
        workers=args.workers,
        omp_threads=args.omp_threads,
        reserve_cores=args.reserve_cores,
        dim_batch=args.dim_batch,
        compile_model=args.compile,
        aggressive=args.aggressive,
    )
    log_fit_plan(plan)
    configure_cpu_threads(plan.omp_threads)

    plan_path = args.plan_out or (ROOT / "artifacts" / "gates" / "cpu_fit_plan.json")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(
            {
                "logical_cores": plan.logical_cores,
                "omp_threads": plan.omp_threads,
                "workers": plan.workers,
                "max_concurrent": args.max_concurrent or plan.max_concurrent,
                "dim_batch": plan.dim_batch,
                "compile": plan.compile_model,
                "available_ram_gb": plan.available_ram_gb,
                "notes": plan.notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    import os
    from datetime import datetime, timezone

    from jlens_qwen.fit_progress import FitJobMeta

    lo, hi = [int(x) for x in args.source_layers.split("-")]
    job_id = f"L{lo}-L{hi}_n{args.n_prompts}"
    FitJobMeta(
        job_id=job_id,
        pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(),
        script="fit_jacobian.py",
        source_layers=args.source_layers,
        n_prompts=args.n_prompts,
        dim_batch=plan.dim_batch,
        max_seq_len=args.max_seq_len,
        omp_threads=plan.omp_threads,
        compile_model=plan.compile_model,
        out_path=str(args.out),
        checkpoint_path=str(args.checkpoint),
    ).save()

    import torch

    source_layers = list(range(lo, hi + 1))

    prompts = load_prompts(args.prompts, n=args.n_prompts)
    logger.info("Loaded %d prompts; fitting layers %s", len(prompts), source_layers)

    bundle = load_qwen_coder(device=args.device, dtype=torch.float16)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    max_concurrent = args.max_concurrent if args.max_concurrent is not None else plan.max_concurrent
    if plan.workers > 1:
        lens = fit_jacobians_sharded(
            bundle,
            prompts,
            source_layers=source_layers,
            dim_batch=plan.dim_batch,
            max_seq_len=args.max_seq_len,
            checkpoint_path=str(args.checkpoint),
            compile_model=plan.compile_model,
            workers=plan.workers,
            max_concurrent=max_concurrent,
            omp_threads=plan.omp_threads,
            shard_out_dir=args.checkpoint.parent / "shards",
        )
    else:
        lens = fit_jacobians(
            bundle,
            prompts,
            source_layers=source_layers,
            dim_batch=plan.dim_batch,
            max_seq_len=args.max_seq_len,
            checkpoint_path=str(args.checkpoint),
            compile_model=plan.compile_model,
        )
    lens.save(str(args.out))
    logger.info("Saved lens to %s (n_prompts=%d)", args.out, lens.n_prompts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
