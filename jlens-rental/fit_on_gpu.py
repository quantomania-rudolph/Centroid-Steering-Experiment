#!/usr/bin/env python3
"""Fit Jacobian lens on CUDA (H100) with quarter-pass + per-prompt checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fit_on_gpu")

ROOT = Path(__file__).resolve().parent


def parse_layer_range(spec: str) -> list[int]:
    lo, hi = [int(x) for x in spec.split("-")]
    return list(range(lo, hi + 1))


def load_prompts(path: Path, n: int | None) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data if isinstance(data, list) else data["prompts"]
    prompts = [p for p in prompts if len(p) >= 40]
    if n is not None:
        prompts = prompts[:n]
    return prompts


def save_fitted_lens(
    path: Path,
    jacobians: dict[int, torch.Tensor],
    *,
    n_prompts: int,
    d_model: int,
    dtype: torch.dtype = torch.float16,
) -> None:
    torch.save(
        {
            "J": {int(k): v.to(dtype) for k, v in jacobians.items()},
            "n_prompts": n_prompts,
            "source_layers": sorted(int(k) for k in jacobians),
            "d_model": d_model,
        },
        path,
    )


def write_progress(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "fit_config.yaml")
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=ROOT)
    args = parser.parse_args()

    work = args.work_dir.resolve()
    sys.path.insert(0, str(work))
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    model_id = os.environ.get("MODEL_ID", cfg["model_id"])
    source_layers = parse_layer_range(
        os.environ.get("SOURCE_LAYERS", cfg["source_layers"])
    )
    target_layer = int(cfg.get("target_layer", 27))
    n_prompts = int(os.environ.get("N_PROMPTS", cfg["n_prompts"]))
    max_seq_len = int(cfg["max_seq_len"])
    skip_first = int(cfg["skip_first"])
    dim_batch = int(os.environ.get("DIM_BATCH", cfg["dim_batch"]))
    compile_model = bool(cfg.get("compile_model", False))
    dtype_name = cfg.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_name, torch.bfloat16)
    checkpoint_path = str(work / cfg.get("checkpoint_path", "ckpt.pt"))
    out_file = work / cfg.get("out_file", "qwen_jacobian_lens.pt")
    progress_path = work / "fit_progress.json"

    prompts_path = args.prompts or work / cfg.get("prompts_file", "corpus_100.json")
    prompts = load_prompts(prompts_path, n_prompts)
    logger.info(
        "Fit plan: model=%s layers=%s prompts=%d dim_batch=%d device=cuda "
        "quarter_checkpoints=ON",
        model_id,
        source_layers,
        len(prompts),
        dim_batch,
    )

    if not torch.cuda.is_available():
        logger.error("CUDA not available — need an H100 (or other CUDA GPU) instance.")
        return 1

    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info("GPU: %s (%.1f GB VRAM)", device_name, vram_gb)

    import transformers
    import jlens
    from checkpoint_fit import fit_with_quarter_checkpoints

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="cuda",
    )
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok, force_bos=True, compile=compile_model)

    t0 = time.perf_counter()
    n_passes = (model.d_model + dim_batch - 1) // dim_batch

    def on_progress(prompt_idx: int, n_prompts_: int, done: int, total: int) -> None:
        elapsed = time.perf_counter() - t0
        rate = (done / elapsed * 60.0) if elapsed > 0 and done > 0 else None
        # crude overall fraction across prompts
        frac = (prompt_idx + done / max(total, 1)) / max(n_prompts_, 1)
        write_progress(
            progress_path,
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "phase": "backward",
                "prompt_index": prompt_idx,
                "n_prompts": n_prompts_,
                "backward_pass": done,
                "backward_passes_total": total,
                "pass_rate_per_min": rate,
                "elapsed_seconds": elapsed,
                "fraction": frac,
                "checkpoint_path": checkpoint_path,
                "quarter_checkpoints": True,
                "gpu": device_name,
            },
        )

    write_progress(
        progress_path,
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "phase": "load_model",
            "prompt_index": 0,
            "n_prompts": len(prompts),
            "backward_pass": 0,
            "backward_passes_total": n_passes,
            "quarter_checkpoints": True,
            "gpu": device_name,
        },
    )

    lens = fit_with_quarter_checkpoints(
        model,
        prompts=prompts,
        source_layers=source_layers,
        target_layer=target_layer,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first=skip_first,
        checkpoint_path=checkpoint_path,
        resume=True,
        on_progress=on_progress,
    )
    elapsed = time.perf_counter() - t0
    logger.info(
        "quarter-checkpoint fit done in %.1f s (%.2f s/prompt)",
        elapsed,
        elapsed / max(lens.n_prompts, 1),
    )

    save_fitted_lens(
        out_file,
        lens.jacobians,
        n_prompts=lens.n_prompts,
        d_model=lens.d_model,
    )
    logger.info("Saved %s (%.1f MB)", out_file, out_file.stat().st_size / 1e6)

    meta = {
        "model_id": model_id,
        "source_layers": source_layers,
        "target_layer": target_layer,
        "n_prompts": lens.n_prompts,
        "dim_batch": dim_batch,
        "elapsed_seconds": elapsed,
        "gpu": device_name,
        "out_file": str(out_file),
        "quarter_checkpoints": True,
        "checkpoint_path": checkpoint_path,
    }
    (work / "fit_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_progress(
        progress_path,
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "phase": "done",
            "prompt_index": lens.n_prompts,
            "n_prompts": lens.n_prompts,
            "backward_pass": n_passes,
            "backward_passes_total": n_passes,
            "elapsed_seconds": elapsed,
            "fraction": 1.0,
            "quarter_checkpoints": True,
            "gpu": device_name,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
