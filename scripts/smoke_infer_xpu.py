#!/usr/bin/env python
"""Smoke inference on XPU: forward + hidden-state hooks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument(
        "--prompt",
        default=(
            "def fibonacci(n):\n"
            "    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
            "# Explain the time complexity of this function.\n"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "gates" / "smoke_infer.json",
    )
    args = parser.parse_args()

    import torch

    from jlens_qwen.model_loader import (
        ActivationRecorder,
        load_qwen_coder,
        save_model_manifest,
    )

    bundle = load_qwen_coder(device=args.device, dtype=torch.float16)
    manifest = save_model_manifest(
        bundle, ROOT / "artifacts" / "gates" / "model_manifest.json"
    )

    input_ids = bundle.encode(args.prompt, max_length=args.seq_len)
    # Warmup
    with torch.no_grad():
        with ActivationRecorder(bundle.layers, at=list(range(bundle.n_layers))) as rec:
            _ = bundle.model(input_ids=input_ids, use_cache=False)

    times = []
    for _ in range(3):
        if str(bundle.device).startswith("xpu"):
            torch.xpu.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with ActivationRecorder(
                bundle.layers, at=list(range(bundle.n_layers))
            ) as recorder:
                outputs = bundle.model(input_ids=input_ids, use_cache=False)
        if str(bundle.device).startswith("xpu"):
            torch.xpu.synchronize()
        times.append(time.perf_counter() - t0)

    times_sorted = sorted(times)
    median = times_sorted[len(times_sorted) // 2]
    peak_vram = None
    if str(bundle.device).startswith("xpu") and hasattr(torch.xpu, "max_memory_allocated"):
        peak_vram = torch.xpu.max_memory_allocated() / (1024**3)

    result = {
        "device": str(bundle.device),
        "dtype": str(bundle.dtype),
        "seq_len": int(input_ids.shape[1]),
        "n_layers_captured": len(recorder.activations),
        "activation_shapes": {
            str(k): list(v.shape) for k, v in recorder.activations.items()
        },
        "logits_shape": list(outputs.logits.shape),
        "latency_s": times,
        "latency_median_s": median,
        "peak_vram_gb": peak_vram,
        "manifest": manifest,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
