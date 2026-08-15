"""GATE-1: Model load + XPU inference."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

from jlens_qwen.model_loader import (
    EXPECTED,
    ActivationRecorder,
    load_qwen_coder,
    save_model_manifest,
)
from tests.gates import save_gate_report, summarize_checks

ROOT = Path(__file__).resolve().parents[2]
PROMPT = (
    "def binary_search(arr, target):\n"
    "    lo, hi = 0, len(arr) - 1\n"
    "    while lo <= hi:\n"
    "        mid = (lo + hi) // 2\n"
    "        if arr[mid] == target:\n"
    "            return mid\n"
    "        if arr[mid] < target:\n"
    "            lo = mid + 1\n"
    "        else:\n"
    "            hi = mid - 1\n"
    "    return -1\n"
)


@pytest.fixture(scope="module")
def bundle(qwen_bundle):
    return qwen_bundle


@pytest.mark.gate1
@pytest.mark.slow
def test_gate1_model(bundle):
    checks = []
    cfg = bundle.config
    for key, expected in EXPECTED.items():
        ok = cfg.get(key) == expected
        checks.append(
            {
                "id": f"G1.1.{key}",
                "name": f"config {key}",
                "pass": ok,
                "detail": f"got={cfg.get(key)} expected={expected}",
            }
        )

    checks.append(
        {
            "id": "G1.2a",
            "name": "W_U shape",
            "pass": tuple(bundle.W_U.shape) == (152064, 3584),
            "detail": f"shape={tuple(bundle.W_U.shape)}",
        }
    )
    checks.append(
        {
            "id": "G1.2b",
            "name": "gamma shape",
            "pass": tuple(bundle.gamma.shape) == (3584,),
            "detail": f"shape={tuple(bundle.gamma.shape)}",
        }
    )

    on_xpu = str(bundle.device).startswith("xpu")
    all_fp16 = all(p.dtype == torch.float16 for p in bundle.model.parameters())
    checks.append(
        {
            "id": "G1.3",
            "name": "fp16 on preferred device",
            "pass": all_fp16 and (on_xpu or str(bundle.device) == "cpu"),
            "detail": f"device={bundle.device} all_fp16={all_fp16}",
        }
    )

    input_ids = bundle.encode(PROMPT, max_length=512)
    if on_xpu and hasattr(torch.xpu, "reset_peak_memory_stats"):
        torch.xpu.reset_peak_memory_stats()

    times = []
    for _ in range(3):
        if on_xpu:
            torch.xpu.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with ActivationRecorder(
                bundle.layers, at=list(range(bundle.n_layers))
            ) as recorder:
                outputs = bundle.model(input_ids=input_ids, use_cache=False)
        if on_xpu:
            torch.xpu.synchronize()
        times.append(time.perf_counter() - t0)

    median = sorted(times)[1]
    peak_vram = None
    if on_xpu and hasattr(torch.xpu, "max_memory_allocated"):
        peak_vram = torch.xpu.max_memory_allocated() / (1024**3)

    checks.append(
        {
            "id": "G1.4",
            "name": "peak VRAM < 15.5 GB",
            "pass": (peak_vram is None and not on_xpu)
            or (peak_vram is not None and peak_vram < 15.5),
            "detail": f"peak_vram_gb={peak_vram}",
        }
    )
    checks.append(
        {
            "id": "G1.5",
            "name": "forward latency < 10s (128-512 tok median)",
            "pass": median < 10.0,
            "detail": f"median_s={median:.3f} times={times}",
        }
    )
    checks.append(
        {
            "id": "G1.6",
            "name": "28 layer hooks",
            "pass": len(recorder.activations) == 28
            and all(v.shape[-1] == 3584 for v in recorder.activations.values()),
            "detail": f"n={len(recorder.activations)} shapes={[list(v.shape) for v in recorder.activations.values()][:3]}...",
        }
    )

    with torch.no_grad():
        out1 = bundle.model(input_ids=input_ids, use_cache=False).logits
        out2 = bundle.model(input_ids=input_ids, use_cache=False).logits
    max_diff = float((out1 - out2).abs().max().item())
    # Arc XPU fp16 matmuls are not bit-exact across runs
    det_thresh = 0.05 if on_xpu else 1e-5
    checks.append(
        {
            "id": "G1.7",
            "name": "deterministic forward",
            "pass": max_diff < det_thresh,
            "detail": f"max_diff={max_diff} thresh={det_thresh}",
        }
    )

    save_model_manifest(bundle, ROOT / "artifacts" / "gates" / "model_manifest.json")
    (ROOT / "artifacts" / "gates" / "latency_bench.json").write_text(
        json.dumps({"times": times, "median": median}, indent=2), encoding="utf-8"
    )
    if peak_vram is not None:
        (ROOT / "artifacts" / "gates" / "vram_peak.json").write_text(
            json.dumps({"peak_vram_gb": peak_vram}, indent=2), encoding="utf-8"
        )

    summary = summarize_checks(checks)
    save_gate_report("GATE-1", summary)
    assert summary["overall_pass"], summary
