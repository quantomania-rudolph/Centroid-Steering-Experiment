"""XPU component tests for persistent probe hooks."""

from __future__ import annotations

import os
import time

import pytest
import torch

from jlens_qwen.chaos.bus import ChaosBus
from jlens_qwen.chaos.live_generate import live_generate
from jlens_qwen.chaos.probe import ProbeEngine
from jlens_qwen.jacobian_lens import FittedJacobianLens

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LENS = os.path.join(ROOT, "artifacts", "lenses", "qwen_jacobian_lens.pt")


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU required",
)
def test_probe_hooks_persistent(qwen_bundle):
    lens = FittedJacobianLens.load(LENS) if os.path.isfile(LENS) else None
    probe = ProbeEngine(qwen_bundle, lens)
    info = probe.setup()
    assert info["n_hooks"] == 3
    n_hooks_1 = len(probe._hooks)
    ids = qwen_bundle.encode("def foo():\n    return 1\n", max_length=64)
    for _ in range(3):
        probe.clear_residuals()
        with torch.inference_mode():
            qwen_bundle.model(input_ids=ids, use_cache=False)
        for L in probe.probe_layers:
            assert L in probe.residuals
            assert probe.residuals[L].shape[-1] == qwen_bundle.d_model
    assert len(probe._hooks) == n_hooks_1  # no re-register
    probe.remove_hooks()


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU required",
)
def test_f1_hot_path_budget(qwen_bundle):
    lens = FittedJacobianLens.load(LENS) if os.path.isfile(LENS) else None
    probe = ProbeEngine(qwen_bundle, lens)
    probe.setup()
    ids = qwen_bundle.encode("x = 1\n", max_length=32)
    times = []
    with torch.inference_mode():
        # warm-up
        for _ in range(5):
            probe.clear_residuals()
            qwen_bundle.model(input_ids=ids, use_cache=False)
            probe.hot_metrics_at(16)
        if str(qwen_bundle.device).startswith("xpu"):
            torch.xpu.synchronize()
        for _ in range(40):
            probe.clear_residuals()
            qwen_bundle.model(input_ids=ids, use_cache=False)
            t0 = time.perf_counter()
            probe.hot_metrics_at(16)  # F1 hot path = single probe
            if str(qwen_bundle.device).startswith("xpu"):
                torch.xpu.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    probe.remove_hooks()
    times = sorted(times)
    p95 = times[int(0.95 * (len(times) - 1))]
    # Single L16 transport on Arc; must stay well under ~200ms/token gen
    assert p95 <= 80.0, f"F1 hot path p95={p95:.2f}ms > 80ms"


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU required",
)
def test_live_generate_emits_chaos(qwen_bundle):
    lens = FittedJacobianLens.load(LENS) if os.path.isfile(LENS) else None
    probe = ProbeEngine(qwen_bundle, lens)
    probe.setup()
    bus = ChaosBus()
    ids = qwen_bundle.encode("print(1+1)\n#", max_length=64)
    out = live_generate(
        qwen_bundle,
        ids,
        max_new_tokens=24,
        temperature=0.0,
        chaos_enabled=True,
        probe=probe,
        bus=bus,
        session_id="test",
        invocation_id="t1",
    )
    probe.remove_hooks()
    assert out["chaos"] is not None
    ch = out["chaos"]
    assert ch["version"] == 2
    assert len(ch["tokens"]) >= 1
    assert "A_vocab" in ch["tokens"][0]
    assert "B_vocab" in ch["tokens"][0]
    assert "C_vocab" in ch["tokens"][0]
    assert any(e.get("kind") == "prompt_end" for e in ch["events"])
    assert any(e.get("kind") == "gen_start" for e in ch["events"])
    assert "curves_E" in ch
