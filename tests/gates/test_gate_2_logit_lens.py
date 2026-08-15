"""GATE-2: Logit lens math proofs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from jlens_qwen.logit_lens import LogitLens, rmsnorm
from jlens_qwen.model_loader import load_qwen_coder
from tests.gates import save_gate_report, summarize_checks

ROOT = Path(__file__).resolve().parents[2]
PROMPT = "Fact: The capital of France is Paris. The currency of France is"


@pytest.fixture(scope="module")
def bundle(qwen_bundle):
    return qwen_bundle


@pytest.mark.gate2
@pytest.mark.slow
def test_gate2_logit_lens(bundle):
    checks = []
    lens = LogitLens(bundle)
    d = bundle.d_model
    gamma = bundle.gamma
    eps = bundle.rms_eps

    # G2.1 RMSNorm parity
    h = torch.randn(16, d, device=bundle.device, dtype=torch.float16)
    ours = rmsnorm(h, gamma, eps)
    ref = bundle.final_norm(h)
    diff = float((ours.float() - ref.float()).abs().max().item())
    checks.append(
        {
            "id": "G2.1",
            "name": "RMSNorm parity vs model.norm",
            # fp16 ulp noise on Arc; G2.3 end-to-end parity is the hard proof
            "pass": diff < 0.02,
            "detail": f"max_abs_diff={diff}",
        }
    )

    # G2.2 fp16 vs fp32 norm
    h32 = h.float()
    n16 = rmsnorm(h, gamma, eps).float()
    n32 = rmsnorm(h32, gamma.float(), eps)
    diff16 = float((n16 - n32).abs().max().item())
    checks.append(
        {
            "id": "G2.2",
            "name": "fp16/fp32 norm stability",
            "pass": diff16 < 1e-2,
            "detail": f"max_abs_diff={diff16}",
        }
    )

    # G2.3 Layer-27 logit lens == model logits
    result = lens.project_layers(PROMPT, layers=[0, 27], max_seq_len=128)
    ll27 = result["lens_logits"][27]
    model_logits = result["model_logits"]
    # model.logits from CausalLM already includes final norm+lm_head on last hidden
    # Our hook at layer 27 is post-block residual BEFORE model.norm — same as last_hidden before norm?
    # Qwen2Model applies norm AFTER all layers, so last_hidden_state = norm(layer27_out).
    # CausalLM logits = lm_head(last_hidden) = lm_head(norm(h27)).
    # Our logit lens does the same: unembed(h27) = lm_head(norm(h27)).
    # So they should match.
    diff27 = float((ll27 - model_logits).abs().max().item())
    checks.append(
        {
            "id": "G2.3",
            "name": "Layer-27 logit lens == model logits",
            "pass": diff27 < 0.05,
            "detail": f"max_abs_diff={diff27}",
        }
    )
    (ROOT / "artifacts" / "gates" / "gate2_layer27_parity.json").write_text(
        __import__("json").dumps({"max_abs_diff": diff27, "pass": diff27 < 0.05}, indent=2),
        encoding="utf-8",
    )

    # G2.4 Layer-0 differs
    ll0 = result["lens_logits"][0]
    diff0 = float((ll0 - model_logits).abs().max().item())
    checks.append(
        {
            "id": "G2.4",
            "name": "Layer-0 != model logits",
            "pass": diff0 > 1.0,
            "detail": f"max_abs_diff={diff0}",
        }
    )

    # G2.5 softmax sums
    s = torch.softmax(ll27[0], dim=-1).sum().item()
    checks.append(
        {
            "id": "G2.5",
            "name": "softmax sums to 1",
            "pass": abs(s - 1.0) < 1e-5,
            "detail": f"sum={s}",
        }
    )

    # G2.6 W_U not tied
    tied = bundle.W_U.data_ptr() == bundle.model.model.embed_tokens.weight.data_ptr()
    checks.append(
        {
            "id": "G2.6",
            "name": "W_U not tied to embed",
            "pass": not tied,
            "detail": f"tied={tied}",
        }
    )

    (ROOT / "artifacts" / "gates" / "gate2_rmsnorm.json").write_text(
        __import__("json").dumps({"G2.1_diff": diff, "G2.2_diff": diff16}, indent=2),
        encoding="utf-8",
    )

    summary = summarize_checks(checks)
    save_gate_report("GATE-2", summary)
    assert summary["overall_pass"], summary
