"""D — Cross-position contrast: ΔH, ΔM_code, Jensen–Shannon."""

from __future__ import annotations

import math
from typing import Any

import torch

from jlens_qwen.chaos.metrics_a import entropy_from_probs


def _align_distributions(
    ids_a: torch.Tensor,
    probs_a: torch.Tensor,
    ids_b: torch.Tensor,
    probs_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align two sparse top-k distributions onto shared id union → dense probs."""
    union = torch.unique(torch.cat([ids_a.long().flatten(), ids_b.long().flatten()]))
    pa = torch.zeros(union.numel(), dtype=torch.float32)
    pb = torch.zeros(union.numel(), dtype=torch.float32)
    id_to_i = {int(u): i for i, u in enumerate(union.tolist())}
    for tid, pr in zip(ids_a.flatten().tolist(), probs_a.flatten().tolist()):
        pa[id_to_i[int(tid)]] = float(pr)
    for tid, pr in zip(ids_b.flatten().tolist(), probs_b.flatten().tolist()):
        pb[id_to_i[int(tid)]] = float(pr)
    # renormalize
    sa, sb = pa.sum(), pb.sum()
    if sa > 0:
        pa = pa / sa
    if sb > 0:
        pb = pb / sb
    return pa, pb


def kl_div(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.clamp(min=1e-12)
    q = q.clamp(min=1e-12)
    return float((p * (p.log() - q.log())).sum().item())


def js_divergence(
    ids_a: torch.Tensor,
    probs_a: torch.Tensor,
    ids_b: torch.Tensor,
    probs_b: torch.Tensor,
) -> float:
    """Symmetric Jensen–Shannon divergence on top-k distributions (nats)."""
    pa, pb = _align_distributions(ids_a, probs_a, ids_b, probs_b)
    m = 0.5 * (pa + pb)
    return 0.5 * kl_div(pa, m) + 0.5 * kl_div(pb, m)


def cross_position(
    *,
    entropy_a: float,
    entropy_b: float,
    code_mass_a: float,
    code_mass_b: float,
    ids_a: torch.Tensor,
    probs_a: torch.Tensor,
    ids_b: torch.Tensor,
    probs_b: torch.Tensor,
) -> dict[str, float]:
    """ΔH, ΔM_code, JS between anchors a and b."""
    return {
        "delta_H": float(entropy_b - entropy_a),
        "delta_M_code": float(code_mass_b - code_mass_a),
        "js_div": js_divergence(ids_a, probs_a, ids_b, probs_b),
        "entropy_a": float(entropy_a),
        "entropy_b": float(entropy_b),
    }
