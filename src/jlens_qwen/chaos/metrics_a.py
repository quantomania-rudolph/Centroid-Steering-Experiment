"""A — Distribution shape metrics (Entropy-Lens / TriLens family)."""

from __future__ import annotations

import math
from typing import Any

import torch


def entropy_from_probs(probs: torch.Tensor) -> float:
    """Shannon entropy H = -sum p log p. Accepts 1-D probs."""
    p = probs.float().flatten()
    p = p.clamp(min=1e-12)
    return float(-(p * p.log()).sum().item())


def entropy_from_logits(logits: torch.Tensor) -> float:
    """True softmax entropy over full / partial logit vector."""
    p = torch.softmax(logits.float().flatten(), dim=-1)
    return entropy_from_probs(p)


def topk_probs(logits: torch.Tensor, k: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (probs_renormalized_over_topk, indices) for top-k."""
    vals, idx = logits.float().flatten().topk(min(k, logits.numel()))
    probs = torch.softmax(vals, dim=-1)
    return probs, idx


def margin(probs: torch.Tensor) -> float:
    """p1 - p2 (0 if fewer than 2 probs)."""
    p = probs.float().flatten()
    if p.numel() < 2:
        return float(p[0].item()) if p.numel() else 0.0
    top2 = p.topk(2).values
    return float((top2[0] - top2[1]).item())


def excess_kurtosis_logits(logits: torch.Tensor) -> float:
    """Excess kurtosis of logit scores (same as workspace_detect)."""
    x = logits.float().flatten()
    x = x - x.mean()
    m2 = (x**2).mean()
    m4 = (x**4).mean()
    if float(m2) < 1e-12:
        return 0.0
    return float((m4 / (m2**2) - 3.0).item())


def tail_mass(probs: torch.Tensor, from_rank: int = 6) -> float:
    """Sum of probability mass from rank `from_rank` (1-indexed) onward."""
    p = probs.float().flatten()
    if p.numel() < from_rank:
        return 0.0
    # ranks 6..k → indices 5..
    return float(p[from_rank - 1 :].sum().item())


def gini(probs: torch.Tensor) -> float:
    """Gini coefficient of probability mass: sum_i sum_j |pi-pj| / (2 k sum pi)."""
    p = probs.float().flatten()
    k = p.numel()
    if k == 0:
        return 0.0
    s = float(p.sum().item())
    if s < 1e-12:
        return 0.0
    # Efficient: sort then use formula
    ps = torch.sort(p).values
    idx = torch.arange(1, k + 1, dtype=p.dtype, device=p.device)
    # G = (2 * sum(i * p_i) / (k * sum p)) - (k+1)/k
    g = (2.0 * (idx * ps).sum() / (k * s)) - (k + 1) / k
    return float(g.item())


def distribution_shape(
    logits: torch.Tensor | None = None,
    *,
    probs: torch.Tensor | None = None,
    k: int = 20,
    full_entropy: bool = False,
) -> dict[str, float]:
    """Compute A metrics from logits (preferred) or precomputed top-k probs.

    When full_entropy=True and logits given, H is true softmax entropy;
    otherwise H is renormalized top-k entropy (F0 proxy).
    """
    if probs is None:
        if logits is None:
            raise ValueError("need logits or probs")
        top_p, _ = topk_probs(logits, k=k)
        work_logits = logits
    else:
        top_p = probs.float().flatten()
        work_logits = None

    if full_entropy and logits is not None:
        h = entropy_from_logits(logits)
    else:
        h = entropy_from_probs(top_p)

    out: dict[str, float] = {
        "entropy": h,
        "margin": margin(top_p),
        "tail_mass": tail_mass(top_p, from_rank=6),
        "gini": gini(top_p),
    }
    if work_logits is not None:
        # kurtosis on top-k logits for speed when full vector huge
        top_vals = work_logits.float().flatten().topk(min(k, work_logits.numel())).values
        out["kurtosis"] = excess_kurtosis_logits(top_vals)
    else:
        # approximate kurtosis from log-probs
        out["kurtosis"] = excess_kurtosis_logits(top_p.clamp(min=1e-12).log())
    return out


def a_scalar_from_proto_scores(scores: torch.Tensor) -> dict[str, float]:
    """Cheap A-like scalars from 4-class (or N-class) prototype softmax scores.

    Used on F1 hot path where we only have B_transport scores, not full vocab.
    """
    p = torch.softmax(scores.float().flatten(), dim=-1)
    return {
        "entropy": entropy_from_probs(p),
        "margin": margin(p),
        "tail_mass": tail_mass(p, from_rank=min(3, p.numel())),
        "gini": gini(p),
        "kurtosis": excess_kurtosis_logits(scores.float().flatten()),
    }
