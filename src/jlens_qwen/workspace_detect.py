"""Workspace boundary metrics from the J-space paper."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from jlens_qwen.jacobian_lens import FittedJacobianLens
from jlens_qwen.logit_lens import LogitLens
from jlens_qwen.model_loader import ModelBundle


def excess_kurtosis(logits: torch.Tensor) -> float:
    """Excess kurtosis of softmax distribution (high = peaked)."""
    p = torch.softmax(logits.float(), dim=-1)
    # For discrete distribution, use kurtosis of probability mass over ranks
    # Paper uses excess kurtosis of the logit/score distribution
    x = logits.float().flatten()
    x = x - x.mean()
    m2 = (x**2).mean()
    m4 = (x**4).mean()
    if m2 < 1e-12:
        return 0.0
    return float((m4 / (m2**2) - 3.0).item())


def topk_match_accuracy(
    lens_logits: torch.Tensor,
    model_logits: torch.Tensor,
    k: int = 5,
) -> float:
    """Fraction of positions where any top-k lens token matches model top-1."""
    if lens_logits.dim() == 1:
        lens_logits = lens_logits.unsqueeze(0)
        model_logits = model_logits.unsqueeze(0)
    model_top1 = model_logits.argmax(dim=-1)
    lens_topk = lens_logits.topk(k, dim=-1).indices
    hits = (lens_topk == model_top1.unsqueeze(-1)).any(dim=-1)
    return float(hits.float().mean().item())


def top1_autocorrelation(top1_ids: torch.Tensor, delta: int = 1) -> float:
    """Rate at which top-1 token stays the same across positions."""
    if top1_ids.numel() <= delta:
        return 0.0
    same = (top1_ids[:-delta] == top1_ids[delta:]).float().mean()
    return float(same.item())


def effective_dimensionality(
    J: torch.Tensor,
    W_U: torch.Tensor | None = None,
    *,
    variance_share: float = 0.9,
    max_samples: int = 2048,
) -> float:
    """Fraction of residual dims needed for `variance_share` of J-lens vector variance.

    Uses SVD of a random sample of rows of W_U @ J (or of J if W_U is None).
    Returns fraction in (0, 1].
    """
    M = J.float()
    if W_U is not None:
        # Sample vocab rows for tractability
        V = W_U.shape[0]
        idx = torch.randperm(V)[: min(max_samples, V)]
        M = W_U.float()[idx] @ J.float()  # [S, d]
    # Center
    M = M - M.mean(dim=0, keepdim=True)
    # Economy SVD
    try:
        _, S, _ = torch.linalg.svd(M, full_matrices=False)
    except RuntimeError:
        return 0.0
    var = S**2
    total = var.sum()
    if total < 1e-12:
        return 0.0
    cum = torch.cumsum(var, dim=0) / total
    k = int((cum >= variance_share).nonzero()[0].item()) + 1
    return k / M.shape[1]


def compute_layer_metrics(
    bundle: ModelBundle,
    lens: FittedJacobianLens | None,
    prompts: list[str],
    *,
    layers: list[int] | None = None,
    max_seq_len: int = 128,
    use_jacobian: bool = True,
) -> dict[str, Any]:
    """Compute paper metrics across layers for workspace boundary detection."""
    if layers is None:
        layers = list(range(bundle.n_layers))
    logit_lens = LogitLens(bundle)

    per_layer: dict[int, dict[str, float]] = {
        l: {
            "topk_acc": [],
            "kurtosis": [],
            "autocorr": [],
        }
        for l in layers
    }

    for prompt in prompts:
        if lens is not None and use_jacobian:
            result = lens.apply(
                bundle,
                prompt,
                layers=layers,
                max_seq_len=max_seq_len,
                use_jacobian=True,
            )
        else:
            result = logit_lens.project_layers(
                prompt, layers=layers, max_seq_len=max_seq_len
            )

        model_logits = result["model_logits"]  # [T, V]
        # Align: model logits at position t predict token t+1;
        # use positions 0..T-2 vs model logits at those positions
        T = model_logits.shape[0]
        for layer in layers:
            ll = result["lens_logits"][layer]  # [T, V]
            # Compare lens at t with model next-token at t (same as paper approx)
            acc = topk_match_accuracy(ll[:-1], model_logits[:-1], k=5)
            kurt = float(
                np.mean([excess_kurtosis(ll[t]) for t in range(0, T, max(1, T // 8))])
            )
            top1 = ll.argmax(dim=-1)
            ac = top1_autocorrelation(top1, delta=1)
            per_layer[layer]["topk_acc"].append(acc)
            per_layer[layer]["kurtosis"].append(kurt)
            per_layer[layer]["autocorr"].append(ac)

    summary: dict[int, dict[str, float]] = {}
    for layer in layers:
        summary[layer] = {
            "topk_acc": float(np.mean(per_layer[layer]["topk_acc"] or [0.0])),
            "kurtosis": float(np.mean(per_layer[layer]["kurtosis"] or [0.0])),
            "autocorr": float(np.mean(per_layer[layer]["autocorr"] or [0.0])),
            "effective_dim": 0.0,
        }
        if lens is not None and layer in lens.jacobians:
            summary[layer]["effective_dim"] = effective_dimensionality(
                lens.jacobians[layer],
                bundle.W_U.cpu(),
            )

    onset = detect_onset(summary, threshold=0.15)
    motor = detect_motor(summary, threshold=0.60)
    kurt_peak = max(summary, key=lambda l: summary[l]["kurtosis"])
    ac_peak = max(summary, key=lambda l: summary[l]["autocorr"])

    return {
        "per_layer": {str(k): v for k, v in summary.items()},
        "onset_layer": onset,
        "motor_layer": motor,
        "kurtosis_peak_layer": kurt_peak,
        "autocorr_peak_layer": ac_peak,
        "workspace_band": [onset, motor] if onset is not None and motor is not None else None,
    }


def detect_onset(summary: dict[int, dict[str, float]], threshold: float = 0.15) -> int | None:
    layers = sorted(summary)
    for l in layers:
        if summary[l]["topk_acc"] >= threshold:
            return l
    return None


def detect_motor(summary: dict[int, dict[str, float]], threshold: float = 0.60) -> int | None:
    layers = sorted(summary)
    for l in layers:
        if summary[l]["topk_acc"] >= threshold:
            return l
    return None


def band_overlap(detected: tuple[int, int], hypothesis: tuple[int, int] = (11, 22)) -> int:
    a0, a1 = detected
    b0, b1 = hypothesis
    return max(0, min(a1, b1) - max(a0, b0))
