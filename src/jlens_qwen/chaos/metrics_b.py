"""B — Token-class hygiene: vocab / jlens / transport tracks."""

from __future__ import annotations

from typing import Any

import torch


CLASS_NAMES = ("code", "distress", "stub", "junk")


def masses_from_topk(
    token_ids: torch.Tensor,
    probs: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute class masses from top-k token ids and renormalized probs.

    masks[class] is BoolTensor[V] or LongTensor of member ids.
    """
    ids = token_ids.long().flatten()
    p = probs.float().flatten()
    if ids.numel() != p.numel():
        raise ValueError("token_ids and probs length mismatch")
    out: dict[str, float] = {}
    for name in CLASS_NAMES:
        m = masks[name]
        if m.dtype == torch.bool:
            # gather membership
            member = m[ids]
        else:
            # set of ids
            member = torch.isin(ids, m.long())
        out[f"{name}_mass"] = float((p * member.float()).sum().item())
    return out


def transport_masses(
    h_prime: torch.Tensor,
    prototypes: torch.Tensor,
) -> dict[str, float]:
    """B_transport: softmax over 4 prototype dots on normalized transported h.

    prototypes: [4, d] (code, distress, stub, junk)
    h_prime: [d] already RMSNorm(J @ h) or transported residual
    """
    h = h_prime.float().flatten()
    n = torch.linalg.norm(h)
    if float(n) < 1e-12:
        return {f"{n}_mass": 0.25 for n in CLASS_NAMES}
    h = h / n
    P = prototypes.float()
    # normalize prototype rows
    pn = torch.linalg.norm(P, dim=-1, keepdim=True).clamp(min=1e-12)
    P = P / pn
    scores = P @ h  # [4]
    p = torch.softmax(scores, dim=-1)
    return {f"{name}_mass": float(p[i].item()) for i, name in enumerate(CLASS_NAMES)}


def transport_scores(h_prime: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Raw prototype scores (pre-softmax) for A_scalar."""
    h = h_prime.float().flatten()
    n = torch.linalg.norm(h)
    if float(n) < 1e-12:
        return torch.zeros(prototypes.shape[0])
    h = h / n
    P = prototypes.float()
    pn = torch.linalg.norm(P, dim=-1, keepdim=True).clamp(min=1e-12)
    P = P / pn
    return P @ h


def build_prototypes_from_masks(
    W_U: torch.Tensor,
    masks: dict[str, torch.Tensor],
    *,
    max_per_class: int = 256,
) -> torch.Tensor:
    """Mean of unit-normalized embedding rows for each class → [4, d]."""
    rows = []
    for name in CLASS_NAMES:
        m = masks[name]
        if m.dtype == torch.bool:
            idx = m.nonzero(as_tuple=False).flatten()
        else:
            idx = m.long().flatten()
        if idx.numel() == 0:
            rows.append(torch.zeros(W_U.shape[1], dtype=torch.float32))
            continue
        if idx.numel() > max_per_class:
            # subsample for speed
            perm = torch.randperm(idx.numel())[:max_per_class]
            idx = idx[perm]
        emb = W_U.float()[idx]  # [n, d]
        norms = torch.linalg.norm(emb, dim=-1, keepdim=True).clamp(min=1e-12)
        emb = emb / norms
        rows.append(emb.mean(dim=0))
    return torch.stack(rows, dim=0)  # [4, d]
