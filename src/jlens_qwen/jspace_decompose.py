"""Sparse nonnegative J-space decomposition via greedy gradient pursuit."""

from __future__ import annotations

import torch


def gradient_pursuit(
    h: torch.Tensor,
    dictionary: torch.Tensor,
    *,
    k: int = 25,
    n_iter: int = 50,
    lr: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Approximate h ≈ dictionary[S].T @ alpha with alpha >= 0, |S| <= k.

    Args:
        h: [d] activation
        dictionary: [V, d] J-lens vectors (rows of W_U @ J)
        k: sparsity budget
        n_iter: refinement steps after support selection
        lr: step size for nonnegative least-squares refinement

    Returns:
        (indices [k], coefficients [k], relative_mse)
    """
    h = h.float().flatten()
    d = h.numel()
    V = dictionary.shape[0]
    # Greedy: repeatedly pick the dictionary atom with largest positive correlation
    residual = h.clone()
    selected: list[int] = []
    for _ in range(min(k, V)):
        # correlations
        scores = dictionary @ residual  # [V]
        scores[selected] = float("-inf")
        # prefer nonnegative contribution
        idx = int(scores.argmax().item())
        if scores[idx] <= 0 and selected:
            break
        selected.append(idx)
        # Project residual onto span of selected (nonnegative LS via projected GD)
        atoms = dictionary[selected]  # [s, d]
        alpha = torch.zeros(len(selected), dtype=torch.float32)
        for _ in range(n_iter):
            pred = alpha @ atoms
            grad = atoms @ (pred - h)
            alpha = torch.clamp(alpha - lr * grad, min=0.0)
        residual = h - (alpha @ atoms)

    if not selected:
        return (
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0),
            1.0,
        )

    atoms = dictionary[selected]
    alpha = torch.zeros(len(selected), dtype=torch.float32)
    for _ in range(n_iter * 2):
        pred = alpha @ atoms
        grad = atoms @ (pred - h)
        alpha = torch.clamp(alpha - lr * grad, min=0.0)
    recon = alpha @ atoms
    mse = float(((h - recon) ** 2).sum().item())
    denom = float((h**2).sum().item()) + 1e-12
    return torch.tensor(selected, dtype=torch.long), alpha, mse / denom


def decompose_activation(
    h: torch.Tensor,
    W_U: torch.Tensor,
    J: torch.Tensor,
    *,
    k: int = 25,
    max_dict: int = 2048,
) -> dict[str, object]:
    """Decompose activation into sparse J-space coordinates.

    For tractability, subsample the vocabulary dictionary to `max_dict` rows
    with highest |W_U @ (J @ h)| scores, then run gradient pursuit.
    """
    h = h.float().flatten().cpu()
    J = J.float().cpu()
    W_U = W_U.float().cpu()
    transported = J @ h  # column convention for score
    # Score without materializing full W@J: use W @ transported
    scores = W_U @ transported  # [V]
    top = scores.abs().topk(min(max_dict, scores.numel())).indices
    dictionary = W_U[top] @ J  # [S, d]
    idx_local, coef, rel_mse = gradient_pursuit(h, dictionary, k=k)
    vocab_indices = top[idx_local] if idx_local.numel() else torch.zeros(0, dtype=torch.long)
    return {
        "vocab_indices": vocab_indices,
        "coefficients": coef,
        "relative_mse": rel_mse,
        "k": k,
    }
