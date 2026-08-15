"""G — Sparse J-space residual (alarm-gated)."""

from __future__ import annotations

from typing import Any

import torch

from jlens_qwen.jspace_decompose import decompose_activation


def sparse_residual(
    h: torch.Tensor,
    W_U: torch.Tensor,
    J: torch.Tensor,
    *,
    k: int = 25,
    max_dict: int = 2048,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Wrap decompose_activation; return relMSE, support size, top atoms."""
    result = decompose_activation(h, W_U, J, k=k, max_dict=max_dict)
    idxs = result["vocab_indices"]
    coefs = result["coefficients"]
    atoms: list[dict[str, Any]] = []
    if idxs is not None and hasattr(idxs, "numel") and idxs.numel():
        for i in range(min(10, idxs.numel())):
            vid = int(idxs[i].item())
            tok = tokenizer.decode([vid]) if tokenizer is not None else str(vid)
            atoms.append(
                {
                    "id": vid,
                    "token": tok,
                    "coeff": float(coefs[i].item()) if coefs.numel() > i else 0.0,
                }
            )
    return {
        "relMSE": float(result["relative_mse"]),
        "support_size": int(idxs.numel()) if hasattr(idxs, "numel") else 0,
        "k": int(result["k"]),
        "top_atoms": atoms,
    }
