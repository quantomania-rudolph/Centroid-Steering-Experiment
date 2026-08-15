"""Exact logit lens: final RMSNorm + unembedding (J_l = I)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from jlens_qwen.model_loader import ActivationRecorder, ModelBundle


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Qwen2 RMSNorm matching HuggingFace Qwen2RMSNorm.

    RMS(x) = sqrt(mean(x^2) + eps)
    RMSNorm(x) = weight * x / RMS(x)

    Variance is computed in float32, then cast back to input dtype.
    """
    input_dtype = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(variance + eps)
    return (weight.to(torch.float32) * x32).to(input_dtype)


class LogitLens:
    """Project intermediate residuals through final norm + W_U."""

    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self.gamma = bundle.gamma
        self.eps = bundle.rms_eps
        self.W_U = bundle.W_U
        self.final_norm = bundle.final_norm

    def project_residual(self, h: torch.Tensor) -> torch.Tensor:
        """logits = W_U @ RMSNorm(h). Shape [..., vocab]."""
        return self.bundle.unembed(h)

    def project_residual_manual(self, h: torch.Tensor) -> torch.Tensor:
        """Hand-rolled RMSNorm + matmul for parity tests."""
        z = rmsnorm(h, self.gamma, self.eps)
        return torch.nn.functional.linear(z.to(self.W_U.dtype), self.W_U)

    @torch.no_grad()
    def project_layers(
        self,
        prompt: str,
        *,
        layers: list[int] | None = None,
        positions: list[int] | None = None,
        max_seq_len: int = 512,
    ) -> dict[str, Any]:
        """Run forward, capture residuals, return per-layer logits."""
        bundle = self.bundle
        n_layers = bundle.n_layers
        if layers is None:
            layers = list(range(n_layers))
        input_ids = bundle.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(bundle.layers, at=layers) as recorder:
            outputs = bundle.model(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=False,
            )
        model_logits = outputs.logits[0]  # [T, V]
        lens_logits: dict[int, torch.Tensor] = {}
        for layer in layers:
            full = recorder.activations[layer][0]  # [T, d]
            selected = full if positions is None else full[list(positions)]
            lens_logits[layer] = self.project_residual(selected).float().cpu()

        if positions is None:
            model_at = model_logits.float().cpu()
        else:
            model_at = model_logits[list(positions)].float().cpu()

        return {
            "input_ids": input_ids.cpu(),
            "lens_logits": lens_logits,
            "model_logits": model_at,
            "activations": {
                k: v.detach().float().cpu() for k, v in recorder.activations.items()
            },
        }

    def topk_tokens(
        self,
        logits: torch.Tensor,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        values, indices = logits.float().topk(k)
        tok = self.bundle.tokenizer
        return [
            (tok.decode([int(i)]), float(v))
            for v, i in zip(values.tolist(), indices.tolist(), strict=True)
        ]
