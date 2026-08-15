"""Jacobian lens apply: transport + final norm + unembed."""

from __future__ import annotations

from typing import Any

import torch

from jlens_qwen.model_loader import ActivationRecorder, ModelBundle


def transport(residual: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """Map residual into final-layer basis: h @ J.T  (row-vector / jlens convention).

    Equivalent to J @ h for column vectors. J shape [d, d], residual [..., d].
    """
    J_bar = J.to(device=residual.device, dtype=torch.float32)
    h = residual.to(torch.float32)
    return h @ J_bar.T


def apply_jacobian_lens(
    residual: torch.Tensor,
    J: torch.Tensor,
    bundle: ModelBundle,
) -> torch.Tensor:
    """lens(h) = W_U · RMSNorm(J @ h). Returns logits [..., vocab]."""
    transported = transport(residual, J)
    return bundle.unembed(transported.to(bundle.dtype))


def jlens_vectors(W_U: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """Rows of W_U @ J — one vector per vocab token. Shape [V, d]."""
    return W_U.float() @ J.float()


class FittedJacobianLens:
    """Thin wrapper around per-layer J_l matrices (compatible with jlens save format)."""

    def __init__(
        self,
        jacobians: dict[int, torch.Tensor],
        *,
        n_prompts: int,
        d_model: int,
    ):
        self.jacobians = {int(k): v.float() for k, v in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = n_prompts
        self.d_model = d_model

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        return transport(residual, self.jacobians[layer])

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        torch.save(
            {
                "J": {layer: J.to(dtype) for layer, J in self.jacobians.items()},
                "n_prompts": self.n_prompts,
                "source_layers": self.source_layers,
                "d_model": self.d_model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> FittedJacobianLens:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if "J" in ckpt:
            return cls(
                jacobians=ckpt["J"],
                n_prompts=int(ckpt["n_prompts"]),
                d_model=int(ckpt["d_model"]),
            )
        # jlens.fit checkpoint mid-run
        raise ValueError(f"Not a JacobianLens file: keys={sorted(ckpt)}")

    @classmethod
    def merge(cls, lenses: list[FittedJacobianLens]) -> FittedJacobianLens:
        if not lenses:
            raise ValueError("merge needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if other.source_layers != first.source_layers or other.d_model != first.d_model:
                raise ValueError("lenses disagree on source_layers / d_model")
        n_total = sum(lens.n_prompts for lens in lenses)
        merged: dict[int, torch.Tensor] = {}
        for layer in first.source_layers:
            weighted = sum(
                lens.jacobians[layer] * lens.n_prompts for lens in lenses
            )
            merged[layer] = weighted / n_total
        return cls(jacobians=merged, n_prompts=n_total, d_model=first.d_model)

    @torch.no_grad()
    def apply(
        self,
        bundle: ModelBundle,
        prompt: str,
        *,
        layers: list[int] | None = None,
        positions: list[int] | None = None,
        max_seq_len: int = 512,
        use_jacobian: bool = True,
    ) -> dict[str, Any]:
        if layers is None:
            layers = self.source_layers
        final_layer = bundle.n_layers - 1
        record_at = sorted(set(layers) | {final_layer})
        input_ids = bundle.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(bundle.layers, at=record_at) as recorder:
            outputs = bundle.model(input_ids=input_ids, use_cache=False)
        model_logits = outputs.logits[0]

        def select(layer: int) -> torch.Tensor:
            full = recorder.activations[layer][0]
            return (full if positions is None else full[list(positions)]).float()

        lens_logits: dict[int, torch.Tensor] = {}
        for layer in layers:
            residual = select(layer)
            if use_jacobian and layer in self.jacobians:
                residual = self.transport(residual, layer)
            lens_logits[layer] = bundle.unembed(residual.to(bundle.dtype)).float().cpu()

        model_at = (
            model_logits.float().cpu()
            if positions is None
            else model_logits[list(positions)].float().cpu()
        )
        return {
            "input_ids": input_ids.cpu(),
            "lens_logits": lens_logits,
            "model_logits": model_at,
        }
