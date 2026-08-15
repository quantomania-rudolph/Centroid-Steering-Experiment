"""Persistent probe-layer hooks + J-transport helpers for live chaos."""

from __future__ import annotations

from typing import Any

import torch

from jlens_qwen.chaos.constants import PROBE_LAYERS, TOP_K
from jlens_qwen.chaos.lexicon import build_lexicon_masks, lexicon_summary
from jlens_qwen.chaos.metrics_a import distribution_shape, topk_probs
from jlens_qwen.chaos.metrics_b import (
    build_prototypes_from_masks,
    masses_from_topk,
    transport_masses,
    transport_scores,
)
from jlens_qwen.jacobian_lens import FittedJacobianLens, transport
from jlens_qwen.logit_lens import rmsnorm
from jlens_qwen.model_loader import ModelBundle, materialize_gamma_cpu, materialize_w_u_cpu


class ProbeEngine:
    """Register once; each forward fills last-position residuals for probe layers."""

    def __init__(
        self,
        bundle: ModelBundle,
        lens: FittedJacobianLens | None,
        *,
        probe_layers: list[int] | None = None,
    ):
        self.bundle = bundle
        self.lens = lens
        self.probe_layers = list(probe_layers or PROBE_LAYERS)
        self.residuals: dict[int, torch.Tensor] = {}
        self._hooks: list[Any] = []
        self.masks: dict[str, torch.Tensor] = {}
        self.prototypes: dict[int, torch.Tensor] = {}
        self.J_xpu: dict[int, torch.Tensor] = {}
        self.W_U_cpu: torch.Tensor | None = None
        self._ready = False

    def setup(self) -> dict[str, Any]:
        """Build lexicon, prototypes, move J to device, register hooks."""
        if self._ready:
            return {
                "probe_layers": self.probe_layers,
                "lexicon": lexicon_summary(self.masks) if self.masks else {},
                "has_lens": self.lens is not None,
                "n_hooks": len(self._hooks),
                "cached": True,
            }
        tok = self.bundle.tokenizer
        self.masks = build_lexicon_masks(tok, vocab_size=self.bundle.vocab_size)
        # Move W_U to CPU before float32 cast — full V float32 on XPU OOMs Arc 16GB
        self.W_U_cpu = materialize_w_u_cpu(self.bundle)
        self.gamma_cpu = materialize_gamma_cpu(self.bundle)
        self._use_cpu_unembed = getattr(self.bundle.W_U, "is_meta", False)
        if str(self.bundle.device).startswith("xpu"):
            try:
                torch.xpu.empty_cache()
            except Exception:
                pass
        proto_cpu = build_prototypes_from_masks(self.W_U_cpu, self.masks)
        device = self.bundle.device
        dtype = self.bundle.dtype

        for L in self.probe_layers:
            if self.lens is not None and L in self.lens.jacobians:
                J = self.lens.jacobians[L].to(device=device, dtype=torch.float16)
                self.J_xpu[L] = J
            self.prototypes[L] = proto_cpu.to(device=device, dtype=torch.float32)

        self._register_hooks()
        self._ready = True
        return {
            "probe_layers": self.probe_layers,
            "lexicon": lexicon_summary(self.masks),
            "has_lens": self.lens is not None,
            "n_hooks": len(self._hooks),
        }

    def _register_hooks(self) -> None:
        self.remove_hooks()
        layers = self.bundle.layers

        def make_hook(idx: int):
            def hook(_module, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                # last position only, detach
                self.residuals[idx] = h[:, -1, :].detach()

            return hook

        for i in self.probe_layers:
            self._hooks.append(layers[i].register_forward_hook(make_hook(i)))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear_residuals(self) -> None:
        self.residuals.clear()

    def transport_last(self, layer: int) -> torch.Tensor | None:
        """RMSNorm(J @ h) at last position for probe layer → [d] on device float32."""
        h = self.residuals.get(layer)
        if h is None:
            return None
        h = h[0] if h.dim() == 2 else h.flatten()
        J = self.J_xpu.get(layer)
        if J is not None:
            hp = transport(h.unsqueeze(0).to(dtype=J.dtype, device=J.device), J)[0]
        else:
            hp = h
        gamma = self.gamma_cpu.to(device=hp.device)
        eps = self.bundle.rms_eps
        return rmsnorm(hp.unsqueeze(0), gamma, eps)[0].float()

    def _unembed_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Final RMSNorm + lm_head; CPU path when device_map left head on meta."""
        if not self._use_cpu_unembed:
            return self.bundle.unembed(h.unsqueeze(0) if h.dim() == 1 else h)[0]
        hp = h[0] if h.dim() == 2 else h.flatten()
        hp_cpu = hp.detach().float().cpu()
        z = rmsnorm(hp_cpu.unsqueeze(0), self.gamma_cpu, self.bundle.rms_eps)[0]
        return torch.nn.functional.linear(
            z.unsqueeze(0).to(self.W_U_cpu.dtype), self.W_U_cpu
        )[0]

    def hot_metrics_at(self, layer: int) -> dict[str, Any] | None:
        """Single transport → B_transport + A_scalar (F1 hot path)."""
        from jlens_qwen.chaos.metrics_a import a_scalar_from_proto_scores

        hp = self.transport_last(layer)
        if hp is None:
            return None
        proto = self.prototypes[layer]
        bt = transport_masses(hp, proto)
        scores = transport_scores(hp, proto)
        return {"B_transport": bt, "A": a_scalar_from_proto_scores(scores)}

    @torch.no_grad()
    def b_jlens_topk(self, layer: int, k: int = TOP_K) -> dict[str, Any] | None:
        """Full unembed top-k + B masses (F2/F3 only — expensive)."""
        hp = self.transport_last(layer)
        if hp is None:
            return None
        logits = self._unembed_logits(hp)
        probs, idx = topk_probs(logits, k=k)
        masks = {n: self.masks[n].to(idx.device) for n in self.masks}
        masses = masses_from_topk(idx, probs, masks)
        shape = distribution_shape(logits, k=k, full_entropy=False)
        tokens = [
            {
                "id": int(i),
                "token": self.bundle.tokenizer.decode([int(i)]),
                "prob": float(p),
            }
            for p, i in zip(probs.tolist(), idx.tolist())
        ]
        return {
            "A": shape,
            "B_jlens": masses,
            "top20": tokens,
            "ids": idx.cpu(),
            "probs": probs.cpu(),
        }

    @torch.no_grad()
    def logit_lens_topk(self, layer: int, k: int = TOP_K) -> dict[str, Any] | None:
        """Naive unembed without J transport."""
        h = self.residuals.get(layer)
        if h is None:
            return None
        h = h[0] if h.dim() == 2 else h.flatten()
        logits = self._unembed_logits(h)
        probs, idx = topk_probs(logits, k=k)
        return {
            "ids": idx.cpu(),
            "probs": probs.cpu(),
            "id_list": [int(x) for x in idx.tolist()],
        }

    def vocab_ab(
        self, logits: torch.Tensor, *, full_entropy: bool = False
    ) -> dict[str, Any]:
        """A_vocab + B_vocab from sampling-head logits."""
        probs, idx = topk_probs(logits, k=TOP_K)
        masks = {n: self.masks[n].to(idx.device) for n in self.masks}
        a = distribution_shape(logits, k=TOP_K, full_entropy=full_entropy)
        b = masses_from_topk(idx, probs, masks)
        return {
            "A_vocab": a,
            "B_vocab": b,
            "ids": idx.cpu(),
            "probs": probs.cpu(),
            "top1_id": int(idx[0].item()),
        }
