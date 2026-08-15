"""Live ActivationCAST: s_cond and gamma from L16 residual (UNI_CAA01 §3.4)."""

from __future__ import annotations

from pathlib import Path

import torch

from jlens_qwen.caa.cast_math import gamma, load_cast_thresholds, s_cond
from jlens_qwen.caa.s_cond_replay import load_u_cond


class ActivationCAST:
    def __init__(
        self,
        metadata_path: str | Path,
        condition_pt: str | Path,
        *,
        u_cond_override: torch.Tensor | None = None,
        alpha_0_override: float | None = None,
    ) -> None:
        thresholds = load_cast_thresholds(metadata_path)
        self.tau = thresholds["tau"]
        self.sigma = thresholds["sigma"]
        self.tau_hard = thresholds["tau_hard"]
        self.alpha_0 = (
            float(alpha_0_override)
            if alpha_0_override is not None
            else thresholds["alpha_0"]
        )
        self.vector_id = thresholds["vector_id"]
        self.u_cond = u_cond_override if u_cond_override is not None else load_u_cond(condition_pt)

    def evaluate(self, h_l16: torch.Tensor | None) -> dict[str, float | bool]:
        if h_l16 is None:
            return {
                "s_cond": 0.0,
                "gamma": 0.0,
                "gamma_zeroed_by_tau_hard": False,
            }
        sc = s_cond(h_l16, self.u_cond)
        g = gamma(sc, self.tau, self.sigma, self.tau_hard)
        zeroed = sc < self.tau_hard
        return {
            "s_cond": float(sc),
            "gamma": float(g),
            "gamma_zeroed_by_tau_hard": bool(zeroed),
        }
