"""Shared CAST math: s_cond, gamma, alpha_eff (UNI_CAA01 §3)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from jlens_qwen.caa.kv_equivalence import cosine_similarity

PANIC_RECIPE_ID = "B_STUB_DISTRESS"
SEVERITY_WARN = frozenset({"warn", "critical"})


def load_cast_thresholds(metadata_path: str | Path) -> dict[str, float]:
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    tau = meta.get("tau")
    sigma = meta.get("sigma")
    if tau is None or sigma is None:
        raise ValueError(f"metadata missing tau/sigma: {metadata_path}")
    alpha_0 = float(meta.get("alpha_0") or 0.85)
    env_alpha = os.environ.get("CAA_ALPHA_0")
    if env_alpha:
        alpha_0 = float(env_alpha)
    return {
        "tau": float(tau),
        "sigma": float(sigma),
        "tau_hard": float(meta.get("tau_hard") if meta.get("tau_hard") is not None else tau),
        "alpha_0": alpha_0,
        "vector_id": str(meta.get("vector_id") or "UNI_CAA01"),
    }


def s_cond(h: torch.Tensor, u_cond: torch.Tensor) -> float:
    return cosine_similarity(h.float().flatten(), u_cond.float().flatten())


def gamma(
    s_cond_val: float,
    tau: float,
    sigma: float,
    tau_hard: float | None = None,
) -> float:
    if tau_hard is not None and s_cond_val < tau_hard:
        return 0.0
    x = (s_cond_val - tau) / max(sigma, 1e-6)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def g_recipe_from_match(recipe_id: str | None, severity: str | None) -> int:
    if recipe_id == PANIC_RECIPE_ID and (severity or "") in SEVERITY_WARN:
        return 1
    return 0


def severity_weight(severity: str | None) -> float:
    if severity == "critical":
        return 1.2
    return 1.0


def alpha_eff_mode_a(
    *,
    alpha_0: float,
    w_sev: float,
    gamma_val: float,
    g_recipe: int,
    g_stream: int,
) -> float:
    return float(alpha_0 * w_sev * gamma_val * g_recipe * g_stream)
