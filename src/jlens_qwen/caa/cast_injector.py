"""CAST live injector state machine + L16 hook (UNI_CAA01 §3.5–3.7)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from jlens_qwen.caa.activation_cast import ActivationCAST
from jlens_qwen.caa.cast_math import (
    alpha_eff_mode_a,
    g_recipe_from_match,
    severity_weight,
)
from jlens_qwen.caa.s_cond_replay import load_u_beh, load_u_cond
from jlens_qwen.caa.stream_dynamics import (
    CAST_DISTRESS_THRESHOLD,
    CAST_STUB_THRESHOLD,
    RollingPanicWindow,
)
from jlens_qwen.model_loader import ModelBundle

MAX_INJECT_TOKENS = 64
INJECT_LAYER = 16
PANIC_RECIPE = "B_STUB_DISTRESS"


@dataclass
class CastInjectController:
    """Mutable state shared with the L16 forward hook."""

    enabled: bool = False
    pending_alpha: float = 0.0
    u_beh: torch.Tensor | None = None
    last_inject_applied: bool = False

    def consume_pending(self) -> float:
        alpha = float(self.pending_alpha)
        self.pending_alpha = 0.0
        self.last_inject_applied = alpha > 0.0
        return alpha


class CastInjectorState:
    def __init__(self) -> None:
        self._episode_inject_tokens = 0
        self._stopped_early = False

    def reset(self) -> None:
        self._episode_inject_tokens = 0
        self._stopped_early = False

    def should_stop_early(self, rolling_distress: float, rolling_stub: float) -> bool:
        if self._stopped_early:
            return True
        if (
            rolling_distress < CAST_DISTRESS_THRESHOLD
            and rolling_stub < CAST_STUB_THRESHOLD
            and self._episode_inject_tokens > 0
        ):
            self._stopped_early = True
            return True
        return False

    def cap_alpha(self, alpha_eff: float) -> float:
        if alpha_eff <= 0.0:
            return 0.0
        if self._stopped_early:
            return 0.0
        if self._episode_inject_tokens >= MAX_INJECT_TOKENS:
            return 0.0
        return float(alpha_eff)

    def note_inject_applied(self, applied: bool) -> None:
        if applied:
            self._episode_inject_tokens += 1

    @property
    def episode_inject_tokens(self) -> int:
        return self._episode_inject_tokens


class CastLiveSession:
    """Coordinator for live CAST shadow/inject in the decode loop."""

    def __init__(
        self,
        metadata_path: str | Path,
        condition_pt: str | Path,
        behavior_pt: str | Path,
        *,
        shadow: bool = False,
        inject: bool = False,
        u_cond_override: torch.Tensor | None = None,
        u_beh_override: torch.Tensor | None = None,
        alpha_0_override: float | None = None,
    ) -> None:
        self.shadow = bool(shadow)
        self.inject = bool(inject)
        self.rolling = RollingPanicWindow()
        self.cast = ActivationCAST(
            metadata_path,
            condition_pt,
            u_cond_override=u_cond_override,
            alpha_0_override=alpha_0_override,
        )
        self.injector = CastInjectorState()
        self.controller = CastInjectController(
            enabled=bool(inject),
            u_beh=u_beh_override if u_beh_override is not None else load_u_beh(behavior_pt),
        )
        self._hook_handle: Any = None
        self._cast_eval_us: list[float] = []

    def reset(self, *, max_new_tokens: int) -> None:
        self.rolling.reset(max_new_tokens=max_new_tokens)
        self.injector.reset()
        self.controller.pending_alpha = 0.0
        self.controller.last_inject_applied = False
        self._cast_eval_us = []

    def register_hook(self, bundle: ModelBundle) -> None:
        if self._hook_handle is not None:
            return
        layers = bundle.layers
        u_beh = self.controller.u_beh
        controller = self.controller
        if u_beh is not None:
            layer = layers[INJECT_LAYER]

            def hook(_module: Any, _inp: Any, out: Any) -> Any:
                if not controller.enabled or controller.pending_alpha <= 0.0 or u_beh is None:
                    return out
                alpha = controller.consume_pending()
                if alpha <= 0.0:
                    return out
                h = out[0] if isinstance(out, tuple) else out
                u = u_beh.to(device=h.device, dtype=h.dtype)
                h[0, -1, :].add_(u, alpha=alpha)
                return out

            self._hook_handle = layer.register_forward_hook(hook)

    def remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def process_token(
        self,
        step: int,
        *,
        distress_mass: float,
        stub_mass: float,
        flags: list[str],
        recipe_id: str | None,
        severity: str | None,
        h_l16: torch.Tensor | None,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        t0 = time.perf_counter_ns()
        g_stream = int(
            self.rolling.push(
                step,
                distress_mass=distress_mass,
                stub_mass=stub_mass,
                flags=flags,
                max_new_tokens=max_new_tokens,
            )
        )
        cast_out = self.cast.evaluate(h_l16)
        g_recipe = g_recipe_from_match(recipe_id, severity)
        gamma_val = float(cast_out["gamma"])
        gamma_zeroed = bool(cast_out.get("gamma_zeroed_by_tau_hard"))
        w_sev = severity_weight(severity)
        alpha_raw = alpha_eff_mode_a(
            alpha_0=self.cast.alpha_0,
            w_sev=w_sev,
            gamma_val=gamma_val,
            g_recipe=g_recipe,
            g_stream=g_stream,
        )
        if self.injector.should_stop_early(*self.rolling.rolling_means()):
            alpha_raw = 0.0
        alpha_eff = self.injector.cap_alpha(alpha_raw)

        inject_applied = False
        if self.inject and alpha_eff > 0.0:
            self.controller.pending_alpha = alpha_eff
        else:
            self.controller.pending_alpha = 0.0

        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        self._cast_eval_us.append(elapsed_us)

        return {
            "g_recipe": g_recipe,
            "g_stream": g_stream,
            "s_cond": float(cast_out["s_cond"]),
            "gamma": gamma_val,
            "gamma_zeroed_by_tau_hard": gamma_zeroed,
            "alpha_eff": float(alpha_eff),
            "inject_applied": inject_applied,
            "cast_eval_us": float(elapsed_us),
            "recipe_id": recipe_id,
            "severity": severity,
        }

    def note_post_forward(self) -> bool:
        applied = bool(self.controller.last_inject_applied)
        self.injector.note_inject_applied(applied)
        self.controller.last_inject_applied = False
        return applied

    def timing_summary(self) -> dict[str, float]:
        xs = sorted(self._cast_eval_us)
        if not xs:
            return {"n": 0.0, "p50_us": 0.0, "p95_us": 0.0, "max_us": 0.0}
        n = len(xs)
        return {
            "n": float(n),
            "p50_us": xs[n // 2],
            "p95_us": xs[int(n * 0.95)],
            "max_us": xs[-1],
        }


def patch_last_token_fields(bus: Any, fields: dict[str, Any]) -> None:
    with bus._lock:
        if bus._tokens:
            bus._tokens[-1].update(fields)
