"""MonitorChaosBus — ultra-low-latency MonitorEvent ring + recipe timeline."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable

from jlens_qwen.chaos.bus import ChaosBus
from jlens_qwen.chaos.flags import e_cliff_alarm
from jlens_qwen.chaos.recipes import (
    ATOM_KEYS,
    RecipeMatcher,
    atoms_from_flags,
    f_disagree_from_payload,
)

MonitorCallback = Callable[[dict[str, Any]], None]

RING_SIZE = 4096
PUBLISH_BUDGET_US = 100.0


class MonitorChaosBus(ChaosBus):
    """ChaosBus with recipe matching and push-friendly monitor events."""

    def __init__(self) -> None:
        super().__init__()
        self._matcher = RecipeMatcher()
        self._monitor_ring: deque[dict[str, Any]] = deque(maxlen=RING_SIZE)
        self._recipe_timeline: list[dict[str, Any]] = []
        self._live_state: dict[str, Any] = {}
        self._subscribers: list[MonitorCallback] = []
        self._seq: int = 0
        self._cliff_layer: int | None = None
        self._cliff_mag: float = 0.0
        self._g_oob_active: bool = False
        self._publish_us_samples: list[float] = []
        self._rolling_plds: float = 0.0
        self._rolling_srs: float = 0.0
        self._rolling_n: int = 0

    def begin_invocation(
        self,
        session_id: str,
        invocation_id: str,
        meta: dict[str, Any] | None = None,
        *,
        device: str | None = None,
    ) -> None:
        super().begin_invocation(session_id, invocation_id, meta, device=device)
        self._matcher.reset()
        self._recipe_timeline.clear()
        self._monitor_ring.clear()
        self._seq = 0
        self._cliff_layer = None
        self._cliff_mag = 0.0
        self._g_oob_active = False
        self._publish_us_samples.clear()
        self._rolling_plds = 0.0
        self._rolling_srs = 0.0
        self._rolling_n = 0
        self._live_state = {
            "session_id": session_id,
            "invocation_id": invocation_id,
            "step": -1,
            "recipe_id": None,
            "severity": "none",
            "atoms": {k: 0 for k in ATOM_KEYS},
            "degraded_mode": False,
        }

    def subscribe(self, callback: MonitorCallback) -> None:
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: MonitorCallback) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def set_cliff_layer(self, layer: int | None) -> None:
        with self._lock:
            self._cliff_layer = layer

    def set_cliff_metrics(self, layer: int | None, mag: float) -> None:
        with self._lock:
            self._cliff_layer = layer
            self._cliff_mag = float(mag)

    def _e_cliff_active(self) -> bool:
        with self._lock:
            return e_cliff_alarm(self._cliff_layer, self._cliff_mag)

    def set_g_oob(self, active: bool) -> None:
        with self._lock:
            self._g_oob_active = bool(active)

    def record_monitor_event(
        self,
        step: int,
        token_id: int,
        flags: list[str],
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Evaluate recipe and publish MonitorEvent. Returns event if recipe fired."""
        t0 = time.perf_counter_ns()
        f_dis = f_disagree_from_payload(payload)
        atoms = atoms_from_flags(
            flags,
            f_disagree=f_dis,
            e_cliff=self._e_cliff_active(),
            g_oob=self._g_oob_active,
        )
        match = self._matcher.evaluate(step, atoms, cliff_layer=self._cliff_layer)

        a = payload.get("A_vocab") or {}
        b = payload.get("B_vocab") or {}
        c = payload.get("C_vocab") or {}
        tok_score = (
            0.25 * float(a.get("entropy", 0))
            + 0.25 * (1.0 - float(b.get("code_mass", 0)))
            + 0.25 * float(c.get("flip_rate", 0))
            + 0.25 * float(b.get("distress_mass", 0))
        )
        srs_score = (
            0.4 * float(b.get("stub_mass", 0))
            + 0.4 * float(b.get("distress_mass", 0))
            + 0.2 * (1.0 - float(b.get("code_mass", 0)))
        )
        self._rolling_n += 1
        n = self._rolling_n
        self._rolling_plds += (tok_score - self._rolling_plds) / n
        self._rolling_srs += (srs_score - self._rolling_srs) / n
        rolling = {
            "PLDS": self._rolling_plds,
            "FCI": self._rolling_plds + float(c.get("flip_rate", 0)),
            "SRS": self._rolling_srs,
        }

        event: dict[str, Any] = {
            "seq": self._seq,
            "t_ns": time.perf_counter_ns(),
            "session_id": self.session_id,
            "invocation_id": self.invocation_id,
            "step": int(step),
            "token_id": int(token_id),
            "atoms": atoms,
            "recipe_id": match.recipe_id if match else None,
            "severity": match.severity if match else "none",
            "alpha_hint": match.alpha_hint if match else 0.0,
            "layer_hint": list(match.layer_hint) if match else [],
            "caa_tags": list(match.caa_tags) if match else [],
            "composites_rolling": rolling,
            "degraded_mode": self.degraded_mode,
            "flags": list(flags),
        }
        self._seq += 1

        publish_us = (time.perf_counter_ns() - t0) / 1000.0
        self._publish_us_samples.append(publish_us)
        if len(self._publish_us_samples) > 512:
            self._publish_us_samples = self._publish_us_samples[-256:]

        with self._lock:
            self._monitor_ring.append(event)
            self._live_state = {
                "session_id": self.session_id,
                "invocation_id": self.invocation_id,
                "step": step,
                "recipe_id": event["recipe_id"],
                "severity": event["severity"],
                "atoms": dict(atoms),
                "alpha_hint": event["alpha_hint"],
                "layer_hint": event["layer_hint"],
                "degraded_mode": self.degraded_mode,
                "seq": event["seq"],
                "t_ns": event["t_ns"],
            }
            subs = list(self._subscribers)
            if match:
                self._recipe_timeline.append(
                    {
                        "step": step,
                        "recipe_id": match.recipe_id,
                        "severity": match.severity,
                        "atoms_mask": match.atoms_mask,
                        "alpha_hint": match.alpha_hint,
                        "layer_hint": match.layer_hint,
                    }
                )

        for cb in subs:
            try:
                cb(event)
            except Exception:
                pass

        self.timing.record("recipe_eval_publish", publish_us)
        return event

    def publish_cast_update(self, step: int, cast: dict[str, Any]) -> dict[str, Any] | None:
        """Attach CAST telemetry to the last monitor event and re-notify SSE subscribers."""
        with self._lock:
            if not self._monitor_ring or self._monitor_ring[-1].get("step") != step:
                return None
            event = dict(self._monitor_ring[-1])
            event["cast"] = dict(cast)
            self._monitor_ring[-1] = event
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(event)
            except Exception:
                pass
        return event

    def live_state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._live_state)

    def monitor_events_since(self, seq: int) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self._monitor_ring if int(e.get("seq", -1)) > seq]

    def snapshot(self) -> dict[str, Any]:
        base = super().snapshot()
        with self._lock:
            base["monitor"] = {
                "live": dict(self._live_state),
                "n_events": len(self._monitor_ring),
                "n_recipe_fires": len(self._recipe_timeline),
                "last_recipe": (
                    self._recipe_timeline[-1] if self._recipe_timeline else None
                ),
                "recipe_eval": self._matcher.eval_timing_summary(),
                "publish": self._publish_timing_summary(),
            }
        return base

    def _publish_timing_summary(self) -> dict[str, float]:
        xs = sorted(self._publish_us_samples)
        if not xs:
            return {"n": 0, "p50_us": 0.0, "p95_us": 0.0, "max_us": 0.0}
        n = len(xs)
        return {
            "n": float(n),
            "p50_us": xs[n // 2],
            "p95_us": xs[int(n * 0.95)],
            "max_us": xs[-1],
        }

    def finalize(self) -> dict[str, Any]:
        rec = super().finalize()
        with self._lock:
            rec["monitor"] = {
                "recipe_timeline": list(self._recipe_timeline),
                "events_tail": list(self._monitor_ring)[-64:],
                "recipe_eval": self._matcher.eval_timing_summary(),
                "publish": self._publish_timing_summary(),
                "n_recipe_fires": len(self._recipe_timeline),
            }
        return rec
