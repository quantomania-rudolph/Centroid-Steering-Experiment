"""ChaosBus — unified in-memory source of truth for A–G live metrics."""

from __future__ import annotations

import threading
import time
from typing import Any

from jlens_qwen.chaos.constants import (
    CHAOS_SCHEMA_VERSION,
    DEGRADE_P95_MS,
    DEGRADE_STREAK,
    F2_STRIDE,
    PROBE_LAYERS,
)
from jlens_qwen.chaos.flags import (
    THRESH,
    composite_fci,
    composite_plds,
    composite_srs,
    evaluate_flags,
    flag_categories,
)
from jlens_qwen.chaos.metrics_e import layer_curve_metrics
from jlens_qwen.chaos.schema import empty_chaos_record
from jlens_qwen.chaos.timing import TimingHist


class ChaosBus:
    """Thread-safe ring of per-token chaos metrics for one live invocation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.session_id: str | None = None
        self.invocation_id: str | None = None
        self.meta: dict[str, Any] = {}
        self.device: str | None = None
        self.degraded_mode: bool = False
        self._degrade_streak: int = 0
        self._tokens: list[dict[str, Any]] = []
        self._token_id_stream: list[int] = []
        self._events: list[dict[str, Any]] = []
        self._flags_timeline: list[dict[str, Any]] = []
        self.timing = TimingHist()
        self._gen_ms_total: float = 0.0
        self._probe_series: dict[str, dict[int, list[float]]] = {
            "entropy": {L: [] for L in PROBE_LAYERS},
            "code_mass": {L: [] for L in PROBE_LAYERS},
            "distress_mass": {L: [] for L in PROBE_LAYERS},
            "flip_rate": {L: [] for L in PROBE_LAYERS},
        }
        self._last_f: dict[str, Any] | None = None
        self._last_g: dict[str, Any] | None = None
        self._curves_E: dict[str, Any] = {}
        self._composites: dict[str, Any] = {}
        self._active: bool = False
        # First-cross indices on the full stream (immune to tokens[] downsample).
        self._first_d_lexicon: int | None = None
        self._first_d_flags: int | None = None
        self._first_d_distress: int | None = None
        self._first_d_stub: int | None = None

    def begin_invocation(
        self,
        session_id: str,
        invocation_id: str,
        meta: dict[str, Any] | None = None,
        *,
        device: str | None = None,
    ) -> None:
        with self._lock:
            self.session_id = session_id
            self.invocation_id = invocation_id
            self.meta = dict(meta or {})
            self.device = device
            self.degraded_mode = False
            self._degrade_streak = 0
            self._tokens.clear()
            self._token_id_stream.clear()
            self._events.clear()
            self._flags_timeline.clear()
            self.timing.reset()
            self._gen_ms_total = 0.0
            for series in self._probe_series.values():
                for L in series:
                    series[L].clear()
            self._last_f = None
            self._last_g = None
            self._curves_E = {}
            self._composites = {}
            self._active = True
            self._first_d_lexicon = None
            self._first_d_flags = None
            self._first_d_distress = None
            self._first_d_stub = None

    def record_token(self, step: int, token_id: int, payload: dict[str, Any]) -> list[str]:
        """Record F0/F1(/F2) row; return firing flags."""
        with self._lock:
            flags = evaluate_flags(payload)
            row = {
                "step": int(step),
                "token_id": int(token_id),
                **payload,
                "flags": flags,
            }
            self._tokens.append(row)
            self._token_id_stream.append(int(token_id))
            s = int(step)
            b = payload.get("B_vocab") or {}
            dm = float(b.get("distress_mass", 0))
            sm = float(b.get("stub_mass", 0))
            if self._first_d_distress is None and dm >= THRESH["B_distress"]:
                self._first_d_distress = s
            if self._first_d_stub is None and sm >= THRESH["B_stub"]:
                self._first_d_stub = s
            if (
                self._first_d_lexicon is None
                and dm >= THRESH["B_distress"]
                and sm >= THRESH["B_stub"]
            ):
                self._first_d_lexicon = s
            if (
                self._first_d_flags is None
                and "B_distress" in flags
                and "B_stub" in flags
            ):
                self._first_d_flags = s
            if flags:
                self._flags_timeline.append({"step": step, "flags": flags})
            # probe series for early E hints
            probes = payload.get("probes") or {}
            for L_str, pdata in probes.items():
                try:
                    L = int(L_str)
                except ValueError:
                    continue
                if L not in PROBE_LAYERS:
                    continue
                a = pdata.get("A") or {}
                b = pdata.get("B_transport") or {}
                c = pdata.get("C") or {}
                if L in self._probe_series["entropy"]:
                    self._probe_series["entropy"][L].append(float(a.get("entropy", 0)))
                    self._probe_series["code_mass"][L].append(
                        float(b.get("code_mass", 0))
                    )
                    self._probe_series["distress_mass"][L].append(
                        float(b.get("distress_mass", 0))
                    )
                    self._probe_series["flip_rate"][L].append(
                        float(c.get("flip_rate", 0))
                    )
            return flags

    def estimate_cliff_metrics(self) -> dict[str, Any]:
        """Running-mean cliff metrics from probe entropy series collected so far."""
        with self._lock:
            series: dict[int, float] = {}
            ent_by_L = self._probe_series.get("entropy") or {}
            for L, xs in ent_by_L.items():
                if xs:
                    series[L] = sum(xs) / len(xs)
            if len(series) < 2:
                return {"cliff_layer": None, "cliff_mag": 0.0}
            return layer_curve_metrics(series)

    def estimate_cliff_layer(self) -> int | None:
        """Running-mean cliff layer from probe entropy series collected so far."""
        cliff = self.estimate_cliff_metrics()
        layer = cliff.get("cliff_layer")
        return int(layer) if layer is not None else None

    def record_event(self, kind: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._events.append({"kind": kind, "t": time.time(), **payload})

    def record_timing(self, metric_name: str, microseconds: float) -> None:
        with self._lock:
            self.timing.record(metric_name, microseconds)
            if metric_name == "token_total":
                ms = microseconds / 1000.0
                if ms > DEGRADE_P95_MS:
                    self._degrade_streak += 1
                    if self._degrade_streak >= DEGRADE_STREAK:
                        self.degraded_mode = True
                else:
                    self._degrade_streak = 0

    def set_gen_ms(self, ms: float) -> None:
        with self._lock:
            self._gen_ms_total = float(ms)

    def record_f(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._last_f = payload
            self._events.append({"kind": "F_disagreement", **payload})

    def record_g(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._last_g = payload
            self._events.append({"kind": "G_sparse", **payload})

    def should_fire_g(self, flags: list[str], step: int) -> bool:
        """True if ≥2 flag categories and debounce ok."""
        from jlens_qwen.chaos.constants import F5_MIN_FLAG_CATEGORIES, G_DEBOUNCE_TOKENS

        cats = flag_categories(flags)
        if len(cats) < F5_MIN_FLAG_CATEGORIES:
            return False
        with self._lock:
            last_g_step = -10_000
            for ev in reversed(self._events):
                if ev.get("kind") == "G_sparse":
                    last_g_step = int(ev.get("step", -10_000))
                    break
            return (step - last_g_step) >= G_DEBOUNCE_TOKENS

    def record_invocation_summary(self, payload: dict[str, Any]) -> None:
        with self._lock:
            if "curves_E" in payload:
                self._curves_E = payload["curves_E"]
            if "composites" in payload:
                self._composites = payload["composites"]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": self._active,
                "session_id": self.session_id,
                "invocation_id": self.invocation_id,
                "n_tokens": len(self._tokens),
                "last_flags": (self._flags_timeline[-1] if self._flags_timeline else None),
                "degraded_mode": self.degraded_mode,
                "timing": self.timing.summary(),
                "probe_layers": list(PROBE_LAYERS),
            }

    def finalize(self) -> dict[str, Any]:
        """Compact ChaosInvocationRecord for disk / finalize JSON."""
        with self._lock:
            # Full stream survives downsample (needed for Method B prefix decode).
            token_ids = list(self._token_id_stream)
            n_tokens_full = len(token_ids)
            b_distress_n = sum(
                1 for t in self._tokens if "B_distress" in (t.get("flags") or [])
            )
            b_stub_n = sum(
                1 for t in self._tokens if "B_stub" in (t.get("flags") or [])
            )
            divergence_hints = {
                "d_lexicon": self._first_d_lexicon,
                "d_flags": self._first_d_flags,
                "d_distress_only": self._first_d_distress,
                "d_stub_only": self._first_d_stub,
                "b_distress_rate": (b_distress_n / n_tokens_full if n_tokens_full else 0.0),
                "b_stub_rate": (b_stub_n / n_tokens_full if n_tokens_full else 0.0),
            }

            # downsample long gens: keep all flagged + every 2nd otherwise
            tokens = self._tokens
            if len(tokens) > 512:
                kept: list[dict[str, Any]] = []
                for i, row in enumerate(tokens):
                    if row.get("flags") or (i % 2 == 0):
                        kept.append(row)
                tokens = kept

            if not self._composites:
                self._composites = {
                    "PLDS": composite_plds(self._tokens),
                    "FCI": composite_fci(self._tokens, self._events),
                    "SRS": composite_srs(self._tokens),
                    "flags_timeline": list(self._flags_timeline),
                }

            # probe means for early E
            probe_means: dict[str, dict[str, float]] = {}
            for name, by_L in self._probe_series.items():
                probe_means[name] = {
                    str(L): (sum(xs) / len(xs) if xs else 0.0) for L, xs in by_L.items()
                }

            timing = self.timing.chaos_per_token_summary()
            timing["gen_ms_total"] = self._gen_ms_total
            timing["hist"] = self.timing.summary()

            recommended = []
            if self._curves_E.get("cliff_layer") is not None:
                recommended = [int(self._curves_E["cliff_layer"])]
            elif PROBE_LAYERS:
                recommended = [PROBE_LAYERS[1]]  # L16 default

            rec = empty_chaos_record(device=self.device)
            rec.update(
                {
                    "version": CHAOS_SCHEMA_VERSION,
                    "session_id": self.session_id,
                    "invocation_id": self.invocation_id,
                    "tokens": tokens,
                    "token_ids": token_ids,
                    "n_tokens_full": n_tokens_full,
                    "divergence_hints": divergence_hints,
                    "events": list(self._events),
                    "curves_E": self._curves_E or {"probe_means": probe_means},
                    "composites": self._composites,
                    "steering_hooks": {
                        "recommended_layers": recommended,
                        "disagreement_F": self._last_f,
                        "g_residual": self._last_g,
                    },
                    "timing": timing,
                    "degraded_mode": self.degraded_mode,
                    "flags_timeline": list(self._flags_timeline),
                    "f2_stride": F2_STRIDE,
                    "probe_means": probe_means,
                }
            )
            self._active = False
            return rec
