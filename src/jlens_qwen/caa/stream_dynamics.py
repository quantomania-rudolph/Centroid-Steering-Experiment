"""Stream dynamics helpers for CAST panic-window labeling (UNI_CAA01 §3.3)."""

from __future__ import annotations

from typing import Any, Callable

CAST_DISTRESS_THRESHOLD = 0.08
CAST_STUB_THRESHOLD = 0.05
CAST_POINTWISE_STRONG_THRESHOLD = 0.5
CAST_ROLLING_WINDOW = 16
CAST_MAX_STEP_FRAC = 0.8


def temporal_resolution(tokens: list[dict]) -> dict[str, Any]:
    """Compare flag rates in first vs last quartile of generation."""
    if len(tokens) < 8:
        return {"enough_tokens": False}
    q = max(1, len(tokens) // 4)
    first, last = tokens[:q], tokens[-q:]
    metrics: dict[str, Any] = {}
    for name, pred in [
        ("B_distress", lambda f: f.startswith("B_distress")),
        ("B_stub", lambda f: f.startswith("B_stub")),
        ("C", lambda f: f.startswith("C_")),
        ("A", lambda f: f.startswith("A_")),
    ]:
        def rate(rows: list[dict], predicate: Callable[[str], bool] = pred) -> float:
            if not rows:
                return 0.0
            n = sum(1 for t in rows if any(predicate(f) for f in (t.get("flags") or [])))
            return n / len(rows)

        f_rate, l_rate = rate(first), rate(last)
        metrics[name] = {
            "first_q": f_rate,
            "last_q": l_rate,
            "resolving": l_rate < f_rate * 0.5 if f_rate > 0.05 else None,
        }
    return {"enough_tokens": True, "metrics": metrics}


def non_resolving_from_temporal(temporal: dict[str, Any]) -> bool:
    """Classify panic episode as non-resolving from stored temporal_resolution output."""
    if not temporal.get("enough_tokens"):
        return False
    metrics = temporal.get("metrics") or {}
    any_elevated = False
    any_clearly_resolving = False
    for key in ("B_distress", "B_stub"):
        m = metrics.get(key) or {}
        fq = float(m.get("first_q") or 0)
        res = m.get("resolving")
        if fq > 0.04:
            any_elevated = True
        if res is True:
            any_clearly_resolving = True
        if res is False:
            return True
    return any_elevated and not any_clearly_resolving


def _token_masses(tok: dict) -> tuple[float, float]:
    b = tok.get("B_vocab") or tok
    return (
        float(b.get("distress_mass") or tok.get("B_distress_mass") or 0.0),
        float(b.get("stub_mass") or tok.get("B_stub_mass") or 0.0),
    )


def panic_window_indices(
    tokens: list[dict],
    *,
    max_new_tokens: int | None = None,
) -> list[int]:
    """Return gen-step indices in CAST panic windows (pointwise distress+stub gate)."""
    n = len(tokens)
    if n == 0:
        return []

    cap = int(max_new_tokens if max_new_tokens is not None else n)
    max_step = min(n, int(CAST_MAX_STEP_FRAC * cap))

    indices: list[int] = []
    for i in range(max_step):
        d_mass, s_mass = _token_masses(tokens[i])
        if (
            d_mass >= CAST_POINTWISE_STRONG_THRESHOLD
            and s_mass >= CAST_POINTWISE_STRONG_THRESHOLD
        ):
            indices.append(i)
    if indices:
        return indices

    if n < 8:
        return []

    temporal = temporal_resolution(tokens)
    if not temporal.get("enough_tokens"):
        return []

    metrics = temporal.get("metrics") or {}
    distress = metrics.get("B_distress") or {}
    stub = metrics.get("B_stub") or {}
    resolving = distress.get("resolving") is True or stub.get("resolving") is True
    if resolving:
        max_step = max(max_step, max(1, n // 4))

    indices = []
    for i in range(max_step):
        start = max(0, i - CAST_ROLLING_WINDOW + 1)
        chunk = tokens[start : i + 1]
        d_vals = [_token_masses(t)[0] for t in chunk]
        s_vals = [_token_masses(t)[1] for t in chunk]
        rolling_distress = sum(d_vals) / len(d_vals)
        rolling_stub = sum(s_vals) / len(s_vals)
        if (
            rolling_distress >= CAST_DISTRESS_THRESHOLD
            and rolling_stub >= CAST_STUB_THRESHOLD
        ):
            indices.append(i)
    return indices


def panic_window_indices_from_records(
    records: list[dict],
    *,
    max_new_tokens: int | None = None,
) -> list[int]:
    """Recompute CAST panic windows from replay JSONL token rows."""
    tokens = [
        {
            "flags": list(r.get("flags") or []),
            "B_vocab": {
                "distress_mass": r.get("B_distress_mass"),
                "stub_mass": r.get("B_stub_mass"),
            },
        }
        for r in records
    ]
    cap = max_new_tokens if max_new_tokens is not None else len(records)
    return panic_window_indices(tokens, max_new_tokens=cap)


def token_dict_from_live(
    *,
    distress_mass: float,
    stub_mass: float,
    flags: list[str] | None,
) -> dict[str, Any]:
    return {
        "flags": list(flags or []),
        "B_vocab": {
            "distress_mass": float(distress_mass),
            "stub_mass": float(stub_mass),
        },
    }


class RollingPanicWindow:
    """Incremental live g_stream; parity with offline panic_window_indices()."""

    def __init__(self) -> None:
        self._tokens: list[dict[str, Any]] = []
        self._max_new_tokens: int = 1024

    def reset(self, *, max_new_tokens: int = 1024) -> None:
        self._tokens = []
        self._max_new_tokens = int(max_new_tokens)

    def push(
        self,
        step: int,
        *,
        distress_mass: float,
        stub_mass: float,
        flags: list[str] | None = None,
        max_new_tokens: int | None = None,
    ) -> bool:
        if max_new_tokens is not None:
            self._max_new_tokens = int(max_new_tokens)
        self._tokens.append(
            token_dict_from_live(
                distress_mass=distress_mass,
                stub_mass=stub_mass,
                flags=flags,
            )
        )
        idxs = panic_window_indices(self._tokens, max_new_tokens=self._max_new_tokens)
        return int(step) in idxs

    def rolling_means(self) -> tuple[float, float]:
        if not self._tokens:
            return 0.0, 0.0
        chunk = self._tokens[-CAST_ROLLING_WINDOW:]
        d_vals = [
            float((t.get("B_vocab") or {}).get("distress_mass") or 0) for t in chunk
        ]
        s_vals = [float((t.get("B_vocab") or {}).get("stub_mass") or 0) for t in chunk]
        return sum(d_vals) / len(d_vals), sum(s_vals) / len(s_vals)

    def snapshot(self) -> dict[str, Any]:
        idxs = (
            panic_window_indices(self._tokens, max_new_tokens=self._max_new_tokens)
            if self._tokens
            else []
        )
        return {
            "n_tokens": len(self._tokens),
            "n_g_stream": len(idxs),
            "window_indices": idxs,
        }
