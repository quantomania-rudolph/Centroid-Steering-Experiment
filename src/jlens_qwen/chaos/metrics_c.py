"""C — Trajectory stability over a rolling window of top-1 token ids."""

from __future__ import annotations

from collections import deque
from typing import Any


class TrajectoryTracker:
    """Rolling window tracker for flip / autocorr / unique / oscillation / settling."""

    def __init__(self, window: int = 16, settle_eps: float = 0.25):
        self.window = window
        self.settle_eps = settle_eps
        self._buf: deque[int] = deque(maxlen=window)
        self._settling_tau: int | None = None
        self._step = 0

    def reset(self) -> None:
        self._buf.clear()
        self._settling_tau = None
        self._step = 0

    def push(self, token_id: int) -> dict[str, float]:
        self._step += 1
        self._buf.append(int(token_id))
        metrics = self.metrics()
        if self._settling_tau is None and len(self._buf) >= self.window:
            if metrics["flip_rate"] < self.settle_eps:
                self._settling_tau = self._step
        metrics["settling_tau"] = float(
            self._settling_tau if self._settling_tau is not None else -1.0
        )
        return metrics

    def metrics(self) -> dict[str, float]:
        ids = list(self._buf)
        n = len(ids)
        if n < 2:
            return {
                "flip_rate": 0.0,
                "autocorr": 1.0 if n else 0.0,
                "unique_ratio": 1.0 if n else 0.0,
                "oscillation": 0.0,
                "window_full": 0.0,
            }
        flips = sum(1 for i in range(n - 1) if ids[i] != ids[i + 1])
        flip_rate = flips / (n - 1)
        unique_ratio = len(set(ids)) / n
        osc = 0
        if n >= 3:
            for i in range(n - 2):
                if ids[i] == ids[i + 2] and ids[i] != ids[i + 1]:
                    osc += 1
            osc_rate = osc / (n - 2)
        else:
            osc_rate = 0.0
        return {
            "flip_rate": float(flip_rate),
            "autocorr": float(1.0 - flip_rate),
            "unique_ratio": float(unique_ratio),
            "oscillation": float(osc_rate),
            "window_full": float(n >= self.window),
        }


def trajectory_from_sequence(
    token_ids: list[int],
    *,
    window: int | None = None,
    settle_eps: float = 0.25,
) -> dict[str, float]:
    """Compute C metrics over a full sequence (optionally last `window` tokens)."""
    ids = list(token_ids)
    if window is not None:
        ids = ids[-window:]
    tracker = TrajectoryTracker(window=max(len(ids), 1), settle_eps=settle_eps)
    out: dict[str, float] = {}
    for tid in ids:
        out = tracker.push(tid)
    return out
