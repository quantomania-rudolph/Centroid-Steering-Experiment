"""Per-metric microsecond timing histograms."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class TimingHist:
    """Simple histogram of microsecond samples per metric name."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)

    def reset(self) -> None:
        self._samples.clear()

    def record(self, name: str, microseconds: float) -> None:
        self._samples[name].append(float(microseconds))

    def percentile(self, name: str, pct: float) -> float:
        xs = sorted(self._samples.get(name, []))
        if not xs:
            return 0.0
        if pct <= 0:
            return xs[0]
        if pct >= 100:
            return xs[-1]
        # nearest-rank
        k = int(round((pct / 100.0) * (len(xs) - 1)))
        return xs[k]

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, xs in self._samples.items():
            if not xs:
                continue
            out[name] = {
                "n": len(xs),
                "p50_us": self.percentile(name, 50),
                "p95_us": self.percentile(name, 95),
                "max_us": max(xs),
                "mean_us": sum(xs) / len(xs),
            }
        return out

    def chaos_per_token_summary(self) -> dict[str, float]:
        """Aggregate F0+F1 samples as chaos_ms_per_token."""
        xs = self._samples.get("token_total", [])
        if not xs:
            # fallback: sum F0+F1 per step if available separately — use F1
            xs = self._samples.get("F1", []) + self._samples.get("F0", [])
        if not xs:
            return {
                "chaos_ms_per_token_p50": 0.0,
                "chaos_ms_per_token_p95": 0.0,
                "chaos_ms_total": 0.0,
            }
        # xs in microseconds
        p50 = self.percentile("token_total", 50) if "token_total" in self._samples else sorted(xs)[len(xs) // 2]
        if "token_total" in self._samples:
            p95 = self.percentile("token_total", 95)
            total = sum(self._samples["token_total"])
        else:
            sx = sorted(xs)
            p95 = sx[int(round(0.95 * (len(sx) - 1)))]
            total = sum(xs)
        return {
            "chaos_ms_per_token_p50": p50 / 1000.0,
            "chaos_ms_per_token_p95": p95 / 1000.0,
            "chaos_ms_total": total / 1000.0,
        }
