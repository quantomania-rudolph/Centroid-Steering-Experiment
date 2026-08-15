"""Dose accumulation — broken R factor."""

from __future__ import annotations

import math


def accumulation_factor(k: float, tau: float) -> float:
  # BUG: uses exp(+k*tau) instead of exp(-k*tau)
  return 1 / (1 - math.exp(k * tau))


def demo() -> float:
  return accumulation_factor(k=0.05, tau=7.5)


def main() -> None:
  r = demo()
  assert 3.0 < r < 3.5, r
  print(f"R={r:.4f}")


if __name__ == "__main__":
  main()
