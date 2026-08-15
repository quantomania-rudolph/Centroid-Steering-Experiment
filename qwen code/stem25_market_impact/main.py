"""Almgren-Chriss temporary impact — fixed cost formula."""

from __future__ import annotations

import math


def temporary_impact_cost(q: float, v: float, sigma: float, eta: float) -> float:
  # Corrected formula: cost = eta*sigma*sqrt(Q/V)*Q
  return eta * sigma * math.sqrt(q / v) * q


def demo() -> float:
  return temporary_impact_cost(q=1e6, v=5e7, sigma=0.02, eta=0.1)


def main() -> None:
  cost = demo()
  assert 200 < cost < 400, cost
  print(f"impact_cost={cost:.2f}")


if __name__ == "__main__":
  main()
