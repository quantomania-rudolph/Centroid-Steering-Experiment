"""WKB tunneling — broken kappa."""

from __future__ import annotations

import math


def wkb_transmission(m: float, v0: float, e: float, a: float, hbar: float = 1.0) -> float:
  # Corrected calculation of kappa
  kappa = math.sqrt(2 * m * max(v0 - e, 1e-12)) / hbar
  return math.exp(-2 * kappa * a)


def demo() -> float:
  return wkb_transmission(m=1.0, v0=5.0, e=2.0, a=1.0)


def main() -> None:
  t = demo()
  assert 0 < t < 0.1, t
  print(f"T={t:.6f}")


if __name__ == "__main__":
  main()
