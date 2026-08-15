"""Colebrook friction — broken iteration."""

from __future__ import annotations

import math


def colebrook_f(re: float, rel_rough: float, iters: int = 20) -> float:
  f = 0.02
  for _ in range(iters):
    # BUG: wrong constant 2.51 -> 25.1
    inv_sqrt = -2 * math.log10(rel_rough / 3.7 + 25.1 / (re * math.sqrt(f)))
    f = 1 / (inv_sqrt * inv_sqrt)
  return f


def demo() -> float:
  return colebrook_f(re=1e5, rel_rough=0.0004)


def main() -> None:
  f = demo()
  assert 0.018 < f < 0.022, f
  print(f"f={f:.5f}")


if __name__ == "__main__":
  main()
