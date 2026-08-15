"""Lorenz-96 — broken periodic RHS."""

from __future__ import annotations

import math


def lorenz96_rhs(x: list[float], f: float) -> list[float]:
  n = len(x)
  out = []
  for i in range(n):
    # Fixed index wrap (uses i-2 instead of i+2)
    out.append((x[(i + 1) % n] - x[(i - 2) % n]) * x[(i - 1) % n] - x[i] + f)
  return out


def rk4_step(x: list[float], f: float, dt: float) -> list[float]:
  k1 = lorenz96_rhs(x, f)
  k2 = lorenz96_rhs([xi + 0.5 * dt * ki for xi, ki in zip(x, k1)], f)
  k3 = lorenz96_rhs([xi + 0.5 * dt * ki for xi, ki in zip(x, k2)], f)
  k4 = lorenz96_rhs([xi + dt * ki for xi, ki in zip(x, k3)], f)
  return [xi + dt * (k1i + 2 * k2i + 2 * k3i + k4i) / 6 for xi, k1i, k2i, k3i, k4i in zip(x, k1, k2, k3, k4)]


def demo() -> float:
  x0 = [1.0, 0.0, 0.0, 0.0, 0.0]
  x1 = rk4_step(x0, f=8.0, dt=0.01)
  return math.sqrt(sum(v * v for v in x1))


def main() -> None:
  norm = demo()
  assert 0.1 < norm < 5.0, norm
  print(f"norm={norm:.4f}")


if __name__ == "__main__":
  main()
