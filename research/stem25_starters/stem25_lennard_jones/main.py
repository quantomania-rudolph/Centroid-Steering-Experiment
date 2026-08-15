"""Lennard-Jones force — wrong derivative."""

from __future__ import annotations


def lj_force(r: float, sigma: float, epsilon: float) -> float:
  x = sigma / r
  # BUG: missing factor of 24/r
  return epsilon * (2 * x**12 - x**6)


def demo() -> float:
  return lj_force(r=0.9 * 1.0, sigma=1.0, epsilon=1.0)


def main() -> None:
  f = demo()
  assert f > 0, f  # repulsive at r < sigma
  print(f"F={f:.4f}")


if __name__ == "__main__":
  main()
