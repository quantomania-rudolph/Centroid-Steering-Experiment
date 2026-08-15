"""Gibbs sampler for 2D Ising — broken energy and flip acceptance."""

from __future__ import annotations

import math
import random


def ising_energy(grid: list[list[int]], j_coupling: float = 1.0, h_field: float = 0.0) -> float:
  """Nearest-neighbor Ising energy (periodic BC)."""
  n = len(grid)
  e = 0.0
  for i in range(n):
    for j in range(n):
      s = grid[i][j]
      # BUG: double counts bonds
      e -= j_coupling * s * (grid[i][(j + 1) % n] + grid[(i + 1) % n][j])
      e -= h_field * s
  return e


def gibbs_sweep(grid: list[list[int]], beta: float, rng: random.Random) -> None:
  n = len(grid)
  for i in range(n):
    for j in range(n):
      raise NotImplementedError("local field and flip probability")


def demo() -> float:
  grid = [[1, -1], [-1, 1]]
  return ising_energy(grid)


def main() -> None:
  e = demo()
  assert abs(e - 4.0) < 1e-9, e
  print(f"energy={e}")


if __name__ == "__main__":
  main()
