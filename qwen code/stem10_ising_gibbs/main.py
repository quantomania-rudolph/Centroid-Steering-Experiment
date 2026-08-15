"""Gibbs sampler for 2D Ising — energy and flip acceptance."""

from __future__ import annotations

import math
import random


def ising_energy(grid: list[list[int]], j_coupling: float = 1.0, h_field: float = 0.0) -> float:
  """Nearest-neighbor Ising energy; count each bond once (right + down, open BC)."""
  n = len(grid)
  e = 0.0
  for i in range(n):
    for j in range(n):
      s = grid[i][j]
      if j + 1 < n:
        e -= j_coupling * s * grid[i][j + 1]
      if i + 1 < n:
        e -= j_coupling * s * grid[i + 1][j]
      e -= h_field * s
  return e


def gibbs_sweep(grid: list[list[int]], beta: float, rng: random.Random) -> None:
  n = len(grid)
  for i in range(n):
    for j in range(n):
      s = grid[i][j]
      # Calculate the energy difference if we flip the spin
      delta_e = 2 * j_coupling * s * (grid[i][(j + 1) % n] + grid[(i + 1) % n][j])
      # Accept the flip with probability exp(-beta * delta_e)
      if delta_e < 0 or rng.random() < math.exp(-beta * delta_e):
        grid[i][j] = -s


def demo() -> float:
  grid = [[1, -1], [-1, 1]]
  return ising_energy(grid)


def main() -> None:
  e = demo()
  assert abs(e - 4.0) < 1e-9, e
  print(f"energy={e}")


if __name__ == "__main__":
  main()
