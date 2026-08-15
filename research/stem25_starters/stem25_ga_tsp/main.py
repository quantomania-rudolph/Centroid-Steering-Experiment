"""GA TSP — broken crossover duplicates cities."""

from __future__ import annotations

import random


def tour_cost(tour: list[int], dist: list[list[float]]) -> float:
  return sum(dist[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def ox_crossover(p1: list[int], p2: list[int], rng: random.Random) -> list[int]:
  # BUG: returns slice of p1 only (duplicates/missing)
  cut = len(p1) // 2
  return p1[:cut] + p1[:cut]


def demo() -> tuple[float, float]:
  dist = [
    [0, 1, 2, 3],
    [1, 0, 1, 2],
    [2, 1, 0, 1],
    [3, 2, 1, 0],
  ]
  rng = random.Random(0)
  t0 = [0, 2, 1, 3]
  c0 = tour_cost(t0, dist)
  child = ox_crossover([0, 2, 1, 3], [2, 0, 3, 1], rng)
  if len(set(child)) != 4:
    return c0, c0 + 1
  c1 = tour_cost(child, dist)
  return c0, c1


def main() -> None:
  c0, c1 = demo()
  assert c1 < c0, (c0, c1)
  print(f"cost {c0} -> {c1}")


if __name__ == "__main__":
  main()
