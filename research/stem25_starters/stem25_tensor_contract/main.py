"""Tensor chain contraction — wrong order cost."""

from __future__ import annotations


def chain_cost_left(shapes: list[tuple[int, int]]) -> int:
  cost = 0
  acc = shapes[0]
  for b in shapes[1:]:
    cost += acc[0] * acc[1] * b[1]
    acc = (acc[0], b[1])
  return cost


def optimal_chain_cost(shapes: list[tuple[int, int]]) -> int:
  # BUG: returns naive left-to-right only (7500 not optimal 4000)
  return chain_cost_left(shapes)


def demo() -> int:
  shapes = [(10, 20), (20, 30), (30, 5)]
  return optimal_chain_cost(shapes)


def main() -> None:
  cost = demo()
  assert cost == 4000, cost
  print(f"cost={cost}")


if __name__ == "__main__":
  main()
