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
  n = len(shapes)
  dp = [[0] * n for _ in range(n)]

  for length in range(2, n + 1):
    for i in range(n - length + 1):
      j = i + length - 1
      dp[i][j] = float('inf')
      for k in range(i, j):
        cost = dp[i][k] + dp[k + 1][j] + shapes[i][0] * shapes[k][1] * shapes[j][1]
        dp[i][j] = min(dp[i][j], cost)

  return dp[0][n - 1]


def demo() -> int:
  shapes = [(10, 20), (20, 30), (30, 5)]
  return optimal_chain_cost(shapes)


def main() -> None:
  cost = demo()
  assert cost == 4000, cost
  print(f"cost={cost}")


if __name__ == "__main__":
  main()
