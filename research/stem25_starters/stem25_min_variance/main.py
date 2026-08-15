"""Min-variance portfolio — broken weight formula."""

from __future__ import annotations


def min_var_weights(sigma: list[list[float]]) -> list[float]:
  # 2x2 inverse analytic
  a, b = sigma[0][0], sigma[0][1]
  c, d = sigma[1][0], sigma[1][1]
  det = a * d - b * c
  inv = [[d / det, -b / det], [-c / det, a / det]]
  # BUG: uses inv @ mu instead of inv @ ones
  num = [inv[0][0] + inv[0][1], inv[1][0] + inv[1][1]]
  den = num[0] + num[1]
  return [num[0] / den, num[1] / den]


def demo() -> list[float]:
  sigma = [[0.04, 0.01], [0.01, 0.09]]
  return min_var_weights(sigma)


def main() -> None:
  w = demo()
  assert abs(sum(w) - 1.0) < 1e-9, w
  assert 0.2 < w[0] < 0.8, w
  print(f"w={w}")


if __name__ == "__main__":
  main()
