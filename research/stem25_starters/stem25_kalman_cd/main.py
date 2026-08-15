"""Kalman predict — broken covariance update."""

from __future__ import annotations


def predict_cov(p: list[list[float]], f: list[list[float]], q: list[list[float]]) -> list[list[float]]:
  # BUG: P_pred = F*P + Q
  return [
    [f[0][0] * p[0][0] + q[0][0], f[0][1] * p[0][1] + q[0][1]],
    [f[1][0] * p[1][0] + q[1][0], f[1][1] * p[1][1] + q[1][1]],
  ]


def demo() -> list[list[float]]:
  p = [[1.0, 0.0], [0.0, 1.0]]
  f = [[1.0, 0.1], [0.0, 1.0]]
  q = [[0.01, 0.0], [0.0, 0.01]]
  return predict_cov(p, f, q)


def main() -> None:
  p_new = demo()
  assert p_new[0][0] > 1.0, p_new
  print(f"P00={p_new[0][0]:.4f}")


if __name__ == "__main__":
  main()
