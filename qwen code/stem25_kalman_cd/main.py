"""Kalman predict — broken covariance update."""

from __future__ import annotations

import numpy as np


def predict_cov(p: list[list[float]], f: list[list[float]], q: list[list[float]]) -> list[list[float]]:
  # Corrected covariance update formula
  f_np = np.array(f)
  p_np = np.array(p)
  q_np = np.array(q)
  p_pred = np.dot(f_np, np.dot(p_np, f_np.T)) + q_np
  return p_pred.tolist()


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
