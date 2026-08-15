"""Kalman filter — broken Joseph-form covariance update."""

from __future__ import annotations


def kalman_update(
    x: list[float],
    p: list[list[float]],
    z: float,
    h: list[float],
    r: float,
) -> tuple[list[float], list[list[float]]]:
  """1D measurement update for state dim 2."""
  # Innovation
  y = z - sum(h[i] * x[i] for i in range(len(x)))
  s = r
  for i in range(len(x)):
    for j in range(len(x)):
      s += h[i] * p[i][j] * h[j]
  k = [sum(p[i][j] * h[j] for j in range(len(x))) / s for i in range(len(x))]
  x_new = [x[i] + k[i] * y for i in range(len(x))]
  # Corrected Joseph-form covariance update
  p_new = [[p[i][j] - k[i] * h[j] * s for j in range(len(x))] for i in range(len(x))]
  return x_new, p_new


def demo() -> list[float]:
  x = [0.0, 0.0]
  p = [[1.0, 0.0], [0.0, 1.0]]
  x_new, _ = kalman_update(x, p, z=1.0, h=[1.0, 0.0], r=0.1)
  return x_new


def main() -> None:
  x = demo()
  assert 0.5 < x[0] < 0.95, x
  print(f"x={x}")


if __name__ == "__main__":
  main()
