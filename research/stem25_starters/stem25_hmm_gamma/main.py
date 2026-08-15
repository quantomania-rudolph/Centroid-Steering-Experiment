"""HMM gamma — broken normalization."""

from __future__ import annotations


def gamma_t(alpha: list[float], beta: list[float], scale: float) -> list[float]:
  raw = [a * b for a, b in zip(alpha, beta)]
  # BUG: forgets to divide by scale**2 total likelihood
  s = sum(raw) or 1.0
  return [r / s for r in raw]


def demo() -> list[float]:
  alpha = [0.4, 0.6]
  beta = [0.5, 0.5]
  return gamma_t(alpha, beta, scale=0.9)


def main() -> None:
  g = demo()
  assert abs(sum(g) - 1.0) < 1e-9, g
  print(f"gamma={g}")


if __name__ == "__main__":
  main()
