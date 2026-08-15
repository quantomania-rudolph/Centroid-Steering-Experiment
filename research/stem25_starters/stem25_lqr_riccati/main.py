"""Discrete LQR Riccati — broken update."""

from __future__ import annotations


def dare_step_scalar(a: float, b: float, q: float, r: float, p_old: float) -> float:
  # BUG: adds instead of correct DARE form
  return q + a * a * p_old + b * b * p_old / r


def demo() -> float:
  return dare_step_scalar(a=1.1, b=0.5, q=1.0, r=0.1, p_old=1.0)


def main() -> None:
  p_new = demo()
  assert 1.2 < p_new < 1.5, p_new
  print(f"P_new={p_new:.4f}")


if __name__ == "__main__":
  main()
