"""Cantilever beam — broken deflection formula."""

from __future__ import annotations


def max_tip_deflection(p: float, l: float, e: float, i_moment: float) -> float:
  # BUG: wrong formula P*L/(E*I)
  return p * l / (e * i_moment)


def demo() -> float:
  return max_tip_deflection(p=7500.0, l=2.0, e=200e9, i_moment=8.33e-6)


def main() -> None:
  d = demo()
  assert abs(d - 0.012) < 1e-4, d
  print(f"delta={d:.6f}")


if __name__ == "__main__":
  main()
