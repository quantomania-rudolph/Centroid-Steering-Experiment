"""1D bar stiffness — broken matrix."""

from __future__ import annotations


def bar_stiffness(ea: float, length: float) -> list[list[float]]:
  k = ea / length
  return [[k, -k], [-k, k]]


def demo() -> list[list[float]]:
  return bar_stiffness(ea=1e6, length=2.0)


def main() -> None:
  k = demo()
  assert abs(k[0][0] - 5e5) < 1.0, k
  assert abs(k[1][1] - 5e5) < 1.0, k
  assert abs(k[0][1] + 5e5) < 1.0, k
  assert abs(k[1][0] + 5e5) < 1.0, k
  print(f"K00={k[0][0]}")


if __name__ == "__main__":
  main()
