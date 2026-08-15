"""Two-compartment PK steady state — broken clearance."""

from __future__ import annotations


def steady_state_c1(r0: float, k10: float, k12: float, k21: float) -> float:
  # Corrected CL calculation to include peripheral clearance path
  cl = k10 + k12 * k21 / (k12 + k21)
  return r0 / cl


def demo() -> float:
  return steady_state_c1(r0=10.0, k10=0.9, k12=0.1, k21=0.1)


def main() -> None:
  css = demo()
  assert 10.4 < css < 10.7, css
  print(f"Css={css:.4f}")


if __name__ == "__main__":
  main()
