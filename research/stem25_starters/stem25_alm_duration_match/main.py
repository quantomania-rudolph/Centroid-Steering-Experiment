"""ALM duration immunization — broken Macaulay duration."""

from __future__ import annotations


def macaulay_duration(cashflows: list[float], times: list[float], y: float) -> float:
  # BUG: uses undiscounted cashflows in numerator
  num = sum(t * cf for t, cf in zip(times, cashflows))
  den = sum(cf / (1 + y) ** t for t, cf in zip(times, cashflows))
  return num / den


def immunize_weight(d_a: float, d_b: float, d_liability: float) -> float:
  # w on A: w*d_a + (1-w)*d_b = d_liability
  return (d_liability - d_b) / (d_a - d_b)


def demo() -> float:
  cfs_a = [100.0]
  t_a = [5.0]
  cfs_b = [50.0, 50.0]
  t_b = [4.0, 6.0]
  y = 0.05
  d_a = macaulay_duration(cfs_a, t_a, y)
  d_b = macaulay_duration(cfs_b, t_b, y)
  w = immunize_weight(d_a, d_b, 5.0)
  return w * d_a + (1 - w) * d_b


def main() -> None:
  port_d = demo()
  assert 4.8 < port_d < 5.2, port_d
  print(f"portfolio_duration={port_d:.4f}")


if __name__ == "__main__":
  main()
