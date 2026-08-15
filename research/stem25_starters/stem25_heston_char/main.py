"""Heston characteristic function — wrong sign at u=0."""

from __future__ import annotations

import cmath


def heston_log_char(u: complex, v0: float, kappa: float, theta: float, sigma: float, rho: float, t: float) -> complex:
  # BUG: adds kappa*theta instead of subtract in drift term (simplified)
  drift = kappa * theta * t
  return drift + 0j


def demo() -> complex:
  return heston_log_char(0 + 0j, v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, t=1.0)


def main() -> None:
  phi = demo()
  assert abs(phi.real) < 1e-9, phi
  print(f"phi(0)={phi}")


if __name__ == "__main__":
  main()
