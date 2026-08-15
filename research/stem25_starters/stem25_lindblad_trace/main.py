"""Lindblad amplitude damping — broken trace preservation."""

from __future__ import annotations


def amplitude_damping_step(rho: list[list[complex]], gamma: float, dt: float) -> list[list[complex]]:
  # BUG: does not preserve trace
  r00 = rho[0][0] * (1 - gamma * dt)
  r11 = rho[1][1] + gamma * dt
  return [[r00, 0], [0, r11]]


def trace(m: list[list[complex]]) -> float:
  return float(m[0][0].real + m[1][1].real)


def demo() -> list[list[complex]]:
  rho = [[0.8 + 0j, 0], [0, 0.2 + 0j]]
  return amplitude_damping_step(rho, gamma=0.1, dt=0.5)


def main() -> None:
  rho_new = demo()
  assert abs(trace(rho_new) - 1.0) < 1e-9, trace(rho_new)
  assert rho_new[0][0].real > 0
  print(f"trace={trace(rho_new):.6f}")


if __name__ == "__main__":
  main()
