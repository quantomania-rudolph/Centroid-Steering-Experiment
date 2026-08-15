"""1D infinite square well — broken normalization and eigenvalue."""

from __future__ import annotations

import math


def infinite_well_energy(n: int, width: float, mass: float = 1.0, hbar: float = 1.0) -> float:
  """Energy E_n for quantum number n >= 1."""
  if n < 1:
    raise ValueError("n must be >= 1")
  # Corrected energy formula
  return (n * n * hbar * hbar * math.pi * math.pi) / (2 * mass * width * width)


def wavefunction(x: float, n: int, width: float) -> float:
  """Normalized psi_n(x) on [0, L]."""
  if not 0 <= x <= width:
    return 0.0
  # Corrected normalization constant
  return math.sqrt(2 / width) * math.sin(n * math.pi * x / width)


def demo() -> tuple[float, float]:
  e1 = infinite_well_energy(1, width=1.0)
  psi = wavefunction(0.5, 1, 1.0)
  return e1, psi


def main() -> None:
  e1, psi = demo()
  assert abs(e1 - math.pi ** 2 / 2) < 1e-6, e1
  assert abs(psi - math.sqrt(2) * math.sin(math.pi / 2)) < 1e-6, psi
  print(f"E1={e1:.6f} psi(0.5)={psi:.6f}")


if __name__ == "__main__":
  main()
