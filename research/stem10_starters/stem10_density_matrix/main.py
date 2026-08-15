"""Density matrix purity — broken trace and commutator."""

from __future__ import annotations

import cmath


def purity(rho: list[list[complex]]) -> float:
  """Tr(rho^2) for 2x2 density matrix."""
  # BUG: wrong matrix multiply
  tr = abs(rho[0][0]) ** 2 + abs(rho[1][1]) ** 2
  return float(tr.real if isinstance(tr, complex) else tr)


def commutator_norm(a: list[list[complex]], b: list[list[complex]]) -> float:
  """Frobenius norm of [A,B]."""
  raise NotImplementedError("implement commutator for Pauli matrices")


def demo() -> tuple[float, float]:
  rho = [[0.5, 0.5], [0.5, 0.5]]
  p = purity(rho)
  return p, 0.0


def main() -> None:
  p, _ = demo()
  assert abs(p - 1.0) < 1e-6, p
  print(f"purity={p}")


if __name__ == "__main__":
  main()
