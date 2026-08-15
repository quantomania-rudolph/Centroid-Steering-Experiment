"""Density matrix purity — broken trace and commutator."""

from __future__ import annotations

import numpy as np


def purity(rho: list[list[complex]]) -> float:
  """Tr(rho^2) for 2x2 density matrix."""
  rho_squared = np.dot(rho, rho)
  tr = np.trace(rho_squared)
  return float(tr.real)


def commutator_norm(a: list[list[complex]], b: list[list[complex]]) -> float:
  """Frobenius norm of [A,B]."""
  commutator = np.dot(a, b) - np.dot(b, a)
  return np.linalg.norm(commutator, 'fro')


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
