"""2-qubit Heisenberg — broken Hamiltonian / ground energy."""

from __future__ import annotations

import cmath


def pauli_matrices() -> dict[str, list[list[complex]]]:
  sx = [[0, 1], [1, 0]]
  sy = [[0, -1j], [1j, 0]]
  sz = [[1, 0], [0, -1]]
  i2 = [[1, 0], [0, 1]]
  return {"I": i2, "X": sx, "Y": sy, "Z": sz}


def kron(a: list[list[complex]], b: list[list[complex]]) -> list[list[complex]]:
  out = []
  for row_a in a:
    for row_b in b:
      out.append([x * y for x in row_a for y in row_b])
  return out


def mat_add(a, b):
  return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def ground_energy(j: float) -> float:
  p = pauli_matrices()
  h = mat_add(kron(p["X"], p["X"]), kron(p["Y"], p["Y"]))
  h = mat_add(h, kron(p["Z"], p["Z"]))
  h = [[j * x for x in row] for row in h]
  # BUG: returns trace instead of min eigenvalue
  return sum(h[i][i].real for i in range(4))


def demo() -> float:
  return ground_energy(1.0)


def main() -> None:
  e0 = demo()
  assert abs(e0 + 3.0) < 1e-9, e0
  print(f"E0={e0}")


if __name__ == "__main__":
  main()
