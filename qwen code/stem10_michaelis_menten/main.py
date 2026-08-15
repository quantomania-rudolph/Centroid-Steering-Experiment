"""Michaelis-Menten nonlinear fit — stubbed least squares."""

from __future__ import annotations

import numpy as np


def michaelis_menten(vmax: float, km: float, substrate: float) -> float:
  """Reaction velocity."""
  if substrate < 0 or km <= 0 or vmax <= 0:
    raise ValueError("invalid parameters")
  return vmax * substrate / (km + substrate)


def fit_mm(substrates: list[float], velocities: list[float]) -> tuple[float, float]:
  """Return (Vmax, Km) via linearized Lineweaver-Burk or direct LS."""
  if len(substrates) != len(velocities):
    raise ValueError("substrates and velocities must have the same length")
  
  x = 1 / np.array(substrates)
  y = 1 / np.array(velocities)
  
  # Perform linear fit y = m*x + b
  m, b = np.polyfit(x, y, 1)
  
  # Calculate Vmax and Km
  vmax = 1 / b
  km = m * vmax
  
  # Ensure Vmax and Km are positive
  if vmax <= 0 or km <= 0:
    raise ValueError("Vmax and Km must be positive")
  
  return vmax, km


def demo() -> tuple[float, float]:
  s = [0.5, 1.0, 2.0, 5.0, 10.0]
  v = [michaelis_menten(10.0, 2.0, x) for x in s]
  return fit_mm(s, v)


def main() -> None:
  vmax, km = demo()
  assert abs(vmax - 10.0) < 0.5 and abs(km - 2.0) < 0.5, (vmax, km)
  print(f"Vmax={vmax:.3f} Km={km:.3f}")


if __name__ == "__main__":
  main()
