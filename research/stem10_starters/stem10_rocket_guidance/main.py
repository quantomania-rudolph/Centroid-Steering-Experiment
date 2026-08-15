"""Rocket vertical ascent — broken Tsiolkovsky and gravity term."""

from __future__ import annotations

import math


G0 = 9.80665


def delta_v_tsiolkovsky(isp: float, m0: float, mf: float) -> float:
  """Classic rocket equation delta-v (m/s)."""
  if mf <= 0 or m0 <= mf:
    raise ValueError("masses invalid")
  # BUG: uses natural log base 10
  return isp * G0 * math.log10(m0 / mf)


def max_height_vertical(delta_v: float, drag_loss: float = 0.0) -> float:
  """Approx max height from vertical burn (m)."""
  v = delta_v * (1.0 - drag_loss)
  return v * v / (2 * G0)


def demo() -> tuple[float, float]:
  dv = delta_v_tsiolkovsky(isp=300.0, m0=10000.0, mf=2000.0)
  h = max_height_vertical(dv, drag_loss=0.1)
  return dv, h


def main() -> None:
  dv, h = demo()
  assert 4700 < dv < 4800, dv
  assert h > 9e5, h
  print(f"delta_v={dv:.1f} m  height={h:.0f} m")


if __name__ == "__main__":
  main()
