"""Competitive inhibition — wrong Km_app."""

from __future__ import annotations


def mm_rate(s: float, vmax: float, km: float) -> float:
  return vmax * s / (km + s)


def competitive_km_app(km: float, i_conc: float, ki: float) -> float:
  # BUG: divides instead of multiplies
  return km / (1 + i_conc / ki)


def demo() -> float:
  km, ki, i_conc, vmax = 2.0, 1.0, 1.0, 10.0
  km_app = competitive_km_app(km, i_conc, ki)
  return mm_rate(km_app, vmax, km_app)


def main() -> None:
  v = demo()
  assert abs(v - 5.0) < 0.5, v
  print(f"v={v:.4f}")


if __name__ == "__main__":
  main()
