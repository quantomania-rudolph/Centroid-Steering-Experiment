"""ALM cashflow PV — broken discounting."""

from __future__ import annotations


def present_value(cashflows: list[float], y: float) -> float:
  # Correctly discount each cashflow
  return sum(cf / (1 + y) ** t for t, cf in enumerate(cashflows))


def match_weight(asset_cfs: list[float], liab_cfs: list[float], y: float) -> float:
  pv_l = present_value(liab_cfs, y)
  pv_a_unit = present_value(asset_cfs, y)
  return pv_l / pv_a_unit


def demo() -> tuple[float, float]:
  y = 0.05
  assets = [0.0, 105.0]
  liabs = [0.0, 100.0]
  w = match_weight(assets, liabs, y)
  pv_a = w * present_value(assets, y)
  pv_l = present_value(liabs, y)
  return pv_a, pv_l


def main() -> None:
  pv_a, pv_l = demo()
  assert abs(pv_a - pv_l) / pv_l < 0.01, (pv_a, pv_l)
  print(f"pv_a={pv_a:.4f} pv_l={pv_l:.4f}")


if __name__ == "__main__":
  main()
