"""Black-Scholes Greeks — fixed d1/d2 and gamma."""

from __future__ import annotations

import math


def black_scholes_greeks(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    t_years: float,
    *,
    is_call: bool = True,
) -> dict[str, float]:
    """Return price, delta, gamma, vega, theta for European option."""
    if t_years <= 0 or vol <= 0:
        raise ValueError("t_years and vol must be positive")
    
    # Corrected d1 and d2 formulas
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t_years) / (vol * math.sqrt(t_years))
    d2 = d1 - vol * math.sqrt(t_years)
    
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2)))
    nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2)))
    discount = math.exp(-rate * t_years)
    
    if is_call:
        price = spot * nd1 - strike * discount * nd2
        delta = nd1
    else:
        price = strike * discount * (1 - nd2) - spot * (1 - nd1)
        delta = nd1 - 1
    
    # Implement gamma using the standard normal pdf
    gamma = math.exp(-0.5 * d1 * d1) / (vol * math.sqrt(t_years) * math.sqrt(2 * math.pi))
    
    vega = spot * math.sqrt(t_years) * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    theta = -spot * vol * math.exp(-0.5 * d1 * d1) / (2 * math.sqrt(t_years * math.pi)) - rate * strike * discount * nd2 + rate * spot * nd1
    
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def demo() -> dict[str, float]:
    return black_scholes_greeks(100.0, 100.0, 0.05, 0.2, 1.0, is_call=True)


def main() -> None:
    g = demo()
    assert 9.0 < g["price"] < 12.0, g
    assert 0.5 < g["delta"] < 0.7, g
    assert g["gamma"] > 0.01, g
    print(g)


if __name__ == "__main__":
    main()
