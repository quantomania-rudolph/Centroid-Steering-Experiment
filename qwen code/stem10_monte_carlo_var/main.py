"""Historical portfolio VaR — broken covariance and quantile."""

from __future__ import annotations

import math


def portfolio_var(
    weights: list[float],
    returns: list[list[float]],
    alpha: float = 0.05,
) -> float:
    """Parametric VaR at level alpha (positive loss number).

    returns[t][i] = return of asset i at time t.
    """
    n = len(returns)
    port = [sum(w * r for w, r in zip(weights, returns[t])) for t in range(n)]
    mu = sum(port) / n
    var = sum((x - mu) ** 2 for x in port) / (n - 1)
    sigma = math.sqrt(var)
    z = 1.645  # z-score for alpha=0.05
    return -(mu - z * sigma)


def demo() -> float:
    weights = [0.6, 0.4]
    returns = [
        [0.01, -0.02],
        [-0.005, 0.03],
        [0.02, -0.01],
        [-0.015, 0.02],
        [0.005, 0.0],
    ]
    return portfolio_var(weights, returns, alpha=0.05)


def main() -> None:
    var = demo()
    assert 0.004 < var < 0.08, var
    print(f"VaR={var:.4f}")


if __name__ == "__main__":
    main()
