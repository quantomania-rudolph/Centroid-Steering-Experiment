"""Min-variance portfolio — correct weight formula."""

from __future__ import annotations

import numpy as np


def min_var_weights(sigma: list[list[float]]) -> list[float]:
    sigma_np = np.array(sigma)
    ones = np.ones((2, 1))
    
    # Calculate the inverse of sigma
    inv_sigma = np.linalg.inv(sigma_np)
    
    # Calculate the numerator and denominator
    numerator = inv_sigma @ ones
    denominator = ones.T @ inv_sigma @ ones
    
    # Calculate the weights
    weights = numerator / denominator
    
    return weights.flatten().tolist()


def demo() -> list[float]:
    sigma = [[0.04, 0.01], [0.01, 0.09]]
    return min_var_weights(sigma)


def main() -> None:
    w = demo()
    assert abs(sum(w) - 1.0) < 1e-9, w
    assert 0.2 < w[0] < 0.8, w
    print(f"w={w}")


if __name__ == "__main__":
    main()
