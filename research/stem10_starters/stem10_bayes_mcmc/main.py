"""Metropolis-Hastings — broken acceptance ratio."""

from __future__ import annotations

import math
import random


def log_target(x: float) -> float:
  """Log density of N(0,1)."""
  return -0.5 * x * x


def metropolis_step(x: float, proposal_std: float, rng: random.Random) -> float:
  x_prop = x + rng.gauss(0, proposal_std)
  # BUG: uses linear ratio instead of log acceptance
  ratio = math.exp(log_target(x_prop) - log_target(x))
  if ratio > rng.random():
    return x_prop
  return x


def mcmc(n_samples: int, seed: int = 42) -> float:
  rng = random.Random(seed)
  x = 0.0
  for _ in range(n_samples):
    x = metropolis_step(x, 1.0, rng)
  return x


def demo() -> float:
  return mcmc(5000)


def main() -> None:
  x = demo()
  assert -0.5 < x < 0.5, x
  print(f"final_sample={x:.4f}")


if __name__ == "__main__":
  main()
