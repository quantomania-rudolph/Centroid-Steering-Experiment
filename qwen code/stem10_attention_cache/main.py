"""Scaled dot-product attention with broken KV cache."""

from __future__ import annotations

import math


def softmax(xs: list[float]) -> list[float]:
  m = max(xs)
  ex = [math.exp(x - m) for x in xs]
  s = sum(ex)
  return [e / s for e in ex]


def attention_step(
    query: list[float],
    keys: list[list[float]],
    values: list[list[float]],
    *,
    scale: float | None = None,
) -> list[float]:
  """Single-query attention over cached K/V."""
  if scale is None:
    scale = math.sqrt(len(query))
  scores = [sum(q * k for q, k in zip(query, key)) / scale for key in keys]
  weights = softmax(scores)
  out = [sum(w * v[d] for w, v in zip(weights, values)) for d in range(len(values[0]))]
  return out


def demo() -> list[float]:
  q = [1.0, 0.0]
  keys = [[1.0, 0.0]]
  vals = [[1.0, 2.0]]
  return attention_step(q, keys, vals)


def main() -> None:
  out = demo()
  assert abs(out[0] - 1.0) < 1e-5 and abs(out[1] - 2.0) < 1e-5, out
  print(out)


if __name__ == "__main__":
  main()
