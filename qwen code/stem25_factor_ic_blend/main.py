"""IC-weighted factor alpha — broken normalization."""

from __future__ import annotations


def zscore(xs: list[float]) -> list[float]:
  mu = sum(xs) / len(xs)
  var = sum((x - mu) ** 2 for x in xs) / len(xs)
  sd = var ** 0.5 or 1.0
  return [(x - mu) / sd for x in xs]


def blend_score(factors: list[list[float]], ics: list[float], idx: int) -> float:
  # BUG: divides by sum(IC) not sum(|IC|)
  zs = [zscore(f) for f in factors]
  num = sum(ics[k] * zs[k][idx] for k in range(len(ics)))
  den = sum(abs(ic) for ic in ics)
  return num / den


def demo() -> float:
  f1 = [1.0, 2.0, 3.0]
  f2 = [3.0, 2.0, 1.0]
  ics = [0.5, -0.3]
  return blend_score([f1, f2], ics, idx=2)


def main() -> None:
  s = demo()
  assert 1.0 < s < 1.4, s
  print(f"score={s:.4f}")


if __name__ == "__main__":
  main()
