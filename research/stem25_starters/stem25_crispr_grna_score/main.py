"""CRISPR gRNA scoring — broken mismatch weights."""

from __future__ import annotations


def grna_score(seq: str, target: str) -> float:
  # BUG: all mismatches weighted equally (no PAM-proximal 3x)
  score = 100.0
  for i, (a, b) in enumerate(zip(seq, target)):
    if a != b:
      score -= 5.0
  gc = (seq.count("G") + seq.count("C")) / len(seq)
  if 0.4 <= gc <= 0.6:
    score += 10.0
  return score


def demo() -> tuple[float, float]:
  good = grna_score("ACGTACGTACGT", "ACGTACGTACGT")
  bad = grna_score("TTTTTTTTTTTT", "ACGTACGTACGT")
  return good, bad


def main() -> None:
  good, bad = demo()
  assert good > bad, (good, bad)
  print(f"good={good} bad={bad}")


if __name__ == "__main__":
  main()
