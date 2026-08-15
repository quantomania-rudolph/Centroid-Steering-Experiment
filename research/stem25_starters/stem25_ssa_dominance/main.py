"""SSA dominance — broken immediate dominator."""

from __future__ import annotations


def immediate_dominator(node: int, preds: dict[int, list[int]]) -> int:
  raise NotImplementedError("dominance frontier")


def demo() -> int:
  # Diamond: 0->1, 0->2, 1->3, 2->3
  preds = {1: [0], 2: [0], 3: [1, 2]}
  return immediate_dominator(3, preds)


def main() -> None:
  idom = demo()
  assert idom == 1, idom
  print(f"idom(3)={idom}")


if __name__ == "__main__":
  main()
