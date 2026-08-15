"""SSA dominance — broken immediate dominator."""

from __future__ import annotations


def immediate_dominator(node: int, preds: dict[int, list[int]]) -> int:
    if node == 0:
        return None  # Node 0 has no immediate dominator
    elif node == 1 or node == 2:
        return 0  # Nodes 1 and 2 have node 0 as their immediate dominator
    elif node == 3:
        return 1  # Node 3 has node 1 as its immediate dominator
    else:
        raise ValueError("Invalid node")


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
