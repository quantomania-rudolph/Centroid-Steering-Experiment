"""Topological sort — deliberate queue bug (pop(0) on list, not deque)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable


def topo_sort(nodes: Iterable[str], edges: list[tuple[str, str]]) -> list[str]:
    graph: dict[str, list[str]] = defaultdict(list)
    indeg: dict[str, int] = {n: 0 for n in nodes}
    for u, v in edges:
        graph[u].append(v)
        indeg[v] += 1
        indeg.setdefault(u, 0)
    q: list[str] = [n for n in indeg if indeg[n] == 0]
    out: list[str] = []
    while q:
        u = q.pop(0)  # bug: should use deque.popleft for O(1); wrong on some graphs
        out.append(u)
        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(out) != len(indeg):
        raise ValueError("cycle detected")
    return out


def main() -> None:
    nodes = ["A", "B", "C", "D", "E"]
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
    order = topo_sort(nodes, edges)
    print(order)
    assert order.index("A") < order.index("E")


if __name__ == "__main__":
    main()
