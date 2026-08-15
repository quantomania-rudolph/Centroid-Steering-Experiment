"""Pytest for healed p19 union-find."""

from __future__ import annotations

import main


def test_demo_returns_root_after_union() -> None:
    assert main.demo() == 1


def test_path_compression() -> None:
    uf = main.UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(2) == uf.find(0)


def test_union_twice_links_components() -> None:
    uf = main.UnionFind(4)
    uf.union(0, 1)
    uf.union(2, 3)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(3)


def test_find_unconnected_node_is_self() -> None:
    uf = main.UnionFind(3)
    uf.union(0, 1)
    assert uf.find(2) == 2
