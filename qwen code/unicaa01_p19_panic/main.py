"""Broken union-find."""


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        raise NotImplementedError("find not implemented")

    def union(self, a, b):
        raise NotImplementedError("union not implemented")


def demo():
    uf = UnionFind(3)
    uf.union(0, 1)
    return uf.find(0)
