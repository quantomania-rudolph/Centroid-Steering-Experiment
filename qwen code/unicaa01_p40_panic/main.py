"""Broken skip list."""
class SkipList:
    def __init__(self):
        self._items = []

    def insert(self, x: int):
        self._items.append(x)

    def contains(self, x: int) -> bool:
        return x in self._items

def demo():
    s = SkipList()
    s.insert(5)
    return s.contains(5)
