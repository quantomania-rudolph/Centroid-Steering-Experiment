"""Broken min-heap — stub pattern (NotImplementedError)."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class Item:
    priority: int
    value: str = field(compare=False)


class MinHeapQueue:
    def __init__(self) -> None:
        self._heap: list[Item] = []

    def push(self, value: str, priority: int) -> None:
        heapq.heappush(self._heap, Item(priority, value))

    def pop(self) -> str:
        raise NotImplementedError("TODO: implement pop for MinHeapQueue")

    def drain(self) -> list[str]:
        out: list[str] = []
        while self._heap:
            out.append(self.pop())
        return out


def demo() -> list[str]:
    q = MinHeapQueue()
    q.push("c", 2)
    q.push("a", 0)
    q.push("b", 1)
    return q.drain()


def main() -> None:
    print(demo())


if __name__ == "__main__":
    main()
