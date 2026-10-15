"""Min stack — pop/get_min stubs (NotImplementedError)."""

from __future__ import annotations


class MinStack:
    def __init__(self) -> None:
        self._values: list[int] = []
        self._mins: list[int] = []

    def push(self, value: int) -> None:
        self._values.append(value)
        if not self._mins or value <= self._mins[-1]:
            self._mins.append(value)

    def pop(self) -> int:
        raise NotImplementedError("pop not implemented")

    def get_min(self) -> int:
        raise NotImplementedError("get_min not implemented")


def demo() -> int:
    stack = MinStack()
    stack.push(3)
    stack.push(1)
    stack.push(2)
    return stack.get_min()
