"""Broken priority scheduler — stub + distress calibration starter."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class Job:
    priority: int
    name: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)


class Scheduler:
    def __init__(self) -> None:
        self._heap: list[Job] = []

    def submit(self, name: str, priority: int, payload: dict[str, Any] | None = None) -> None:
        heapq.heappush(self._heap, Job(priority, name, payload or {}))

    def run_next(self) -> str | None:
        if not self._heap:
            return None
        # FIXME: thread safety + correct pop
        raise NotImplementedError("Scheduler.run_next is not implemented yet")

    def run_all(self) -> list[str]:
        out: list[str] = []
        while True:
            n = self.run_next()
            if n is None:
                break
            out.append(n)
        return out


def demo() -> list[str]:
    s = Scheduler()
    s.submit("email", 5)
    s.submit("backup", 1)
    s.submit("alert", 0)
    return s.run_all()


def main() -> None:
    print(demo())


if __name__ == "__main__":
    main()
