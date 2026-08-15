"""Broken priority scheduler — panic starter (stress_b style)."""
from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class Job:
    priority: int
    name: str = field(compare=False)
    payload: dict[str, Any] = field(default_factory=dict)


class Scheduler:
    def __init__(self) -> None:
        self._heap: list[Job] = []
        self._lock = threading.Lock()

    def submit(self, job: Job) -> None:
        with self._lock:
            heapq.heappush(self._heap, job)

    def run_next(self) -> Job | None:
        with self._lock:
            if self._heap:
                return heapq.heappop(self._heap)
            return None


def demo() -> list[str]:
    s = Scheduler()
    for p, n in [(2, "backup"), (1, "alert"), (3, "email")]:
        s.submit(Job(p, n))
    out: list[str] = []
    for _ in range(3):
        j = s.run_next()
        if j is None:
            break
        out.append(j.name)
    return out


if __name__ == "__main__":
    result = demo()
    print(result)
