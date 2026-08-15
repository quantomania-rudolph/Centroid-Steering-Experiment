"""p01 scheduler without Lock — fails SEM_invariants."""

import heapq
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class Job:
    priority: int
    name: str = field(compare=False)


class Scheduler:
    def __init__(self) -> None:
        self._heap: list[Job] = []

    def submit(self, job: Job) -> None:
        heapq.heappush(self._heap, job)

    def run_next(self) -> Job | None:
        return heapq.heappop(self._heap) if self._heap else None


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
