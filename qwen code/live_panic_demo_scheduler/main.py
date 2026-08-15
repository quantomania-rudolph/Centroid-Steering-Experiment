"""Thread-safe priority job scheduler — stress calibration starter."""

from __future__ import annotations

import heapq
import threading
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
        self._lock = threading.Lock()

    def submit(self, name: str, priority: int, payload: dict[str, Any] | None = None) -> None:
        with self._lock:
            heapq.heappush(self._heap, Job(priority, name, payload or {}))

    def run_next(self) -> str | None:
        with self._lock:
            if not self._heap:
                return None
            job = heapq.heappop(self._heap)
        return job.name


def demo() -> list[str]:
    s = Scheduler()
    s.submit("email", 5, {"to": "ops"})
    s.submit("backup", 1, {"path": "/data"})
    s.submit("alert", 0, {"msg": "disk full"})
    out: list[str] = []
    while True:
        n = s.run_next()
        if n is None:
            break
        out.append(n)
    return out


def main() -> None:
    result = demo()
    print(result)


if __name__ == "__main__":
    main()
