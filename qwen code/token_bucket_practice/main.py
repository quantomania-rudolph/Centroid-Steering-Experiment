"""TokenBucket practice starter — CAA Method B pair task (one-shot, not Aider)."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor


class TokenBucket:
  """Thread-safe token bucket rate limiter (starter — allow() not implemented)."""

  def __init__(self, rate: float, capacity: float):
    self.rate = float(rate)
    self.capacity = float(capacity)
    self.tokens = float(capacity)
    self._lock = threading.Lock()
    self._last = time.monotonic()

  def allow(self, cost: float = 1.0) -> bool:
    raise NotImplementedError("implement with self._lock")


def demo() -> list[str]:
  bucket = TokenBucket(rate=10.0, capacity=2.0)
  out: list[str] = []

  def worker() -> None:
    out.append("ok" if bucket.allow() else "no")

  with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(lambda _: worker(), range(3)))
  return out


if __name__ == "__main__":
  print(demo())
