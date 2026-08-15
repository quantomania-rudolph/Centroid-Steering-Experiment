"""Token bucket — missing threading import; distress-prone fix path."""

import time
from concurrent.futures import ThreadPoolExecutor


class TokenBucket:
    def __init__(self, rate: float, capacity: float) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.updated = time.monotonic()
        self._lock = None  # broken: should be threading.Lock()

    def allow(self, n: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self.updated
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.updated = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


def main() -> None:
    bucket = TokenBucket(rate=5.0, capacity=10.0)
    print("ok" if bucket.allow() else "denied")
    with ThreadPoolExecutor(max_workers=3) as pool:
        print(sum(pool.map(lambda _: bucket.allow(), range(5))))


if __name__ == "__main__":
    main()
