import threading
import time
from concurrent.futures import ThreadPoolExecutor


class TokenBucket:
    def __init__(self, rate: float, capacity: float) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def allow(self, n: float = 1.0) -> bool:
        with self._lock:
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

    # ThreadPoolExecutor demo
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(bucket.allow) for _ in range(10)]
        results = [future.result() for future in futures]
        print("All tokens allowed:", all(results))

if __name__ == "__main__":
    main()
