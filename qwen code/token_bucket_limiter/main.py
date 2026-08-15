import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class TokenBucket:
    def __init__(self, capacity: int, refill_rate_per_sec: float):
        self.capacity = capacity
        self.refill_rate_per_sec = refill_rate_per_sec
        self.tokens = capacity
        self.last_refill_time = time.monotonic()
        self.lock = threading.Lock()

    def refill(self) -> None:
        now = time.monotonic()
        elapsed_time = now - self.last_refill_time
        tokens_to_add = elapsed_time * self.refill_rate_per_sec
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill_time = now

    def allow(self, n_tokens: int = 1) -> bool:
        with self.lock:
            self.refill()
            if self.tokens >= n_tokens:
                self.tokens -= n_tokens
                return True
            return False

    def acquire(self, n_tokens: int = 1, timeout: float | None = None) -> bool:
        start_time = time.monotonic()
        while True:
            with self.lock:
                self.refill()
                if self.tokens >= n_tokens:
                    self.tokens -= n_tokens
                    return True
            if timeout is not None and time.monotonic() - start_time > timeout:
                return False
            time.sleep(0.01)

def main() -> None:
    bucket = TokenBucket(capacity=5, refill_rate_per_sec=1.0)

    def worker(token_count: int) -> None:
        result = bucket.acquire(token_count)
        print(f"Acquired {token_count} tokens: {'Success' if result else 'Denied'}")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, i % 3 + 1) for i in range(20)]
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    main()
