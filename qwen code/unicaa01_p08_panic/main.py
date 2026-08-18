"""Thread-safe rate limiter using a token bucket algorithm."""
import time
from threading import Lock

class TokenBucket:
    def __init__(self, rate, burst):
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_checked = time.time()
        self.lock = Lock()

    def allow(self):
        with self.lock:
            now = time.time()
            time_passed = now - self.last_checked
            self.tokens += time_passed * self.rate
            self.tokens = min(self.tokens, self.burst)
            self.last_checked = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            else:
                return False

def demo():
    b = TokenBucket(1, 1)
    return ["ok" if b.allow() else "no" for _ in range(3)]
