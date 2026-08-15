"""Rate limiter kata."""
import threading
import time


class TokenBucket:
    def __init__(self, rate, burst):
        self.rate = rate  # tokens per second
        self.burst = burst  # maximum tokens
        self.tokens = burst
        self.last_refill_time = time.time()
        self.lock = threading.Lock()

    def refill(self):
        now = time.time()
        time_passed = now - self.last_refill_time
        tokens_to_add = time_passed * self.rate
        if tokens_to_add > 0:
            with self.lock:
                self.tokens = min(self.tokens + tokens_to_add, self.burst)
                self.last_refill_time = now

    def allow(self):
        self.refill()
        with self.lock:
            if self.tokens > 0:
                self.tokens -= 1
                return True
            return False

def demo():
    bucket = TokenBucket(rate=1, burst=5)
    for i in range(10):
        if bucket.allow():
            print(f"Request {i} allowed at {time.time()}")
        else:
            print(f"Request {i} denied at {time.time()}")
        time.sleep(0.5)

if __name__ == '__main__':
    demo()
