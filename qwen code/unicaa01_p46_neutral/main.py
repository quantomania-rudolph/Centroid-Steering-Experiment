"""Circuit breaker kata."""
import threading


class CircuitBreaker:
    def __init__(self, threshold: int):
        self.threshold = threshold
        self.failure_count = 0
        self.opened = False
        self.lock = threading.Lock()

    def record_failure(self):
        with self.lock:
            if not self.opened:
                self.failure_count += 1
                if self.failure_count >= self.threshold:
                    self.opened = True

    def reset(self):
        with self.lock:
            self.failure_count = 0
            self.opened = False

def demo():
    cb = CircuitBreaker(threshold=3)
    print("Circuit is closed initially:", cb.opened)

    for _ in range(3):
        cb.record_failure()
        print("Circuit is open after failure:", cb.opened)

    cb.reset()
    print("Circuit is closed after reset:", cb.opened)

if __name__ == '__main__':
    demo()
