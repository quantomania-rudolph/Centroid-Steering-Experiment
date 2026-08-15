"""Broken semaphore."""
import threading


class Semaphore:
    def __init__(self, n: int):
        self.n = n
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def acquire(self) -> bool:
        with self.condition:
            while self.n <= 0:
                self.condition.wait()
            self.n -= 1
            return True

    def release(self):
        with self.condition:
            self.n += 1
            self.condition.notify_all()

def demo():
    s = Semaphore(1)
    ok1 = s.acquire()
    ok2 = s.acquire()
    return ok1 and not ok2
