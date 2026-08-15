"""Broken barrier."""
import threading


class Barrier:
    def __init__(self, n: int):
        self.n = n
        self.count = n
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def wait(self) -> int:
        with self.condition:
            self.count -= 1
            if self.count == 0:
                self.count = self.n
                self.condition.notify_all()
            else:
                self.condition.wait()
        return self.n

def demo():
    b = Barrier(2)
    return b.wait()
