"""Bulkhead kata."""
import threading


class Bulkhead:
    def __init__(self, limit: int):
        self.semaphore = threading.Semaphore(limit)

    def acquire(self) -> bool:
        return self.semaphore.acquire(blocking=False)

    def release(self):
        self.semaphore.release()

def demo():
    bulkhead = Bulkhead(2)

    # First acquire should succeed
    if bulkhead.acquire():
        print("First acquire succeeded")
        bulkhead.release()
    else:
        print("First acquire failed")

    # Second acquire should succeed
    if bulkhead.acquire():
        print("Second acquire succeeded")
        bulkhead.release()
    else:
        print("Second acquire failed")

    # Third acquire should fail
    if bulkhead.acquire():
        print("Third acquire succeeded")
        bulkhead.release()
    else:
        print("Third acquire failed")

if __name__ == "__main__":
    demo()
