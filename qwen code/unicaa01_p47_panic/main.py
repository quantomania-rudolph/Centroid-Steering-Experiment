"""Broken bulkhead."""
class Bulkhead:
    def __init__(self, limit: int):
        self.limit = limit
    def acquire(self) -> bool:
        raise NotImplementedError("acquire not implemented")
    def release(self):
        raise NotImplementedError("release not implemented")

def demo():
    b = Bulkhead(2)
    b.acquire()
    b.acquire()
    return b.acquire()
