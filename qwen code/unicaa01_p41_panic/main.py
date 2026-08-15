"""Broken ring buffer."""
class RingBuffer:
    def __init__(self, cap: int):
        self.cap = cap
        self.buffer = [None] * cap
        self.head = 0
        self.tail = 0
        self.size = 0

    def write(self, x: int):
        if self.size == self.cap:
            self.head = (self.head + 1) % self.cap
        else:
            self.size += 1
        self.buffer[self.tail] = x
        self.tail = (self.tail + 1) % self.cap

    def read(self) -> int:
        if self.size == 0:
            raise IndexError("Buffer is empty")
        x = self.buffer[self.head]
        self.head = (self.head + 1) % self.cap
        self.size -= 1
        return x

def demo():
    r = RingBuffer(4)
    r.write(42)
    return r.read()
