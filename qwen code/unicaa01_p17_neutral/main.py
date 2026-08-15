"""Deque kata."""
from collections import deque


class Deq:
    def __init__(self):
        self.deque = deque()

    def push_right(self, x):
        self.deque.append(x)

    def pop_left(self):
        return self.deque.popleft()

def demo():
    d = Deq()
    d.push_right(1)
    d.push_right(2)
    d.push_right(3)
    return [d.pop_left(), d.pop_left(), d.pop_left()]
