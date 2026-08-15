"""Fixed LRU."""
from collections import OrderedDict


class LRU:
    def __init__(self, cap):
        self.cap = cap
        self.data = OrderedDict()

    def get(self, k):
        if k not in self.data:
            return -1
        self.data.move_to_end(k)
        return self.data[k]

    def put(self, k, v):
        if k in self.data:
            self.data.move_to_end(k)
        self.data[k] = v
        if len(self.data) > self.cap:
            self.data.popitem(last=False)

def demo():
    c = LRU(2)
    c.put(1, 1); c.put(2, 2); c.get(1); c.put(3, 3)
    return [c.get(1), c.get(2)]
