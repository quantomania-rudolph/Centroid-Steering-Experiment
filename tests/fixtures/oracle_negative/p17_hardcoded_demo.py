class Deq:
    def __init__(self):
        self._items = []

    def push_right(self, x):
        self._items.append(x)

    def pop_left(self):
        # LIFO / stack mistake — fails true FIFO cases
        return self._items.pop()


def demo():
    q = Deq()
    for x in [1, 2, 3]:
        q.push_right(x)
    return [q.pop_left(), q.pop_left(), q.pop_left()]
