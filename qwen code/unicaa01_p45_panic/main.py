"""Broken producer-consumer."""
import queue


def drain(q: queue.Queue) -> list[int]:
    result = []
    while not q.empty():
        result.append(q.get())
    return result

def demo():
    q = queue.Queue()
    q.put(1)
    q.put(2)
    return drain(q)
