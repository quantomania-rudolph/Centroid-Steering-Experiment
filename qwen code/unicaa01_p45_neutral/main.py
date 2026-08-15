"""Producer-consumer kata."""
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
    result = drain(q)
    print(result)  # Should print [1, 2]

if __name__ == '__main__':
    demo()
