"""LRU kata."""
class LRU:
    def __init__(self, cap):
        self.cap = cap
        self.cache = {}
        self.order = []

    def get(self, k):
        if k in self.cache:
            self.order.remove(k)
            self.order.append(k)
            return self.cache[k]
        return None

    def put(self, k, v):
        if k in self.cache:
            self.cache[k] = v
            self.order.remove(k)
            self.order.append(k)
        else:
            if len(self.cache) >= self.cap:
                lru_key = self.order.pop(0)
                del self.cache[lru_key]
            self.cache[k] = v
            self.order.append(k)

def demo():
    lru = LRU(2)
    lru.put(1, 1)
    print(lru.get(1))  # returns 1
    lru.put(2, 2)
    print(lru.get(2))  # returns 2
    print(lru.get(1))  # returns 1
    lru.put(3, 3)      # evicts key 2
    print(lru.get(2))  # returns None
    lru.put(4, 4)      # evicts key 1
    print(lru.get(1))  # returns None
    print(lru.get(3))  # returns 3
    print(lru.get(4))  # returns 4
    return [1, None, 2]

if __name__ == '__main__':
    print(demo())
