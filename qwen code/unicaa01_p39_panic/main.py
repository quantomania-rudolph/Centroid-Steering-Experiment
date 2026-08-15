"""Simple bloom filter."""
class BloomFilter:
    def __init__(self, size: int):
        self.size = size
        self.bits = [False] * size

    def add(self, item: str):
        hash1 = hash(item) % self.size
        hash2 = (hash(item) // self.size) % self.size
        self.bits[hash1] = True
        self.bits[hash2] = True

    def maybe(self, item: str) -> bool:
        hash1 = hash(item) % self.size
        hash2 = (hash(item) // self.size) % self.size
        return self.bits[hash1] and self.bits[hash2]

def demo():
    b = BloomFilter(64)
    b.add("x")
    return b.maybe("x")
