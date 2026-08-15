"""Bloom kata."""
import hashlib


class BloomFilter:
    def __init__(self, size: int):
        self.size = size
        self.filter = [False] * size

    def add(self, item: str):
        for i in range(3):
            hash_value = int(hashlib.sha256(item.encode()).hexdigest(), 16) % self.size
            self.filter[hash_value] = True

    def maybe(self, item: str) -> bool:
        for i in range(3):
            hash_value = int(hashlib.sha256(item.encode()).hexdigest(), 16) % self.size
            if not self.filter[hash_value]:
                return False
        return True

def demo():
    bf = BloomFilter(10)
    bf.add('x')
    print(bf.maybe('x'))  # Should print True
    print(bf.maybe('y'))  # Should print False
