"""Broken hash chain."""
import hashlib


def hash_chain(seed: str, n: int) -> list[str]:
    hashes = []
    current_hash = seed.encode('utf-8')
    for _ in range(n):
        current_hash = hashlib.sha256(current_hash).hexdigest()
        hashes.append(current_hash)
    return hashes

def demo():
    return hash_chain("a", 3)
