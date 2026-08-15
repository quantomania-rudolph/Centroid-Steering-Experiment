"""Hash chain kata."""
import hashlib


def hash_chain(seed: str, n: int) -> list[str]:
    hashes = []
    current_hash = seed
    for _ in range(n):
        current_hash = hashlib.sha256(current_hash.encode()).hexdigest()
        hashes.append(current_hash)
    return hashes

def demo():
    chain = hash_chain('a', 3)
    print(chain)

if __name__ == '__main__':
    demo()
