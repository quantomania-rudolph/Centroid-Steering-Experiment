"""Merkle kata."""
import hashlib


def merkle_root(leaves: list[bytes]) -> str:
    if not leaves:
        return ""

    while len(leaves) > 1:
        new_leaves = []
        for i in range(0, len(leaves), 2):
            if i + 1 < len(leaves):
                combined = leaves[i] + leaves[i + 1]
            else:
                combined = leaves[i]
            new_leaves.append(hashlib.sha256(combined).digest())
        leaves = new_leaves

    return hashlib.sha256(leaves[0]).hexdigest()

def demo():
    leaves = [
        b"hello",
        b"world",
        b"this",
        b"is",
        b"a",
        b"test"
    ]
    root = merkle_root(leaves)
    print(f"Merkle root: {root}")

if __name__ == '__main__':
    demo()
