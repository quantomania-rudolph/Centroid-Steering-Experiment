import hashlib


def merkle_root(leaves: list[bytes]) -> str:
    if len(leaves) == 0:
        return ""
    if len(leaves) == 1:
        return hashlib.sha256(leaves[0]).hexdigest()
    
    new_leaves = []
    for i in range(0, len(leaves), 2):
        if i + 1 < len(leaves):
            combined = leaves[i] + leaves[i + 1]
        else:
            combined = leaves[i]
        new_leaves.append(hashlib.sha256(combined).digest())
    
    return merkle_root(new_leaves)

def demo():
    leaves = [hashlib.sha256(b"x").digest(), hashlib.sha256(b"y").digest()]
    return merkle_root(leaves)
