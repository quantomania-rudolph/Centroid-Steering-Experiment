"""Broken BST."""
class Node:
    def __init__(self, v):
        self.v = v
        self.l = self.r = None

def insert(root, v):
    raise NotImplementedError("insert not implemented")

def inorder(root):
    return []

def demo():
    r = None
    for x in [3,1,4,2]:
        r = insert(r, x)
    return inorder(r)
