"""BST kata."""
class Node:
    def __init__(self, v):
        self.v = v
        self.l = self.r = None

def insert(root, v):
    if root is None:
        return Node(v)
    if v < root.v:
        root.l = insert(root.l, v)
    else:
        root.r = insert(root.r, v)
    return root

def inorder(root):
    if root is None:
        return []
    return inorder(root.l) + [root.v] + inorder(root.r)

def demo():
    root = None
    root = insert(root, 2)
    root = insert(root, 1)
    root = insert(root, 4)
    root = insert(root, 3)
    return inorder(root)
