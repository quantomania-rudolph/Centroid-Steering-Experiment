"""Skip list kata."""
import random


class SkipList:
    def __init__(self):
        self.head = Node(float('-inf'))
        self.tail = Node(float('inf'))
        self.head.right = self.tail
        self.tail.left = self.head

    def insert(self, x: int):
        node = Node(x)
        current = self.head
        while current.right.value < x:
            current = current.right
        node.left = current
        node.right = current.right
        current.right.left = node
        current.right = node
        self._flip_coin(node)

    def contains(self, x: int) -> bool:
        current = self.head
        while current.right.value < x:
            current = current.right
        return current.right.value == x

    def _flip_coin(self, node):
        level = 0
        while random.choice([True, False]):
            level += 1
        if level > self._max_level:
            self._max_level = level
            self._add_level()

    def _add_level(self):
        new_head = Node(float('-inf'))
        new_tail = Node(float('inf'))
        new_head.right = new_tail
        new_tail.left = new_head
        new_head.down = self.head
        new_tail.down = self.tail
        self.head.up = new_head
        self.tail.up = new_tail
        self.head = new_head
        self.tail = new_tail

class Node:
    def __init__(self, value, up=None, down=None, left=None, right=None):
        self.value = value
        self.up = up
        self.down = down
        self.left = left
        self.right = right

def demo():
    skip_list = SkipList()
    skip_list.insert(5)
    print(skip_list.contains(5))  # Should print True

if __name__ == '__main__':
    demo()
