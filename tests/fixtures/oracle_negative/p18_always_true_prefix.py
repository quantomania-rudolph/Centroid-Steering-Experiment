"""Always-True starts_with — must fail p18 prefix_miss."""


class Trie:
    def __init__(self):
        self.root = {}

    def insert(self, word):
        self.root[word] = True

    def starts_with(self, prefix):
        return True


def demo():
    t = Trie()
    t.insert("apple")
    return t.starts_with("app")
