"""Good trie prefix matcher."""


class Trie:
    def __init__(self):
        self.root = {}

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.setdefault(ch, {})

    def starts_with(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node:
                return False
            node = node[ch]
        return True


def demo():
    t = Trie()
    t.insert("apple")
    return t.starts_with("app")
