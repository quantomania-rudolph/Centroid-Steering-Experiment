"""Trie kata."""
class Trie:
    def __init__(self):
        self.root = {}

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['end_of_word'] = True

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True

def demo():
    trie = Trie()
    trie.insert('apple')
    return 'app' in trie

if __name__ == '__main__':
    print(demo())
