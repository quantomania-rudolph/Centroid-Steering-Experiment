"""ROT13 kata."""
def rot13(s: str) -> str:
    def encode(c: str) -> str:
        if 'a' <= c <= 'z':
            return chr((ord(c) - ord('a') + 13) % 26 + ord('a'))
        elif 'A' <= c <= 'Z':
            return chr((ord(c) - ord('A') + 13) % 26 + ord('A'))
        else:
            return c

    def decode(c: str) -> str:
        return encode(c)

    return ''.join(decode(c) for c in s)

def demo():
    assert rot13(rot13('abc')) == 'abc', "Demo failed"

if __name__ == '__main__':
    demo()
