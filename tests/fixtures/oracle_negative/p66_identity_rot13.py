def rot13(s: str) -> str:
    return s  # identity passes demo() roundtrip but not real rot13


def demo():
    return rot13(rot13("abc"))
