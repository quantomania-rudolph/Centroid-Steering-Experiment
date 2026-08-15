"""Broken ROT13."""
def rot13(s: str) -> str:
    result = []
    for char in s:
        if 'a' <= char <= 'z':
            result.append(chr((ord(char) - ord('a') + 13) % 26 + ord('a')))
        elif 'A' <= char <= 'Z':
            result.append(chr((ord(char) - ord('A') + 13) % 26 + ord('A')))
        else:
            result.append(char)
    return ''.join(result)

def demo():
    return rot13(rot13("abc")) == "abc"
