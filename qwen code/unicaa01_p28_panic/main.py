def glob_match(pattern, text):
    if pattern == text:
        return True
    if not pattern or not text:
        return False
    if pattern[0] == '*':
        return glob_match(pattern[1:], text[1:]) or glob_match(pattern, text[1:])
    if pattern[0] == '?':
        return glob_match(pattern[1:], text[1:])
    return pattern[0] == text[0] and glob_match(pattern[1:], text[1:])

def demo():
    return glob_match("foo*.txt", "foobar.txt")
