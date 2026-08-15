import fnmatch


def glob_match(pattern, text):
    return fnmatch.fnmatch(text, pattern)

def demo():
    print(glob_match('foo*.txt', 'foobar.txt'))  # Should print True

if __name__ == '__main__':
    demo()
