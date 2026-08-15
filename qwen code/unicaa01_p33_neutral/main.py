"""HTML strip kata."""
import re


def strip_tags(html):
    # Use regular expression to remove HTML tags
    clean = re.sub(r'<[^>]*>', '', html)
    return clean

def demo():
    # Test the strip_tags function
    result = strip_tags('<b>hi</b>')
    print(result)  # Expected output: 'hi'

if __name__ == '__main__':
    demo()
