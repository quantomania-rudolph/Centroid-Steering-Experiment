"""XML escape kata."""

def xml_escape(s):
    """Escape XML special characters."""
    escape_dict = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&apos;'
    }
    return ''.join(escape_dict.get(c, c) for c in s)

def demo():
    """Demo the xml_escape function."""
    input_str = "This is a test & string."
    escaped_str = xml_escape(input_str)
    print(f"Input: {input_str}")
    print(f"Escaped: {escaped_str}")

if __name__ == '__main__':
    demo()
