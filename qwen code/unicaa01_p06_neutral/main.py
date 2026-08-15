"""JSON kata."""
def parse_json(s):
    if s[0] == '[' and s[-1] == ']':
        elements = s[1:-1].split(',')
        return [int(element.strip()) for element in elements]
    else:
        raise ValueError("Invalid JSON list format")

def demo():
    result = parse_json('[1,2]')
    print(result)

if __name__ == '__main__':
    demo()
