"""Query parse kata."""
from urllib.parse import parse_qs


def parse_query(qs):
    params = parse_qs(qs)
    return params.get('a', [None])[0]

def demo():
    query_string = 'a=1&b=2'
    result = parse_query(query_string)
    print(f"The value of 'a' is: {result}")

if __name__ == '__main__':
    demo()
