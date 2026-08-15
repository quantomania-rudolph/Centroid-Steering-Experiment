"""Base64 kata."""
import base64


def b64_encode(data: bytes) -> str:
    """Encode bytes to a valid Base64 string."""
    return base64.b64encode(data).decode('utf-8')

def demo():
    """Encode b'hi' to a valid Base64 string and print the result."""
    encoded_data = b64_encode(b'hi')
    print(encoded_data)

if __name__ == '__main__':
    demo()
