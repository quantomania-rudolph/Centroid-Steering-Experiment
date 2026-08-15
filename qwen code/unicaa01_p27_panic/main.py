"""Broken base64."""
import base64


def b64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode('utf-8')

def demo():
    return b64_encode(b"hi")
