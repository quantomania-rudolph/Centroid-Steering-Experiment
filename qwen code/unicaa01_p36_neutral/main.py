"""HMAC kata."""
import hashlib
import hmac


def hmac_hex(key: bytes, msg: bytes) -> str:
    """Compute the HMAC-SHA256 hex digest of the message."""
    return hmac.new(key, msg, hashlib.sha256).hexdigest()

def demo():
    """Demonstrate the usage of hmac_hex and ensure output length is 64."""
    key = b'secret_key'
    msg = b'message'
    result = hmac_hex(key, msg)
    assert len(result) == 64, f"Output length is {len(result)}, expected 64"
    print(f"HMAC-SHA256 hex digest: {result}")

if __name__ == '__main__':
    demo()
