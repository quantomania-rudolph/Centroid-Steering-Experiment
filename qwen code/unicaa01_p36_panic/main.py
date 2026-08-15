"""Broken HMAC."""
import hashlib
import hmac


def hmac_hex(key: bytes, msg: bytes) -> str:
    return hmac.new(key, msg, hashlib.sha256).hexdigest()

def demo():
    return len(hmac_hex(b"k", b"m"))
