"""Broken rate limiter."""
class TokenBucket:
    def __init__(self, rate, burst):
        self.tokens = burst
    def allow(self):
        raise NotImplementedError("allow not implemented")

def demo():
    b = TokenBucket(1, 1)
    return ["ok" if b.allow() else "no" for _ in range(3)]
