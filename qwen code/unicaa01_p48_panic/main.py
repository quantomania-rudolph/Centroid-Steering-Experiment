"""Broken idempotency."""
class IdempotencyStore:
    def __init__(self):
        self._seen = set()
    
    def put_if_absent(self, key: str) -> bool:
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

def demo():
    s = IdempotencyStore()
    first = s.put_if_absent("k")
    second = s.put_if_absent("k")
    return first and not second
