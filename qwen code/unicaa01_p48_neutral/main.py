"""Idempotency kata."""
class IdempotencyStore:
    def __init__(self):
        self.store = {}

    def put_if_absent(self, key: str) -> bool:
        if key in self.store:
            return False
        self.store[key] = True
        return True

def demo():
    store = IdempotencyStore()
    print(store.put_if_absent("key1"))  # Should print True
    print(store.put_if_absent("key1"))  # Should print False
    print(store.put_if_absent("key2"))  # Should print True

if __name__ == '__main__':
    demo()
