"""Broken cookie jar."""
class CookieJar:
    def __init__(self):
        self._c = {}
    
    def set(self, k, v):
        self._c[k] = v
    
    def get(self, k):
        return self._c.get(k, None)

def demo():
    j = CookieJar()
    j.set("k", "v")
    return j.get("k")
