"""Cookie jar kata."""
class CookieJar:
    def __init__(self):
        self.cookies = {}

    def set(self, k, v):
        self.cookies[k] = v

    def get(self, k):
        return self.cookies.get(k, None)

def demo():
    jar = CookieJar()
    jar.set('k', 'v')
    return jar.get('k')
