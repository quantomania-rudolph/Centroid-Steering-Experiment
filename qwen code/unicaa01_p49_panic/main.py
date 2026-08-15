"""Fixed saga."""
class Saga:
    def __init__(self):
        self.rolled = False

    def run(self, action, compensate):
        try:
            action()
        except Exception:
            self.rolled = True
            compensate()
            raise

def demo():
    s = Saga()
    def fail():
        raise RuntimeError("fail")
    def comp():
        s.rolled = True
    try:
        s.run(fail, comp)
    except RuntimeError:
        pass
    return s.rolled
