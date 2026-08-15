class Turnstile:
    def __init__(self):
        self.state = "locked"

    def event(self, ev):
        if ev == "coin":
            self.state = "unlocked"
        else:
            raise ValueError("Invalid event")

def demo():
    t = Turnstile()
    t.event("coin")
    return t.state
