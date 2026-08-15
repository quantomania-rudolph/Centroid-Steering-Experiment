"""FSM kata."""
class Turnstile:
    def __init__(self):
        self.state = 'locked'

    def event(self, ev):
        if self.state == 'locked':
            if ev == 'coin':
                self.state = 'unlocked'
        elif self.state == 'unlocked' and ev == 'pass':
            self.state = 'locked'

def demo():
    turnstile = Turnstile()
    print(f"Initial state: {turnstile.state}")
    turnstile.event('coin')
    print(f"After coin: {turnstile.state}")
    turnstile.event('pass')
    print(f"After pass: {turnstile.state}")
