"""Snapshot kata."""
class Store:
    def __init__(self):
        self.balance = 0
        self.events = []

    def apply(self, events):
        for event in events:
            if event == '+10':
                self.balance += 10
                self.events.append(event)
            elif event == '+20':
                self.balance += 20
                self.events.append(event)
            elif event == '-5':
                self.balance -= 5
                self.events.append(event)

    def snapshot(self):
        return self.balance, self.events

    def replay(self, snapshot):
        self.balance, self.events = snapshot
        for event in self.events:
            self.apply([event])

def demo():
    store = Store()
    store.apply(['+10', '+20', '-5'])
    print(f"Final balance: {store.balance}")  # Should print 30

    # Take a snapshot
    snapshot = store.snapshot()

    # Reset store and replay events
    store = Store()
    store.replay(snapshot)
    print(f"Replayed balance: {store.balance}")  # Should print 30

if __name__ == '__main__':
    demo()
