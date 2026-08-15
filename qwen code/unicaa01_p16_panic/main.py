class Store:
    def __init__(self):
        self.balance = 0

    def apply(self, events):
        for event in events:
            if event[0] == "add":
                self.balance += event[1]
            elif event[0] == "sub":
                self.balance -= event[1]

def demo():
    s = Store()
    s.apply([("add", 10), ("add", 20), ("sub", 5)])
    return s.balance
