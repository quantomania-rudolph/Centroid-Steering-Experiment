"""Outbox kata."""
class Outbox:
    def __init__(self):
        self.events = []

    def add(self, ev):
        self.events.append(ev)

    def drain(self):
        events = self.events
        self.events = []
        return events

def demo():
    outbox = Outbox()
    outbox.add("Event 1")
    outbox.add("Event 2")
    return outbox.drain()
