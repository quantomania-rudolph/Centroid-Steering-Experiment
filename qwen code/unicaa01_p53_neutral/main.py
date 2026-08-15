"""Pubsub kata."""
class PubSub:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, topic, fn):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(fn)

    def publish(self, topic, msg):
        if topic in self.subscribers:
            for fn in self.subscribers[topic]:
                fn(msg)

def demo():
    pubsub = PubSub()

    def subscriber(msg):
        print(msg)

    pubsub.subscribe('test', subscriber)
    pubsub.publish('test', 'hi')
