"""Broken pubsub."""
class PubSub:
    def __init__(self):
        self._subs = {}
    def subscribe(self, topic, fn):
        raise NotImplementedError("subscribe not implemented")
    def publish(self, topic, msg):
        raise NotImplementedError("publish not implemented")

def demo():
    ps = PubSub()
    out = []
    ps.subscribe("t", out.append)
    ps.publish("t", "hi")
    return out[0]
