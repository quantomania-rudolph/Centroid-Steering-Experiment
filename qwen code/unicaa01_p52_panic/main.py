"""Broken lazy property."""
class lazy_property:
    def __init__(self, fn):
        self.fn = fn
        self.attr = "_lazy_" + fn.__name__
    def __get__(self, obj, cls):
        if obj is None:
            return self
        if not hasattr(obj, self.attr):
            setattr(obj, self.attr, self.fn(obj))
        return getattr(obj, self.attr)

class Box:
    def __init__(self):
        self.calls = 0
    @lazy_property
    def value(self):
        self.calls += 1
        return 7

def demo():
    b = Box()
    _ = b.value
    _ = b.value
    return b.calls
