"""Lazy property kata."""
class lazy_property:
    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self.doc = fn.__doc__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        if self.name not in obj.__dict__:
            obj.__dict__[self.name] = self.fn(obj)
        return obj.__dict__[self.name]

class Box:
    def __init__(self):
        self.calls = 0
    @lazy_property
    def value(self):
        self.calls += 1
        return 7

def demo():
    box = Box()
    print(box.value)  # Should print 7
    print(box.value)  # Should print 7 without incrementing calls
    print(box.calls)  # Should print 1

if __name__ == '__main__':
    demo()
