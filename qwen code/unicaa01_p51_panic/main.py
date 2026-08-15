"""Memoize decorator to cache function results."""
def memoize(fn):
    cache = {}
    def memoized_fn(*args):
        if args in cache:
            return cache[args]
        result = fn(*args)
        cache[args] = result
        return result
    return memoized_fn

def demo():
    @memoize
    def fib(n):
        if n < 2:
            return n
        return fib(n-1) + fib(n-2)
    return fib(10)
