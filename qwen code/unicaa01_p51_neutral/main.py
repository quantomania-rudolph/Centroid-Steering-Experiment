"""Memoize kata."""
def memoize(fn):
    cache = {}
    
    def memoized_fn(*args):
        if args in cache:
            return cache[args]
        result = fn(*args)
        cache[args] = result
        return result
    
    return memoized_fn

@memoize
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

def demo():
    print(fib(10))  # Should print 55

if __name__ == '__main__':
    demo()
