"""Good fib — recursive with negative guard."""


def fib(n):
    if n < 0:
        raise ValueError("n must be >= 0")
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


def demo():
    return fib(10)
