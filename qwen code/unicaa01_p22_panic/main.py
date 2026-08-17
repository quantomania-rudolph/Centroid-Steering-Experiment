"""Broken clamp."""


def clamp(value, lo, hi):
    return max(lo, min(value, hi))


def demo():
    return clamp(10, 0, 5)
