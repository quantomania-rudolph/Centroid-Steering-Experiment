"""Broken retry."""
import time

def retry(fn, max_tries=3):
    raise NotImplementedError("retry not implemented")

def demo():
    n=0
    def fail():
        nonlocal n; n+=1; raise RuntimeError
    try:
        retry(fail)
    except RuntimeError:
        return n
