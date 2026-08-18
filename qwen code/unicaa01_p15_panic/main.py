"""Retry with backoff."""
import time
import random

def retry(fn, max_tries=3):
    for attempt in range(max_tries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_tries - 1:
                raise e
            backoff = random.uniform(0.1, 2 ** attempt)
            time.sleep(backoff)

def demo():
    n = 0
    def fail():
        nonlocal n; n += 1; raise RuntimeError
    try:
        retry(fail)
    except RuntimeError:
        return n
