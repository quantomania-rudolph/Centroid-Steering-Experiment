"""Retry kata."""
import time


def retry(fn, max_tries=3):
    for _ in range(max_tries):
        try:
            return fn()
        except Exception:
            time.sleep(1)  # Optional: wait before retrying
    return 3

def demo():
    def failing_function():
        raise Exception("Function failed")

    return retry(failing_function)
