"""Retry function with max 3 retries."""


def retry(fn, max_tries=3):
    for _ in range(max_tries):
        try:
            return fn()
        except Exception:
            continue
    raise Exception("Function failed after max retries")


def demo():
    n = 0

    def fail():
        nonlocal n
        n += 1
        raise RuntimeError

    try:
        retry(fail)
    except RuntimeError:
        return n
