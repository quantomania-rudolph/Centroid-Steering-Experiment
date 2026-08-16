"""Retry function with max 3 retries."""


def retry(fn, max_tries=3):
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_tries:
                raise Exception("Function failed after max retries")
            continue


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
