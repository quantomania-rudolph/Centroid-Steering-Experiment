"""Retry implementation with max 3 retries."""


def retry(fn, max_tries=3):
    for attempt in range(max_tries):
        try:
            return fn()
        except Exception:
            if attempt == max_tries - 1:
                raise


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
