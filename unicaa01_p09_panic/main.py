"""Fixed pool."""
from concurrent.futures import ThreadPoolExecutor


def run_pool(fn, items):
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(fn, items))
    return results


def demo():
    return sorted(run_pool(lambda x: x, [3, 1, 2]))
