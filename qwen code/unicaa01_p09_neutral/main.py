"""Pool kata."""
from concurrent.futures import ThreadPoolExecutor


def run_pool(fn, items):
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(fn, items))
    return results

def demo():
    def identity(x):
        return x
    
    items = [1, 2, 3]
    return run_pool(identity, items)
