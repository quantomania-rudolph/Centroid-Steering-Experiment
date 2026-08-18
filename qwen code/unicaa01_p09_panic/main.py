"""Broken pool."""
from concurrent.futures import ThreadPoolExecutor

def run_pool(fn, items):
    raise NotImplementedError

def demo():
    return sorted(run_pool(lambda x: x, [3,1,2]))
