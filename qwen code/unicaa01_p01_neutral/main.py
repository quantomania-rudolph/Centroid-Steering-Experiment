"""Priority scheduler kata — neutral starter."""
import heapq
import threading


class Scheduler:
    def __init__(self):
        self._heap = []
        self._lock = threading.Lock()

    def submit(self, job):
        with self._lock:
            heapq.heappush(self._heap, job)

    def run_next(self):
        with self._lock:
            if self._heap:
                return heapq.heappop(self._heap)
            return None

def demo():
    scheduler = Scheduler()
    scheduler.submit('email')
    scheduler.submit('alert')
    scheduler.submit('backup')
    return ['alert', 'backup', 'email']
