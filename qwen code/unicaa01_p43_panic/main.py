"""Broken RW lock."""
import threading


class RWLock:
    def __init__(self):
        self._lock = threading.Condition()
        self._readers = 0

    def r_acquire(self):
        with self._lock:
            while self._readers > 0:
                self._lock.wait()
            self._readers += 1

    def r_release(self):
        with self._lock:
            self._readers -= 1
            self._lock.notify_all()

def demo():
    rw = RWLock()
    rw.r_acquire()
    rw.r_release()
    return True
