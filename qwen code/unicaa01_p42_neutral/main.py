"""Semaphore kata."""
import threading
import time


class Semaphore:
    def __init__(self, n: int):
        self.n = n
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def acquire(self) -> bool:
        with self.condition:
            while self.n <= 0:
                self.condition.wait()
            self.n -= 1
            return True

    def release(self):
        with self.condition:
            self.n += 1
            self.condition.notify_all()

def demo():
    semaphore = Semaphore(1)
    def acquire_and_print():
        if semaphore.acquire():
            print("Acquired")
            time.sleep(1)
            semaphore.release()
            print("Released")
        else:
            print("Failed to acquire")

    thread1 = threading.Thread(target=acquire_and_print)
    thread2 = threading.Thread(target=acquire_and_print)

    thread1.start()
    time.sleep(0.5)  # Ensure the first thread has a chance to acquire
    thread2.start()

    thread1.join()
    thread2.join()

if __name__ == '__main__':
    demo()
