"""Saga kata."""
import threading


class Saga:
    def __init__(self):
        self.lock = threading.Lock()

    def run(self, action, compensate):
        with self.lock:
            try:
                action()
            except Exception:
                compensate()
                raise

def demo():
    def action():
        print("Running action...")
        # Simulate a failure
        raise Exception("Action failed")

    def compensate():
        print("Compensating for action failure...")

    saga = Saga()
    try:
        saga.run(action, compensate)
    except Exception as e:
        print(f"Demo failed: {e}")

if __name__ == '__main__':
    demo()
