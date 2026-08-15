"""Broken circuit breaker."""
class CircuitBreaker:
    def __init__(self, threshold: int):
        self.threshold = threshold
        self.failures = 0
        self.state = "closed"
    def record_failure(self):
        raise NotImplementedError("record_failure not implemented")

def demo():
    cb = CircuitBreaker(3)
    for _ in range(3):
        cb.record_failure()
    return cb.state
