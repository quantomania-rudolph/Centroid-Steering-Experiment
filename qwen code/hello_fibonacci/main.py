"""Task workspace for Aider + Qwen — fibonacci."""

def fibonacci(n: int) -> int:
    """Return nth Fibonacci number (0-indexed)."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def main() -> None:
    print(fibonacci(10))


if __name__ == "__main__":
    main()
