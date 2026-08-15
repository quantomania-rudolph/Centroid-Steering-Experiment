"""Task workspace for Aider + Qwen."""

def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def main() -> None:
    print(fibonacci(10))

if __name__ == "__main__":
    main()
