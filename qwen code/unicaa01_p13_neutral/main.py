"""Matmul kata."""
def matmul(a, b):
    """Multiply two 2x2 matrices."""
    return [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]]
    ]

def demo():
    """Check the matmul function with known results."""
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    expected = [[19, 22], [43, 50]]
    result = matmul(a, b)
    assert result == expected, f"Expected {expected}, but got {result}"
    print("Matrix multiplication demo passed!")

if __name__ == "__main__":
    demo()
