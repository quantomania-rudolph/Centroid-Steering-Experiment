"""Bisect kata."""
def bisect_left(a: list[int], x: int) -> int:
    left, right = 0, len(a)
    while left < right:
        mid = (left + right) // 2
        if a[mid] < x:
            left = mid + 1
        else:
            right = mid
    return left

def demo():
    a = [1, 3]
    x = 2
    index = bisect_left(a, x)
    print(f"Insert {x} at index {index} in {a} to maintain sorted order: {a[:index] + [x] + a[index:]}")

if __name__ == '__main__':
    demo()
