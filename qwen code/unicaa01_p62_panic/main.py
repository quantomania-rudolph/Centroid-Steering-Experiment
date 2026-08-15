"""Broken bisect."""
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
    return bisect_left([1, 3], 2)
