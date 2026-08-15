"""Kth smallest kata."""
import heapq


def kth_smallest(nums: list[int], k: int) -> int:
    # Use a min-heap to find the kth smallest element
    heapq.heapify(nums)
    for _ in range(k - 1):
        heapq.heappop(nums)
    return heapq.heappop(nums)

def demo():
    # Test the kth_smallest function with the example provided
    nums = [7, 3, 5]
    k = 2
    result = kth_smallest(nums, k)
    print(f"The {k}th smallest element in {nums} is {result}")

if __name__ == '__main__':
    demo()
