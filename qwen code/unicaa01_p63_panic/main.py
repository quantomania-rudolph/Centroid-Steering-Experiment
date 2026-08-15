import heapq


def kth_smallest(nums: list[int], k: int) -> int:
    return heapq.nsmallest(k, nums)[-1]

def demo():
    return kth_smallest([7, 3, 5], 2)
