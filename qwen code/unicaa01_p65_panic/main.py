"""Merge overlapping intervals."""


def merge_intervals(iv: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = []
    for interval in sorted(iv):
        if merged and interval[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
        else:
            merged.append(interval)
    return merged


def demo():
    return merge_intervals([(1, 3), (2, 6)])
