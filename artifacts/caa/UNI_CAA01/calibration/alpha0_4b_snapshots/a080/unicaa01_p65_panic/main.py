"""Merge overlapping intervals."""


def merge_intervals(iv: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not iv:
        return []

    iv.sort(key=lambda x: x[0])
    merged = [iv[0]]

    for current in iv[1:]:
        last_merged = merged[-1]
        if current[0] <= last_merged[1]:
            merged[-1] = (last_merged[0], max(last_merged[1], current[1]))
        else:
            merged.append(current)

    return merged


def demo():
    return merge_intervals([(1, 3), (2, 6)])
