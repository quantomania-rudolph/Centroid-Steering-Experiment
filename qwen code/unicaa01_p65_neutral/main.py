"""Interval merge kata."""
def merge_intervals(iv: list[tuple[int,int]]) -> list[tuple[int,int]]:
    if not iv:
        return []

    # Sort intervals based on the start time
    iv.sort(key=lambda x: x[0])

    merged = [iv[0]]
    for current in iv[1:]:
        last_merged = merged[-1]
        if current[0] <= last_merged[1]:
            # Merge overlapping intervals
            merged[-1] = (last_merged[0], max(last_merged[1], current[1]))
        else:
            merged.append(current)

    return merged

def demo():
    intervals = [(1, 6)]
    print(merge_intervals(intervals))  # Output: [(1, 6)]

if __name__ == '__main__':
    demo()
