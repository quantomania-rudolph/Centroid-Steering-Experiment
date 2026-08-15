def edit_distance(a, b):
    if a == b:
        return 0
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            if ca == cb:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j - 1], dp[j])
            prev = cur
    return dp[m]


def demo():
    return edit_distance("kitten", "sitting")
