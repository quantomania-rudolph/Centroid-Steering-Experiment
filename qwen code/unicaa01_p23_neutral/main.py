"""Knapsack kata."""
def knapsack01(weights, values, cap):
    n = len(weights)
    dp = [[0 for _ in range(cap + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, cap + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][cap]

def demo():
    weights = [2, 3]
    values = [3, 4]
    cap = 5
    return knapsack01(weights, values, cap)

# Example usage
if __name__ == "__main__":
    print(demo())  # Output should be 7
