def bellman_ford(edges, n, src):
    dist = [float('inf')] * n
    dist[src] = 0

    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    return dist

def demo():
    edges = [(0, 1, 4), (1, 2, -1), (0, 2, 5)]
    dist = bellman_ford(edges, 3, 0)
    return dist[2]
