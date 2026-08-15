"""Bellman-Ford kata."""
def bellman_ford(edges, n, src):
    # Initialize distances from src to all other vertices as INFINITE
    dist = [float('inf')] * n
    dist[src] = 0

    # Relax all edges |V| - 1 times. A simple shortest path from src to any other vertex can have at-most |V| - 1 edges
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    return dist

def demo():
    edges = [
        (0, 1, -1),
        (0, 2, 4),
        (1, 2, 3),
        (1, 3, 2),
        (1, 4, 2),
        (3, 2, 5),
        (3, 1, 1),
        (4, 3, -3)
    ]
    n = 5
    src = 0
    distances = bellman_ford(edges, n, src)
    return distances[2]  # Distance to node 2

# Example usage
if __name__ == '__main__':
    print(demo())  # Output should be 3
