"""BFS kata."""
from collections import deque


def bfs(adj, start):
    queue = deque([start])
    visited = set()
    path = []

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            path.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

    return path

def demo():
    adj = {0: [1, 2], 1: [3], 2: [], 3: []}
    return bfs(adj, 0)
