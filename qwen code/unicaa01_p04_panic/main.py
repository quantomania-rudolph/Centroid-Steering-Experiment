from collections import deque


def bfs(adj, start):
    visited = set()
    queue = deque([start])
    result = []

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            result.append(node)
            queue.extend([n for n in adj[node] if n not in visited])

    return result

def demo():
    adj = {0: [1, 2], 1: [3], 2: [], 3: []}
    return bfs(adj, 0)
