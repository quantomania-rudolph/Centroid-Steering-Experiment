"""Topo kata."""
from collections import defaultdict, deque


def topo_sort(graph):
    in_degree = defaultdict(int)
    result = []
    queue = deque()

    # Calculate in-degree for each node
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    # Enqueue all nodes with in-degree 0
    for node in graph:
        if in_degree[node] == 0:
            queue.append(node)

    # Process the queue
    while queue:
        node = queue.popleft()
        result.append(node)

        # Decrease in-degree of neighbors and enqueue if in-degree becomes 0
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result

def demo():
    g = {"a": ["b"], "b": ["c"], "c": []}
    sorted_nodes = topo_sort(g)
    print("Topological Sort:", sorted_nodes)

if __name__ == "__main__":
    demo()
