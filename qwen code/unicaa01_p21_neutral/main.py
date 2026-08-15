"""Dijkstra kata."""
import heapq


def dijkstra(adj, src):
    n = len(adj)
    dist = [float('inf')] * n
    dist[src] = 0
    pq = [(0, src)]
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        if current_dist > dist[u]:
            continue
        
        for v, weight in adj[u]:
            distance = current_dist + weight
            
            if distance < dist[v]:
                dist[v] = distance
                heapq.heappush(pq, (distance, v))
    
    return dist

def demo():
    # Sample graph represented as an adjacency list
    adj = [
        [(1, 1), (2, 5)],
        [(2, 2)],
        []
    ]
    
    src = 0
    dist = dijkstra(adj, src)
    print(f"Shortest distances from node {src}: {dist}")
    assert dist[2] == 5, "The shortest path from node 0 to node 2 should be 5"

if __name__ == '__main__':
    demo()
