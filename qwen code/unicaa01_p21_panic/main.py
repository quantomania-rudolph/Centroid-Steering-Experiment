"""Broken Dijkstra."""
import heapq


def dijkstra(adj, src):
    dist = {node: float('inf') for node in adj}
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
    adj = {0: [(1, 1), (2, 5)], 1: [(2, 3)], 2: []}
    return dijkstra(adj, 0)[2]
