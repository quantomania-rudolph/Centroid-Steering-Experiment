"""Broken A*."""


def astar(grid, start, goal):
    from heapq import heappop, heappush

    rows, cols = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_list = []
    heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_list:
        current = heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dr, dc in directions:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0]][neighbor[1]] == 0:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(open_list, (f_score[neighbor], neighbor))

    return []


def demo():
    g = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    path = astar(g, (0, 0), (2, 2))
    return len(path)
