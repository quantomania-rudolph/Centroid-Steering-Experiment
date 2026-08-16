"""Broken A*."""


def astar(grid, start, goal):
    open_list = []
    closed_list = []
    open_list.append(start)

    while open_list:
        current = min(open_list, key=lambda x: x[1])
        open_list.remove(current)
        closed_list.append(current)

        if current == goal:
            path = []
            while current in closed_list:
                path.append(current)
                current = closed_list[current]
            return len(path)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] == 0 and neighbor not in closed_list:
                open_list.append(neighbor)

    return None


def demo():
    g = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    path = astar(g, (0, 0), (2, 2))
    return len(path)
