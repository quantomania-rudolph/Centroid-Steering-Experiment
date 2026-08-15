

class GridMap:
    def __init__(self, grid: list[list[int]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def get_neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols and self.grid[nx][ny] != -1:
                neighbors.append((nx, ny))
        return neighbors

    def is_wall(self, x: int, y: int) -> bool:
        return self.grid[x][y] == -1

    def get_cost(self, x: int, y: int) -> int:
        return self.grid[x][y]


def reconstruct_path(came_from: dict, current: tuple[int, int]) -> list[tuple[int, int]]:
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]


def dijkstra(start: tuple[int, int], goal: tuple[int, int], grid_map: GridMap) -> tuple[int | None, list[tuple[int, int]]]:
    open_set = {start}
    came_from = {}
    g_score = {start: 0}
    f_score = {start: grid_map.get_cost(*start)}

    while open_set:
        current = min(open_set, key=lambda pos: f_score[pos])

        if current == goal:
            return g_score[goal], reconstruct_path(came_from, current)

        open_set.remove(current)

        for neighbor in grid_map.get_neighbors(*current):
            tentative_g_score = g_score[current] + grid_map.get_cost(*neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + grid_map.get_cost(*neighbor)
                if neighbor not in open_set:
                    open_set.add(neighbor)

    return None, []


def main() -> None:
    grid = [
        [0, 1, 1, 1, 1],
        [0, -1, -1, -1, 1],
        [0, 1, 0, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0]
    ]
    grid_map = GridMap(grid)
    start = (0, 0)
    goal = (4, 4)

    result = dijkstra(start, goal, grid_map)
    if result[0] is not None:
        print(f"Total cost: {result[0]}")
        print("Path:", result[1])
    else:
        print("No path found")


if __name__ == "__main__":
    main()
