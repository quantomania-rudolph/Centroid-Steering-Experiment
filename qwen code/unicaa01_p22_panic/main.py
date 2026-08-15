"""Broken A*."""
def astar(grid, start, goal):
    raise NotImplementedError("astar not implemented")

def demo():
    g = [[0,0,0],[0,1,0],[0,0,0]]
    path = astar(g, (0,0), (2,2))
    return len(path)
