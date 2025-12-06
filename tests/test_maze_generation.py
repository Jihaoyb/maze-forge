# tests/test_maze_generation.py
import os
import sys

# Ensure project root (where maze.py lives) is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from maze import Maze, MAZE_ROWS, MAZE_COLS, MAZE_CELL_SIZE


def _path_exists(maze: Maze) -> bool:
    """
    Simple DFS/BFS over the maze graph using wall flags.
    Return True if there is a path from entrance to exit.
    """
    start_r, start_c = maze.entrance
    end_r, end_c = maze.exit

    rows, cols = maze.rows, maze.cols
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    stack = [(start_r, start_c)]
    visited[start_r][start_c] = True

    # (dir_key, dr, dc)
    directions = [
        ("N", -1, 0),
        ("S",  1, 0),
        ("W",  0,-1),
        ("E",  0, 1),
    ]

    while stack:
        r, c = stack.pop()
        if (r, c) == (end_r, end_c):
            return True

        cell = maze.grid[r][c]
        for dir_key, dr, dc in directions:
            if not cell["walls"][dir_key]:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                    visited[nr][nc] = True
                    stack.append((nr, nc))

    return False


def test_maze_has_correct_dimensions():
    maze = Maze(MAZE_ROWS, MAZE_COLS, MAZE_CELL_SIZE)
    maze.generate_random(seed=0)

    assert maze.rows == MAZE_ROWS
    assert maze.cols == MAZE_COLS
    assert len(maze.grid) == MAZE_ROWS
    assert len(maze.grid[0]) == MAZE_COLS


def test_maze_has_path_from_entrance_to_exit():
    maze = Maze(10, 10, 2.0)
    maze.generate_random(seed=0)

    assert _path_exists(maze), "Maze should have a path from entrance to exit"
