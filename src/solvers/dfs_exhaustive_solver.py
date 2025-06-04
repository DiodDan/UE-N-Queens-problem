from typing import override

from src.solver import Solver


class DFSExhaustiveSolver(Solver):
    def __init__(self):
        self.name = "DFS Exhaustive Solver"

    @override
    def solve(self, n) -> list[tuple[int, int]]:
        def is_safe(queens: list[tuple[int, int]], row: int, col: int) -> bool:
            """Check if placing a queen at (row, col) is safe given existing queens."""
            return all(
                c != col and (r - row) != (c - col) and (r - row) != (col - c)
                for r, c in queens
            )

        def dfs(row: int, queens: list[tuple[int, int]]) -> list[tuple[int, int]]:
            """Depth-first search to find a valid queen placement."""
            if row == n:
                return queens.copy()
            for col in range(n):
                if is_safe(queens, row, col):
                    queens.append((row, col))
                    result = dfs(row + 1, queens)
                    if result:
                        return result
                    queens.pop()
            return []

        solution = dfs(0, [])
        return solution if solution else []
