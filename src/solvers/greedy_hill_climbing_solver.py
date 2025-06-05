import random
from functools import lru_cache
from typing import override, Optional

from src.solver import Solver


class GreedyHillClimbingSolver(Solver):
    def __init__(self, max_restarts: int = 200):
        self.name = "Hill Climbing Solver"
        self.base_restarts = max_restarts

    @override
    def solve(self, n: int) -> list[tuple[int, int]]:
        @lru_cache(maxsize=10000)
        def heuristic(board_tuple):
            conflicts = 0
            board = list(board_tuple)
            for i in range(n):
                for j in range(i + 1, n):
                    if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                        conflicts += 1
            return conflicts

        def get_neighbors(board):
            conflicts = set()
            for i in range(n):
                for j in range(i + 1, n):
                    if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                        conflicts.add(i)
                        conflicts.add(j)

            neighbors = []
            for row in conflicts:
                for col in range(n):
                    if col != board[row]:
                        neighbor = board.copy()
                        neighbor[row] = col
                        neighbors.append(neighbor)
            return neighbors

        def hill_climb():
            board = tuple(random.randint(0, n - 1) for _ in range(n))
            current_h = heuristic(board)

            for _ in range(1000):  # Max steps per restart
                neighbors = get_neighbors(list(board))
                if not neighbors:
                    break

                # Find best neighbor
                best_h, best_neighbor = float('inf'), None
                for nb in neighbors:
                    nb_tuple = tuple(nb)
                    h = heuristic(nb_tuple)
                    if h < best_h:
                        best_h, best_neighbor = h, nb_tuple
                        if best_h == 0:
                            return best_neighbor  # Early exit

                if best_h >= current_h:
                    break

                board, current_h = best_neighbor, best_h

            return board if current_h == 0 else None

        max_restarts = min(100, max(self.base_restarts, n * 2))
        for _ in range(max_restarts):
            solution = hill_climb()
            if solution:
                return [(i, col) for i, col in enumerate(solution)]
        return []


class OptimizedHillClimbingSolver(Solver):
    def __init__(self, max_restarts: int = 20):
        self.name = "Optimized Hill Climbing Solver"
        self.max_restarts = max_restarts

    def solve(self, n: int) -> list[tuple[int, int]]:
        def count_conflicts(board: list[int], row: int, col: int) -> int:
            """Count conflicts for a specific queen position (optimized)"""
            conflicts = 0
            for r in range(n):
                if r == row:
                    continue
                c = board[r]
                if c == col or abs(row - r) == abs(col - c):
                    conflicts += 1
            return conflicts

        def get_conflicting_queens(board: list[int]) -> list[int]:
            """Identify only queens involved in conflicts"""
            conflicts = set()
            for i in range(n):
                for j in range(i + 1, n):
                    if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                        conflicts.add(i)
                        conflicts.add(j)
            return list(conflicts)

        def hill_climb() -> Optional[list[int]]:
            board = [random.randint(0, n - 1) for _ in range(n)]

            for _ in range(200 * n):  # Max steps scales with board size
                conflicting = get_conflicting_queens(board)
                if not conflicting:
                    return board

                # Pick random conflicting queen
                row = random.choice(conflicting)

                # Find best move for this queen
                current_col = board[row]
                min_conflicts = count_conflicts(board, row, current_col)
                best_cols = [current_col]

                for col in range(n):
                    if col == current_col:
                        continue
                    conflicts = count_conflicts(board, row, col)
                    if conflicts < min_conflicts:
                        min_conflicts = conflicts
                        best_cols = [col]
                    elif conflicts == min_conflicts:
                        best_cols.append(col)

                # Move to minimal conflict position
                board[row] = random.choice(best_cols)

            return None

        for _ in range(self.max_restarts):
            solution = hill_climb()
            if solution:
                return [(i, col) for i, col in enumerate(solution)]
        return []
