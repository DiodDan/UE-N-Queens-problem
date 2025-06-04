import random
import math
from typing import override

from src.solver import Solver


class SimulatedAnnealingSolver(Solver):
    def __init__(
        self,
        initial_temp=100.0,
        cooling_rate=0.99,
        max_iterations=100000,
        max_retries=10,
    ):
        self.name = "Simulated Annealing Solver"
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.max_retries = max_retries

    @override
    def solve(self, n: int) -> list[tuple[int, int]]:
        def heuristic(board):
            """Number of attacking pairs."""
            conflicts = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                        conflicts += 1
            return conflicts

        def get_random_neighbor(board):
            neighbor = board[:]
            row = random.randint(0, n - 1)
            new_col = random.randint(0, n - 1)
            while new_col == neighbor[row]:
                new_col = random.randint(0, n - 1)
            neighbor[row] = new_col
            return neighbor

        def single_run():
            current = [random.randint(0, n - 1) for _ in range(n)]
            current_cost = heuristic(current)
            T = self.initial_temp

            for _ in range(self.max_iterations):
                if current_cost == 0:
                    break

                neighbor = get_random_neighbor(current)
                neighbor_cost = heuristic(neighbor)
                delta = neighbor_cost - current_cost

                if delta < 0:
                    current = neighbor
                    current_cost = neighbor_cost
                else:
                    probability = math.exp(-delta / T)
                    if random.random() < probability:
                        current = neighbor
                        current_cost = neighbor_cost

                T *= self.cooling_rate
                if T < 1e-6:
                    break

            return current if current_cost == 0 else None

        for _ in range(self.max_retries):
            result = single_run()
            if result:
                return [(i, col) for i, col in enumerate(result)]

        return []  # All retries failed
