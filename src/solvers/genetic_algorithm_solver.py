import random
from typing import override
from src.solver import Solver


class GeneticAlgorithmSolver(Solver):
    def __init__(
            self,
            population_size=100,
            mutation_rate=0.05,
            max_generations=1000,
    ):
        self.name = "Genetic Algorithm Solver"
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations

    @override
    def solve(self, n: int) -> list[tuple[int, int]]:
        def heuristic(board):
            """Returns the number of attacking pairs."""
            conflicts = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(board[i] - board[j]) == abs(i - j):  # Diagonal conflict
                        conflicts += 1
            return conflicts

        def generate_board():
            """Generate a random permutation of 0..n-1 (one queen per row/col)."""
            board = list(range(n))
            random.shuffle(board)
            return board

        def crossover(parent1, parent2):
            """Order 1 crossover (preserves permutation)."""
            a, b = sorted(random.sample(range(n), 2))
            child = [None] * n
            child[a:b] = parent1[a:b]
            fill = [gene for gene in parent2 if gene not in child]
            fill_idx = 0
            for i in range(n):
                if child[i] is None:
                    child[i] = fill[fill_idx]
                    fill_idx += 1
            return child

        def mutate(board):
            """Swap two positions (row-wise permutation mutation)."""
            if random.random() < self.mutation_rate:
                i, j = random.sample(range(n), 2)
                board[i], board[j] = board[j], board[i]
            return board

        # Step 1: Initialize population
        population = [generate_board() for _ in range(self.population_size)]

        for generation in range(self.max_generations):
            population.sort(key=heuristic)
            best = population[0]
            if heuristic(best) == 0:
                return [(i, col) for i, col in enumerate(best)]

            # Step 2: Selection — top 50% survive
            survivors = population[: self.population_size // 2]

            # Step 3: Crossover + mutation to form next generation
            next_generation = survivors[:]
            while len(next_generation) < self.population_size:
                parents = random.sample(survivors, 2)
                child = crossover(parents[0], parents[1])
                child = mutate(child)
                next_generation.append(child)

            population = next_generation

        # No perfect solution found
        return []
