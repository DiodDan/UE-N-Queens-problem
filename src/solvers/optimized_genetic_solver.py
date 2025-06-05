import random
import math
from typing import override, List, Tuple
from src.solver import Solver


class OptimizedGeneticSolver(Solver):
    def __init__(
            self,
            population_size: int = 200,
            initial_mutation_rate: float = 0.15,
            max_generations: int = 2000,
            elite_size: int = 10,
            tournament_size: int = 5,
            restart_threshold: int = 100
    ):
        self.name = "Optimized Genetic Solver"
        self.population_size = population_size
        self.initial_mutation_rate = initial_mutation_rate
        self.max_generations = max_generations
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.restart_threshold = restart_threshold

    @override
    def solve(self, n: int) -> List[Tuple[int, int]]:
        def generate_board() -> List[int]:
            """Generate board with minimized initial conflicts using Fisher-Yates shuffle"""
            board = list(range(n))
            for i in range(n - 1, 0, -1):
                j = random.randint(0, i)
                board[i], board[j] = board[j], board[i]
            return board

        def heuristic(board: List[int]) -> int:
            """Fast conflict counter using diagonal checks"""
            conflicts = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(board[i] - board[j]) == abs(i - j):
                        conflicts += 1
            return conflicts

        def cycle_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
            """Fixed Cycle Crossover for permutations"""
            child = [None] * n
            visited = set()
            cycle = 0

            while None in child:
                # Find next unassigned position
                idx = child.index(None)
                val = parent1[idx]

                # Follow the cycle
                while idx not in visited:
                    child[idx] = parent1[idx] if cycle % 2 == 0 else parent2[idx]
                    visited.add(idx)
                    val = parent1[idx]
                    idx = parent2.index(val)

                cycle += 1

            return child

        def mutate(board: List[int], mutation_rate: float) -> List[int]:
            """Swap mutation with increased probability for conflicting queens"""
            if random.random() < mutation_rate:
                # Find conflicting queens
                conflicts = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if abs(board[i] - board[j]) == abs(i - j):
                            conflicts.append(i)
                            conflicts.append(j)

                # If conflicts exist, mutate one of them
                if conflicts:
                    i = random.choice(list(set(conflicts)))  # Remove duplicates
                    j = random.randint(0, n - 1)
                    board[i], board[j] = board[j], board[i]
                else:
                    # Random swap if no conflicts found
                    i, j = random.sample(range(n), 2)
                    board[i], board[j] = board[j], board[i]
            return board

        def tournament_selection(population: List[List[int]]) -> List[int]:
            """Tournament selection with pressure"""
            candidates = random.sample(population, min(self.tournament_size, len(population)))
            return min(candidates, key=heuristic)

        # Initialize population
        population = [generate_board() for _ in range(self.population_size)]
        best_fitness = math.inf
        stagnation = 0
        best_solution = None

        for generation in range(self.max_generations):
            # Evaluate population
            population.sort(key=heuristic)
            current_best = population[0]
            current_fitness = heuristic(current_best)

            # Track best solution
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution = current_best.copy()
                stagnation = 0
                if best_fitness == 0:
                    break

            stagnation += 1

            # Adaptive mutation rate
            mutation_rate = min(0.5, self.initial_mutation_rate * (1 + math.log(1 + stagnation)))

            # Population restart if stagnated
            if stagnation > self.restart_threshold:
                # Keep top 10% and replace the rest
                keep = max(1, self.population_size // 10)
                population = population[:keep] + [generate_board() for _ in range(self.population_size - keep)]
                stagnation = 0
                mutation_rate = self.initial_mutation_rate
                continue

            # Elitism
            new_population = population[:self.elite_size]

            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = tournament_selection(population)
                parent2 = tournament_selection(population)
                child = cycle_crossover(parent1, parent2)
                child = mutate(child, mutation_rate)
                new_population.append(child)

            population = new_population

        # Return best solution found
        if best_solution is None:
            population.sort(key=heuristic)
            best_solution = population[0]
        return [(i, col) for i, col in enumerate(best_solution)]