import enum
import random
import math
from typing import override, Self
from collections import defaultdict

from tqdm import trange

from src.solver import Solver
from src.solvers.greedy_hill_climbing_solver import OptimizedHillClimbingSolver


class FitnessStrategy(enum.Enum):
    DEFAULT = "default"
    DIAG_SETS = "diag_sets"


class Chromosome:
    def __init__(
            self,
            genes: list[int],
            fitness_strategy: FitnessStrategy = FitnessStrategy.DIAG_SETS,
    ):
        self.genes = genes
        self.fitness_strategy = fitness_strategy
        self.fitness = None
        self.most_conflicting_index = None  # index of the most conflicting queen

    def calculate_fitness(self) -> int:
        match self.fitness_strategy:
            case FitnessStrategy.DEFAULT:
                return self.calculate_fitness_default()
            case FitnessStrategy.DIAG_SETS:
                return self.calculate_fitness_diag_sets()
        return 0

    def calculate_fitness_default(self) -> int:
        """Calculate the fitness of the chromosome based on the number of conflicts."""
        conflicts = 0
        for i in range(len(self.genes)):
            for j in range(i + 1, len(self.genes)):
                if abs(self.genes[i] - self.genes[j]) == abs(i - j):
                    conflicts += 1
        self.fitness = conflicts
        return self.fitness

    def calculate_fitness_diag_sets(self) -> int:
        """Optimized fitness calculation that matches the original conflict counting, including row conflicts"""
        pos_diag_counts = defaultdict(int)  # (col + row) counts
        neg_diag_counts = defaultdict(int)  # (col - row) counts
        row_counts = defaultdict(int)  # row counts for row conflicts
        conflicts = 0
        max_conflicts_per_queen = 0

        for col, row in enumerate(self.genes):
            pos_diag = col + row
            neg_diag = col - row

            # Add conflicts from existing queens on these diagonals
            conflicts_per_queen = pos_diag_counts[pos_diag] + neg_diag_counts[neg_diag] + row_counts[row]
            conflicts += conflicts_per_queen

            # Track the maximum conflicts for any queen
            if conflicts_per_queen > max_conflicts_per_queen:
                max_conflicts_per_queen = conflicts_per_queen
                self.most_conflicting_index = col

            # Increment counts for these diagonals and row
            pos_diag_counts[pos_diag] += 1
            neg_diag_counts[neg_diag] += 1
            row_counts[row] += 1

        if self.most_conflicting_index is None:
            self.most_conflicting_index = random.randint(0, len(self.genes) - 1)
        self.fitness = conflicts
        return conflicts

    def mutate(self) -> None:
        if self.most_conflicting_index is None:
            self.calculate_fitness()
        self.genes[self.most_conflicting_index] = random.randint(0, len(self.genes) - 1)

    def __str__(self):
        return str(self.genes)


class NovelMutationGeneticSolver(Solver):
    def __init__(
            self,
            population_size: int = 200,
            max_generations: int = 5000,
            crossover_rate: float = 0.8,
            mutation_rate: float = 0.2,
            elite_size: int = 30,
            fitness_strategy: FitnessStrategy = FitnessStrategy.DIAG_SETS,
            kill_bad_genes_on_born: bool = True,
            bad_genes_lower_avg: float = 0.1,
            max_bad_genes_retries: int = 10,
            crossover_type: str = "pmx",
            use_hill_climbing: bool = True,
            stagination_threshold: int = 600,
            name: str = "Novel Mutation Genetic Solver",
    ):
        self.name = name
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.fitness_strategy = fitness_strategy
        self.kill_bad_genes_on_born = kill_bad_genes_on_born
        self.bad_genes_lower_avg = bad_genes_lower_avg
        self.max_bad_genes_retries = max_bad_genes_retries
        self.crossover_type = crossover_type
        self.use_hill_climbing = use_hill_climbing
        self.hill_climber = OptimizedHillClimbingSolver(max_restarts=1) if use_hill_climbing else None
        self.stagnation_threshold = stagination_threshold

    @override
    def solve(self, n: int) -> list[tuple[int, int]]:
        population = self._initialize_population(n)
        best_solution = None
        best_fitness = float('inf')
        stagnation_counter = 0

        for generation in range(self.max_generations):
            # Evaluate population
            population = sorted(population, key=lambda c: c.calculate_fitness())
            current_best = population[0]

            # Update best solution found so far
            if current_best.fitness < best_fitness:
                best_fitness = current_best.fitness
                best_solution = current_best.genes.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Check for solution
            if best_fitness == 0:
                return self._convert_to_queen_positions(best_solution)

            # Apply hill climbing to nearly-solved solutions
            if self.use_hill_climbing and best_fitness <= 1:
                hc_solution = self.hill_climber.solve(n)
                if hc_solution and len(hc_solution) == n:
                    return hc_solution

            # Create next generation
            next_generation = population[:self.elite_size]

            while len(next_generation) < self.population_size:
                parent1, parent2 = self._select_parents(population)

                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2, self.crossover_type)
                else:
                    child1, child2 = parent1, parent2

                child2.calculate_fitness()
                child1.calculate_fitness()

                # Mutation with adaptive rate
                current_mutation_rate = min(0.5, self.mutation_rate + (stagnation_counter * 0.005))
                if random.random() < current_mutation_rate:
                    child1.mutate()
                if random.random() < current_mutation_rate:
                    child2.mutate()

                # Add the better child to next generation
                if child1.fitness < child2.fitness:
                    next_generation.append(child1)
                else:
                    next_generation.append(child2)

            population = next_generation

            # Restart mechanism if stuck
            if stagnation_counter > self.stagnation_threshold:
                population = self._restart_population(population, n)
                stagnation_counter = 0

        # Final attempt with hill climbing if we're close
        if self.use_hill_climbing and best_fitness <= 2:
            hc_solution = self.hill_climber.solve(n)
            if hc_solution and len(hc_solution) == n:
                return hc_solution

        return self._convert_to_queen_positions(best_solution)

    def _restart_population(self, population: list[Chromosome], n: int) -> list[Chromosome]:
        """Partial population restart to maintain diversity"""
        keep = min(5, len(population) // 10)
        new_population = sorted(population, key=lambda c: c.fitness)[:keep]
        new_population.extend(self._initialize_population(n, size=self.population_size - keep))
        return new_population

    def _initialize_population(self, n: int, size: int = None) -> list[Chromosome]:
        """Initialize population with optional size parameter"""
        if size is None:
            size = self.population_size
        return [self.create_chromosome(n) for _ in range(size)]

    def _select_parents(self, population: list[Chromosome]) -> tuple[Chromosome, Chromosome]:
        """Select parents using tournament selection."""
        tournament_size = max(2, len(population) // 10)
        tournament = random.sample(population, tournament_size)
        parent1 = min(tournament, key=lambda c: c.fitness)

        tournament = random.sample(population, tournament_size)
        parent2 = min(tournament, key=lambda c: c.fitness)

        return parent1, parent2

    def _convert_to_queen_positions(self, genes: list[int]) -> list[tuple[int, int]]:
        """Convert chromosome genes to queen positions (col, row)."""
        return [(row, col) for col, row in enumerate(genes)]

    def create_chromosome(self, size: int) -> Chromosome:
        """Create a new chromosome with the given size."""
        genes = self.generate_random_genes(size)
        return Chromosome(genes=genes, fitness_strategy=self.fitness_strategy)

    def generate_random_genes(self, n: int) -> list[int]:
        """Generate a random permutation of genes for the chromosome."""
        best = None
        best_fitness = None
        retry = 0
        average = self.approx_avg_conflicts(n)

        while (self.kill_bad_genes_on_born and
               retry < self.max_bad_genes_retries):
            genes = random.sample(range(n), n)
            chromosome = Chromosome(genes=genes, fitness_strategy=self.fitness_strategy)
            fitness = chromosome.calculate_fitness()

            if best_fitness is None or fitness < best_fitness:
                best_fitness = fitness
                best = genes.copy()

            if fitness <= (average - average * self.bad_genes_lower_avg):
                break

            retry += 1

        return best if best is not None else random.sample(range(n), n)

    def crossover(self, parent1: Chromosome, parent2: Chromosome, crossover_type: str = "pmx") -> tuple[
        Chromosome, Chromosome]:
        """Perform crossover between two parent chromosomes.

        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            crossover_type: Type of crossover to perform ("pmx" or "ox")

        Returns:
            Tuple of two child chromosomes
        """
        if len(parent1.genes) != len(parent2.genes):
            raise ValueError("Parent chromosomes must be of same length")

        if crossover_type == "pmx":
            return self._pmx_crossover(parent1, parent2)
        elif crossover_type == "ox":
            return self._ordered_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover type: {crossover_type}")

    def _pmx_crossover(self, parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
        """Improved Partially Matched Crossover (PMX) implementation without infinite loops."""
        size = len(parent1.genes)
        if size < 2:
            return parent1, parent2

        # Ensure distinct crossover points
        cx1, cx2 = 0, 0
        while cx1 == cx2:
            cx1, cx2 = sorted(random.sample(range(size), 2))

        def _pmx_child(p1_genes, p2_genes):
            child_genes = [None] * size
            # Copy the segment between cx1 and cx2 from p2 to child
            child_genes[cx1:cx2] = p2_genes[cx1:cx2]

            # Create mapping for genes in the segment
            mapping = {}
            for i in range(cx1, cx2):
                if p2_genes[i] not in mapping:
                    mapping[p2_genes[i]] = p1_genes[i]

            # Fill remaining positions with safety checks
            for i in list(range(0, cx1)) + list(range(cx2, size)):
                gene = p1_genes[i]
                visited = set()
                while gene in mapping and gene not in visited:
                    visited.add(gene)
                    gene = mapping[gene]
                child_genes[i] = gene

            return child_genes

        child1_genes = _pmx_child(parent1.genes, parent2.genes)
        child2_genes = _pmx_child(parent2.genes, parent1.genes)

        return (
            Chromosome(genes=child1_genes, fitness_strategy=self.fitness_strategy),
            Chromosome(genes=child2_genes, fitness_strategy=self.fitness_strategy)
        )

    def _ordered_crossover(self, parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
        """Improved Ordered Crossover (OX) implementation without infinite loops."""
        size = len(parent1.genes)
        if size < 2:
            return parent1, parent2

        # Ensure distinct crossover points
        cx1, cx2 = 0, 0
        while cx1 == cx2:
            cx1, cx2 = sorted(random.sample(range(size), 2))

        def _ox_child(p1_genes, p2_genes):
            child_genes = [None] * size
            # Copy the segment between cx1 and cx2 from p1 to child
            child_genes[cx1:cx2] = p1_genes[cx1:cx2]

            # Create a set of already included genes for faster lookup
            included_genes = set(child_genes[cx1:cx2])

            # Fill remaining positions with genes from p2 in order
            p2_ptr = 0
            remaining_positions = list(range(0, cx1)) + list(range(cx2, size))

            for i in remaining_positions:
                # Find next gene in p2 that's not already in the child
                while p2_ptr < size and p2_genes[p2_ptr] in included_genes:
                    p2_ptr += 1

                if p2_ptr >= size:
                    # Shouldn't happen for valid permutations, but fallback to random
                    remaining_genes = [g for g in p2_genes if g not in included_genes]
                    if not remaining_genes:
                        # Emergency fallback - this should never happen with valid inputs
                        remaining_genes = [g for g in range(size) if g not in included_genes]
                    child_genes[i] = random.choice(remaining_genes)
                else:
                    child_genes[i] = p2_genes[p2_ptr]
                    included_genes.add(p2_genes[p2_ptr])
                    p2_ptr += 1

            return child_genes

        child1_genes = _ox_child(parent1.genes, parent2.genes)
        child2_genes = _ox_child(parent2.genes, parent1.genes)

        return (
            Chromosome(genes=child1_genes, fitness_strategy=self.fitness_strategy),
            Chromosome(genes=child2_genes, fitness_strategy=self.fitness_strategy)
        )

    @staticmethod
    def approx_avg_conflicts(n: int) -> float:
        return (2 * n - 1) / 3 - math.pi / 4



