import threading
import time
from typing import Optional

from tools.benchmark_engine import BenchmarkEngine, aggregate_benchmarks
from src.solvers.dfs_exhaustive_solver import DFSExhaustiveSolver
from src.solvers.greedy_hill_climbing_solver import GreedyHillClimbingSolver, OptimizedHillClimbingSolver
from src.solvers.genetic_algorithm_solver import GeneticAlgorithmSolver
from src.solvers.optimized_genetic_solver import OptimizedGeneticSolver
from src.solvers.novel_mutation_genetic_solver import NovelMutationGeneticSolver, FitnessStrategy
from src.solvers.simulated_annealing_solver import SimulatedAnnealingSolver

RUN_IN_MULTITHREAD = False
RUN_FULL_TESTS = True

EXECUTION_CONFIGS = [
    (4, 50),
    (5, 50),
    (6, 50),
    (7, 50),
    (8, 50),
    (9, 30),
    (10, 30),
    (16, 30),
    (20, 10),
    (26, 10),
    (30, 10),
    (35, 10),
    (40, 10),
    (50, 10),
    (60, 10),
    (70, 10),
    (80, 10),
    (90, 10),
    (100, 5),

]


def run_DFSExhaustiveSolver_benchmark() -> Optional[threading.Thread]:
    solver = DFSExhaustiveSolver()

    benchmark_engine = BenchmarkEngine(
        solver=solver,
        execution_configs=EXECUTION_CONFIGS,
        plot_name="DFSExhaustiveSolver",
        print_logs=False
    )
    if RUN_IN_MULTITHREAD:
        return benchmark_engine.run_in_thread()
    else:
        benchmark_engine.run()
        return None


def run_GeneticAlgorithmSolver_benchmark() -> Optional[threading.Thread]:
    solver = GeneticAlgorithmSolver(
        population_size=100,
        mutation_rate=0.01,
        max_generations=1000,
    )

    benchmark_engine = BenchmarkEngine(
        solver=solver,
        execution_configs=EXECUTION_CONFIGS,
        plot_name="GeneticAlgorithmSolver",
        print_logs=False
    )

    if RUN_IN_MULTITHREAD:
        return benchmark_engine.run_in_thread()
    else:
        benchmark_engine.run()
        return None


def run_OptimizedGeneticSolver_benchmark() -> Optional[threading.Thread]:
    solver = OptimizedGeneticSolver()

    benchmark_engine = BenchmarkEngine(
        solver=solver,
        execution_configs=EXECUTION_CONFIGS,
        plot_name="OptimizedGeneticSolver",
        print_logs=False
    )

    if RUN_IN_MULTITHREAD:
        return benchmark_engine.run_in_thread()
    else:
        benchmark_engine.run()
        return None


def run_GreedyHillClimbingSolver_benchmark() -> Optional[threading.Thread]:
    solver = GreedyHillClimbingSolver(max_restarts=100)

    benchmark_engine = BenchmarkEngine(
        solver=solver,
        execution_configs=EXECUTION_CONFIGS,
        plot_name="GreedyHillClimbingSolver",
        print_logs=False
    )

    if RUN_IN_MULTITHREAD:
        return benchmark_engine.run_in_thread()
    else:
        benchmark_engine.run()
        return None


def run_OptimizedHillClimbingSolver_benchmark() -> Optional[threading.Thread]:
    solver = OptimizedHillClimbingSolver()

    benchmark_engine = BenchmarkEngine(
        solver=solver,
        execution_configs=EXECUTION_CONFIGS,
        plot_name="OptimizedHillClimbingSolver_test",
        print_logs=False
    )

    if RUN_IN_MULTITHREAD:
        return benchmark_engine.run_in_thread()
    else:
        benchmark_engine.run()
        return None


def run_NovelMutationGeneticSolver_benchmark() -> Optional[threading.Thread]:
    solver = NovelMutationGeneticSolver(
        population_size=200,
        max_generations=5000,
        crossover_rate=0.8,
        mutation_rate=0.2,
        elite_size=30,
        fitness_strategy=FitnessStrategy.DIAG_SETS,
        kill_bad_genes_on_born=True,
        bad_genes_lower_avg=0.1,
        max_bad_genes_retries=10,
        crossover_type="pmx",
        use_hill_climbing=True,
        stagination_threshold=600,
        name="Novel Mutation Genetic Solver",
    )

    benchmark_engine = BenchmarkEngine(
        solver=solver,
        execution_configs=EXECUTION_CONFIGS,
        plot_name="NovelMutationGeneticSolver",
        print_logs=False
    )

    if RUN_IN_MULTITHREAD:
        return benchmark_engine.run_in_thread()
    else:
        benchmark_engine.run()
        return None


def run_SimulatedAnnealingSolver_benchmark() -> Optional[threading.Thread]:
    solver = SimulatedAnnealingSolver(
        initial_temp=1000,
        cooling_rate=0.998,
        max_iterations=100000,
        max_retries=50,
    )

    benchmark_engine = BenchmarkEngine(
        solver=solver,
        execution_configs=EXECUTION_CONFIGS,
        plot_name="SimulatedAnnealingSolver",
        print_logs=False
    )

    if RUN_IN_MULTITHREAD:
        return benchmark_engine.run_in_thread()
    else:
        benchmark_engine.run()
        return None


def seconds_to_human_readable(seconds: float) -> str:
    """Convert seconds to a human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} minutes"
    else:
        return f"{seconds / 3600:.2f} hours"


if __name__ == "__main__":
    start = time.time()
    if RUN_FULL_TESTS:
        # Running all algorithms benchmarks
        threads = []

        # threads.append(run_DFSExhaustiveSolver_benchmark())  # DFS Exhaustive Solver
        #
        # threads.append(run_GeneticAlgorithmSolver_benchmark())  # Genetic Algorithm Solver
        #
        # threads.append(run_OptimizedGeneticSolver_benchmark())  # Optimized Genetic Solver
        #
        # threads.append(run_GreedyHillClimbingSolver_benchmark())  # Greedy Hill Climbing Solver

        threads.append(run_OptimizedHillClimbingSolver_benchmark())  # Optimized Hill Climbing Solver

        # threads.append(run_SimulatedAnnealingSolver_benchmark())  # Simulated Annealing Solver
        #
        # threads.append(run_NovelMutationGeneticSolver_benchmark())  # Novel Mutation Genetic Solver

        # Wait for all threads to complete
        for thread in threads:
            if thread is not None:
                thread.join()

        # Aggregate all benchmarks
        aggregate_benchmarks()

    if not RUN_FULL_TESTS:
        # run_NovelMutationGeneticSolver_benchmark()
        # run_NovelMutationGeneticSolver_benchmark()
        run_DFSExhaustiveSolver_benchmark()

    print("All benchmarks completed in", seconds_to_human_readable(time.time() - start))
