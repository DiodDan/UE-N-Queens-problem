import threading
from typing import Optional

from tools.benchmark_engine import BenchmarkEngine, aggregate_benchmarks
from src.solvers.dfs_exhaustive_solver import DFSExhaustiveSolver
from src.solvers.greedy_hill_climbing_solver import GreedyHillClimbingSolver, OptimizedHillClimbingSolver
from src.solvers.genetic_algorithm_solver import GeneticAlgorithmSolver
from src.solvers.optimized_genetic_solver import OptimizedGeneticSolver

RUN_IN_MULTITHREAD = True


def run_DFSExhaustiveSolver_benchmark() -> Optional[threading.Thread]:
    solver = DFSExhaustiveSolver()

    benchmark_engine = BenchmarkEngine(
        solver=solver,
        execution_configs=[
            (4, 30),
            (8, 20),
            (16, 20),
            (24, 5),
            (26, 2)
        ],
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
        execution_configs=[
            (4, 30),
            (8, 20),
            (16, 10),
            (30, 4),
            (50, 2)
        ],
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
        execution_configs=[
            (4, 30),
            (8, 20),
            (16, 10),
            (30, 10),
            (50, 5),
            (100, 2),
        ],
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
        execution_configs=[
            (4, 30),
            (8, 20),
            (16, 10),
            (30, 5),
            (35, 2),
        ],
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
        execution_configs=[
            (4, 50),
            (8, 30),
            (16, 30),
            (30, 30),
            (100, 30),
            (150, 20),
            (200, 10),
            (250, 10),
            (300, 5),
            (350, 2)
        ],
        plot_name="OptimizedHillClimbingSolver",
        print_logs=False
    )

    if RUN_IN_MULTITHREAD:
        return benchmark_engine.run_in_thread()
    else:
        benchmark_engine.run()
        return None


if __name__ == "__main__":
    # Running all algorithms benchmarks
    threads = []

    threads.append(run_DFSExhaustiveSolver_benchmark())  # DFS Exhaustive Solver

    threads.append(run_GeneticAlgorithmSolver_benchmark())  # Genetic Algorithm Solver

    threads.append(run_OptimizedGeneticSolver_benchmark())  # Optimized Genetic Solver

    threads.append(run_GreedyHillClimbingSolver_benchmark())  # Greedy Hill Climbing Solver

    threads.append(run_OptimizedHillClimbingSolver_benchmark())  # Optimized Hill Climbing Solver

    # Wait for all threads to complete
    for thread in threads:
        if thread is not None:
            thread.join()

    # Aggregate all benchmarks
    aggregate_benchmarks()
