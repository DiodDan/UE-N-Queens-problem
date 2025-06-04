from tools.testing_engine import TestingEngine
from src.solvers.dfs_exhaustive_solver import DFSExhaustiveSolver
from src.solvers.greedy_hill_climbing_solver import GreedyHillClimbingSolver, OptimizedHillClimbingSolver
from src.solvers.simulated_annealing_solver import SimulatedAnnealingSolver
from src.solvers.genetic_algorithm_solver import GeneticAlgorithmSolver

if __name__ == "__main__":
    dfs_exhaustive_solver = DFSExhaustiveSolver()
    testing_engine_dfs = TestingEngine(dfs_exhaustive_solver)

    greedy_hill_climbing_solver = OptimizedHillClimbingSolver(max_restarts=100)
    testing_engine_hill_climbing = TestingEngine(greedy_hill_climbing_solver)

    simulated_annealing_solver = SimulatedAnnealingSolver(
        initial_temp=1000.0,
        cooling_rate=0.9995,
        max_iterations=100000,
        max_retries=10
    )
    testing_engine_simulated_annealing = TestingEngine(simulated_annealing_solver)

    genetic_algorithm_solver = GeneticAlgorithmSolver(
        population_size=100,
        mutation_rate=0.01,
        max_generations=1000,
    )
    testing_engine_genetic_algorithm = TestingEngine(genetic_algorithm_solver)


    testing_engine_hill_climbing.test_solver(250, runs=10)