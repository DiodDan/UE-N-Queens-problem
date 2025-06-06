import time
import tracemalloc

from src.solver import Solver, TimeoutException
from tqdm import trange


class TestResult:
    def __init__(self, result: list[tuple[int, int]], time: float, peak_memory: float, correct: float, n=0, timeout: bool = False):
        self.result = result
        self.time = time
        self.peak_memory = peak_memory
        self.correct = correct
        self.n = n
        self.timeout = timeout

    def __repr__(self):
        return f"TestResult(result={self.result}, time={self.time:.2f}s, peak_memory={self.peak_memory:.2f}KB, correct={self.correct}, n={self.n})"


class TestingEngine:
    def __init__(self, solver: Solver, print_logs: bool = True, timeout_seconds: int = 120):
        self.solver: Solver = solver
        self.print_logs = print_logs
        self.timeout_seconds = timeout_seconds

    def run_and_measure(self, n: int, runs: int = 1) -> tuple[list[tuple[int, int]], float, float, float, TestResult]:
        print(f"{f'Testing {self.solver.name} with {n=} and {runs=}':^30}".ljust(30, '#'), flush=True)
        time.sleep(0.1)
        times = []
        peak_memories = []
        correct_count = 0
        result = []
        for _ in trange(runs, desc="Running tests", unit="run"):
            tracemalloc.start()
            start_time = time.perf_counter()
            try:
                result = self.solver.solve_with_timeout(n, timeout_seconds=self.timeout_seconds)
            except TimeoutException:
                print(f"Timed out for {n}.")
                return [], 0, 0, 0, TestResult([], 0, 0, 0, n, timeout=True)
            elapsed = time.perf_counter() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            times.append(elapsed)
            peak_memories.append(peak)
            if self.check_answer(n, result):
                correct_count += 1
        avg_time = sum(times) / runs
        avg_peak = sum(peak_memories) / runs
        right_percentage = (correct_count / runs) * 100
        run_results = TestResult(result, avg_time, avg_peak / 1024, right_percentage)
        return result, avg_time, avg_peak, right_percentage, run_results

    def test_solver(self, n: int, runs: int = 1) -> TestResult:
        result, avg_time, avg_peak, right_percentage, run_results = self.run_and_measure(n, runs)

        if self.print_logs:
            print(f"Right answers: {right_percentage:.2f}%")
            print(f'Execution time: {avg_time:.2f} seconds')
            print(f'Peak memory usage: {avg_peak / 1024:.2f} KB')
        return run_results

    def check_answer(self, n: int, result: list[tuple[int, int]]) -> bool:
        if len(result) != n:
            print("Result length does not match n.")
            return False
        for r, c in result:
            if not (0 <= r < n and 0 <= c < n):
                print(f"Invalid position: ({r}, {c}) not in range 0 to {n-1}.")
                return False
            for r2, c2 in result:
                if (r, c) != (r2, c2):
                    if r == r2:
                        print(f"Row conflict at row {r} with columns {c} and {c2}.")
                        return False
                    if c == c2:
                        print(f"Column conflict at column {c} with rows {r} and {r2}.")
                        return False
                    if abs(r - r2) == abs(c - c2):
                        print(f"Diagonal conflict between ({r}, {c}) and ({r2}, {c2}).")
                        return False
        return True
