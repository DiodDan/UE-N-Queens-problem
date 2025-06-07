import signal
from abc import ABC, abstractmethod
from typing import List, Tuple


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException()


class Solver(ABC):
    name: str

    @abstractmethod
    def solve(self, n: int) -> List[Tuple[int, int]]:
        raise NotImplementedError("Subclasses must implement the solve method.")

    def solve_with_timeout(self, n: int, timeout_seconds: int) -> List[Tuple[int, int]]:
        """
        Calls solve(n) and raises TimeoutException if it takes longer than timeout_seconds.
        """
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            return self.solve(n)
        finally:
            signal.alarm(0)  # Always cancel the alarm
