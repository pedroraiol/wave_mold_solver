from abc import ABC, abstractmethod

from core.models import ModeResult, Problem


class BaseSolver(ABC):
    @abstractmethod
    def solve(self, problem: Problem) -> list[ModeResult]:
        pass
