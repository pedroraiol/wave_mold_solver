from core.models import ModeResult, Problem
from solvers import BaseSolver


class FEMSolver(BaseSolver):
    def solve(self, problem: Problem) -> list[ModeResult]:
        # proxima fase: implementar em 2D escalar
        raise NotImplementedError("FEM ainda não implementado")
