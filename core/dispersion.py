import numpy as np

from core.models import Problem
from solvers import BaseSolver


def sweep_wavelengths(
    problem: Problem, solver: BaseSolver, wavelengths_nm: np.ndarray
) -> dict[int, list[tuple[float, float]]]:
    """Resolve o problema em cada comprimento de onda e agrupa (wavelength_nm, neff) por mode_index.

    Modos guiados não se cruzam em neff para um guia regular, então agrupar
    por mode_index (0=fundamental, 1=primeira ordem, ...) já produz curvas de
    dispersão contínuas; um modo que atinge o corte dentro do range
    simplesmente para de aparecer a partir dali.
    """
    modes_by_index: dict[int, list[tuple[float, float]]] = {}
    original_wavelength = problem.wavelength_nm
    try:
        for wavelength_nm in wavelengths_nm:
            problem.wavelength_nm = float(wavelength_nm)
            for result in solver.solve(problem):
                modes_by_index.setdefault(result.mode_index, []).append((float(wavelength_nm), result.neff))
    finally:
        problem.wavelength_nm = original_wavelength
    return modes_by_index
