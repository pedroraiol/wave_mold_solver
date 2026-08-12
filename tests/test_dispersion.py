import numpy as np
import pytest

from core.dispersion import sweep_wavelengths
from core.postprocess import save_dispersion_plot
from solvers.tmm import TMMSolver

SOLVER = TMMSolver()


def test_sweep_groups_by_mode_index_and_restores_wavelength(soi_problem):
    problem = soi_problem(method="TMM")
    original_wavelength = problem.wavelength_nm
    wavelengths_nm = np.linspace(1500.0, 1600.0, 5)

    modes_by_index = sweep_wavelengths(problem, SOLVER, wavelengths_nm)

    assert 0 in modes_by_index
    assert len(modes_by_index[0]) == len(wavelengths_nm)
    assert problem.wavelength_nm == original_wavelength


def test_fundamental_neff_decreases_with_wavelength(soi_problem):
    problem = soi_problem(method="TMM")
    wavelengths_nm = np.linspace(1500.0, 1600.0, 5)

    modes_by_index = sweep_wavelengths(problem, SOLVER, wavelengths_nm)
    points = sorted(modes_by_index[0])
    neffs = [neff for _, neff in points]

    assert neffs == sorted(neffs, reverse=True)


def test_higher_order_mode_curve_ends_before_cutoff(soi_problem):
    # nucleo com espessura tal que o corte do modo 1 acontece dentro do range
    # (cutoff analitico do slab simetrico ~1894nm para essa espessura)
    problem = soi_problem(thickness_nm=300.0, method="TMM")
    wavelengths_nm = np.linspace(1300.0, 2000.0, 15)

    modes_by_index = sweep_wavelengths(problem, SOLVER, wavelengths_nm)

    assert len(modes_by_index[0]) == len(wavelengths_nm)
    assert 1 in modes_by_index
    assert len(modes_by_index[1]) < len(wavelengths_nm)


def test_save_dispersion_plot_writes_png(tmp_path, soi_problem):
    problem = soi_problem(method="TMM")
    wavelengths_nm = np.linspace(1500.0, 1600.0, 5)
    modes_by_index = sweep_wavelengths(problem, SOLVER, wavelengths_nm)

    path = save_dispersion_plot(problem, modes_by_index, out_dir=str(tmp_path))

    assert path.exists()
    assert path.suffix == ".png"
