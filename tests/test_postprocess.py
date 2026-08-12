import pytest

from core.postprocess import confinement_factor, reconstruct_field, save_mode_plot
from solvers.fem import FEMSolver
from solvers.tmm import TMMSolver

SOLVER = TMMSolver()


def test_reconstruct_field_matches_solver_output(soi_problem):
    problem = soi_problem(method="TMM")
    results = SOLVER.solve(problem)
    assert results

    field = reconstruct_field(problem, results[0].neff)
    assert len(field.x_nm) == len(field.amplitude)
    assert abs(field.amplitude).max() == pytest.approx(1.0)


def test_save_mode_plot_writes_png(tmp_path, soi_problem):
    problem = soi_problem(method="TMM")
    results = SOLVER.solve(problem)

    paths = save_mode_plot(problem, results, out_dir=str(tmp_path))

    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].suffix == ".png"


def test_confinement_factor_is_between_zero_and_one(soi_problem):
    problem = soi_problem(method="TMM")
    result = SOLVER.solve(problem)[0]

    gamma = confinement_factor(problem, result)

    assert 0.0 < gamma < 1.0


def test_confinement_factor_increases_with_core_thickness(soi_problem):
    thin = soi_problem(thickness_nm=80.0, method="TMM")
    thick = soi_problem(thickness_nm=1200.0, method="TMM")

    gamma_thin = confinement_factor(thin, SOLVER.solve(thin)[0])
    gamma_thick = confinement_factor(thick, SOLVER.solve(thick)[0])

    assert gamma_thin < gamma_thick


@pytest.mark.parametrize("thickness_nm", [80.0, 220.0, 1200.0])
def test_confinement_factor_agrees_between_tmm_and_fem(soi_problem, thickness_nm):
    problem = soi_problem(thickness_nm=thickness_nm, method="TMM")
    tmm_gamma = confinement_factor(problem, TMMSolver().solve(problem)[0])

    problem.method = "FEM"
    fem_gamma = confinement_factor(problem, FEMSolver().solve(problem)[0])

    assert fem_gamma == pytest.approx(tmm_gamma, abs=0.03)
