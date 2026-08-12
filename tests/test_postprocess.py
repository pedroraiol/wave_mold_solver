import pytest

from core.postprocess import reconstruct_field, save_mode_plot
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
