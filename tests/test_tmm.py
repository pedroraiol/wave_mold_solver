import pytest

from solvers.tmm import TMMSolver

from _helpers import symmetric_slab_mode_count

SOLVER = TMMSolver()


@pytest.mark.parametrize("thickness_nm", [130.0, 700.0, 1100.0])
@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_symmetric_slab_mode_count_matches_v_number(symmetric_slab, thickness_nm, polarization):
    problem = symmetric_slab(thickness_nm, polarization=polarization, method="TMM")
    results = SOLVER.solve(problem)
    expected = symmetric_slab_mode_count(3.4, 1.45, thickness_nm, problem.wavelength_nm)
    assert len(results) == expected


def test_modes_are_sorted_by_descending_neff(symmetric_slab):
    problem = symmetric_slab(1100.0, method="TMM")
    results = SOLVER.solve(problem)
    neffs = [r.neff for r in results]
    assert neffs == sorted(neffs, reverse=True)
    assert [r.mode_index for r in results] == list(range(len(results)))


def test_neff_within_guided_bounds(soi_problem):
    problem = soi_problem(method="TMM")
    results = SOLVER.solve(problem)
    assert len(results) >= 1
    for result in results:
        assert 1.44 < result.neff < 3.47


def test_field_profile_is_normalized(soi_problem):
    problem = soi_problem(method="TMM")
    results = SOLVER.solve(problem)
    for result in results:
        assert result.field is not None
        assert abs(result.field.amplitude).max() == pytest.approx(1.0)


def test_no_guided_mode_returns_empty_list(no_mode_problem):
    problem = no_mode_problem(method="TMM")
    assert SOLVER.solve(problem) == []


def test_invalid_polarization_raises(soi_problem):
    problem = soi_problem(method="TMM")
    problem.polarization = "XX"
    with pytest.raises(ValueError):
        SOLVER.solve(problem)
