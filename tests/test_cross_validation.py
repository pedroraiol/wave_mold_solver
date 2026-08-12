import pytest

from solvers.fem import FEMSolver
from solvers.tmm import TMMSolver

# TMM (analitico por camada) e FEM (elementos finitos truncados em caixa) resolvem
# o mesmo problema fisico por metodos independentes -- devem concordar em neff
# dentro da margem esperada de discretizacao/truncamento do FEM (ver README).
NEFF_TOLERANCE = 4e-3

TMM = TMMSolver()
FEM = FEMSolver()


def _assert_neffs_agree(problem_tmm_method):
    problem_tmm_method.method = "TMM"
    tmm_neffs = [r.neff for r in TMM.solve(problem_tmm_method)]
    problem_tmm_method.method = "FEM"
    fem_neffs = [r.neff for r in FEM.solve(problem_tmm_method)]

    assert len(fem_neffs) == len(tmm_neffs)
    for tmm_neff, fem_neff in zip(tmm_neffs, fem_neffs):
        assert fem_neff == pytest.approx(tmm_neff, abs=NEFF_TOLERANCE)


@pytest.mark.parametrize("thickness_nm", [220.0, 1200.0])
@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_fem_matches_tmm_on_soi_stack(soi_problem, thickness_nm, polarization):
    problem = soi_problem(thickness_nm=thickness_nm, polarization=polarization)
    _assert_neffs_agree(problem)


@pytest.mark.parametrize("thickness_nm", [130.0, 700.0, 1100.0])
@pytest.mark.parametrize("polarization", ["TE", "TM"])
def test_fem_matches_tmm_on_symmetric_slab(symmetric_slab, thickness_nm, polarization):
    problem = symmetric_slab(thickness_nm, polarization=polarization)
    _assert_neffs_agree(problem)


def test_both_solvers_agree_on_no_guided_mode(no_mode_problem):
    problem = no_mode_problem()
    _assert_neffs_agree(problem)
