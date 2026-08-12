import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.models import Layer, Problem  # noqa: E402

SYMMETRIC_N_CORE = 3.4
SYMMETRIC_N_CLAD = 1.45
SOI_N_CORE = 3.47
SOI_N_CLAD = 1.44


def _make_problem(n_left, n_core, n_right, thickness_nm, wavelength_nm, polarization, method):
    layers = [
        Layer(n=n_left, thickness_nm=None),
        Layer(n=n_core, thickness_nm=thickness_nm),
        Layer(n=n_right, thickness_nm=None),
    ]
    return Problem(wavelength_nm=wavelength_nm, polarization=polarization, method=method, layers=layers)


@pytest.fixture
def symmetric_slab():
    """Slab simetrico (mesmo clad dos dois lados) -- permite checar contagem de modos via V-number."""

    def _make(thickness_nm, polarization="TE", method="TMM", wavelength_nm=1550.0):
        return _make_problem(
            SYMMETRIC_N_CLAD, SYMMETRIC_N_CORE, SYMMETRIC_N_CLAD, thickness_nm, wavelength_nm, polarization, method
        )

    return _make


@pytest.fixture
def soi_problem():
    """Guia SOI padrao (clad 1.44 / core 3.47), mesmo do config.json do projeto."""

    def _make(thickness_nm=220.0, polarization="TE", method="TMM", wavelength_nm=1550.0):
        return _make_problem(SOI_N_CLAD, SOI_N_CORE, SOI_N_CLAD, thickness_nm, wavelength_nm, polarization, method)

    return _make


@pytest.fixture
def no_mode_problem():
    """Slab assimetrico com nucleo fino demais para guiar a fundamental (0 modos esperado)."""

    def _make(polarization="TE", method="TMM", wavelength_nm=1550.0):
        return _make_problem(1.44, 1.47, 1.46, 20.0, wavelength_nm, polarization, method)

    return _make
