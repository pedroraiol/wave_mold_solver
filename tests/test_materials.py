import numpy as np
import pytest

from core.materials import (
    boundary_kappa,
    boundary_slope_factor,
    coupling_factor,
    guided_index_bounds,
    transverse_k,
    validate_stack,
)
from core.models import Layer


def test_validate_stack_rejects_too_few_layers():
    with pytest.raises(ValueError):
        validate_stack([Layer(n=1.44, thickness_nm=None), Layer(n=1.44, thickness_nm=None)])


def test_validate_stack_rejects_finite_outer_layers():
    layers = [Layer(n=1.44, thickness_nm=100), Layer(n=3.47, thickness_nm=220), Layer(n=1.44, thickness_nm=None)]
    with pytest.raises(ValueError):
        validate_stack(layers)


def test_validate_stack_rejects_inner_layer_without_thickness():
    layers = [Layer(n=1.44, thickness_nm=None), Layer(n=3.47, thickness_nm=None), Layer(n=1.44, thickness_nm=None)]
    with pytest.raises(ValueError):
        validate_stack(layers)


def test_validate_stack_rejects_non_positive_thickness():
    layers = [Layer(n=1.44, thickness_nm=None), Layer(n=3.47, thickness_nm=0), Layer(n=1.44, thickness_nm=None)]
    with pytest.raises(ValueError):
        validate_stack(layers)


def test_validate_stack_accepts_valid_stack():
    layers = [Layer(n=1.44, thickness_nm=None), Layer(n=3.47, thickness_nm=220), Layer(n=1.44, thickness_nm=None)]
    validate_stack(layers)  # nao deve levantar


def test_guided_index_bounds_asymmetric():
    layers = [Layer(n=1.44, thickness_nm=None), Layer(n=3.47, thickness_nm=220), Layer(n=1.46, thickness_nm=None)]
    n_lower, n_upper = guided_index_bounds(layers)
    assert n_lower == pytest.approx(1.46)
    assert n_upper == pytest.approx(3.47)


def test_transverse_k_propagating_is_real():
    kx = transverse_k(n=3.47, neff=2.0, k0=0.004)
    assert kx.imag == pytest.approx(0.0, abs=1e-12)
    assert kx.real > 0


def test_transverse_k_evanescent_is_imaginary():
    kx = transverse_k(n=1.44, neff=2.0, k0=0.004)
    assert kx.real == pytest.approx(0.0, abs=1e-12)
    assert kx.imag > 0


def test_coupling_factor_te_is_kx():
    kx = 0.001 + 0j
    assert coupling_factor(kx, n=3.0, polarization="TE") == kx


def test_coupling_factor_tm_scales_by_index_squared():
    kx = 0.001 + 0j
    n = 3.0
    assert coupling_factor(kx, n=n, polarization="TM") == pytest.approx(kx / n**2)


def test_boundary_kappa_matches_formula():
    n, neff, k0 = 1.44, 2.0, 0.004
    assert boundary_kappa(n, neff, k0) == pytest.approx(k0 * np.sqrt(neff**2 - n**2))


def test_boundary_slope_factor_te_is_one():
    assert boundary_slope_factor(n=3.0, polarization="TE") == 1.0


def test_boundary_slope_factor_tm_scales_by_index_squared():
    assert boundary_slope_factor(n=3.0, polarization="TM") == pytest.approx(1 / 9.0)
