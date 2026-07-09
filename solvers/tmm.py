import numpy as np
from scipy.optimize import brentq

from core.materials import (
    boundary_kappa,
    boundary_slope_factor,
    coupling_factor,
    guided_index_bounds,
    transverse_k,
    validate_stack,
)
from core.models import Layer, ModeResult, Problem
from core.postprocess import reconstruct_field
from solvers import BaseSolver


class TMMSolver(BaseSolver):
    def solve(self, problem: Problem) -> list[ModeResult]:
        validate_stack(problem.layers)
        polarization = problem.polarization.upper()
        if polarization not in ("TE", "TM"):
            raise ValueError(f"Polarização inválida: {problem.polarization}")

        k0 = 2 * np.pi / problem.wavelength_nm  # 1/nm
        wavelength_m = problem.wavelength_nm * 1e-9
        n_lower, n_upper = guided_index_bounds(problem.layers)

        def characteristic(neff: float) -> float:
            return _boundary_residual(neff, problem.layers, k0, polarization)

        neffs = _scan_and_refine(characteristic, n_lower, n_upper)

        results = []
        for mode_index, neff in enumerate(sorted(neffs, reverse=True)):
            beta = 2 * np.pi * neff / wavelength_m  # rad/m
            field = reconstruct_field(problem, neff)
            results.append(ModeResult(neff=neff, beta=beta, mode_index=mode_index, field=field))
        return results


def _boundary_residual(neff: float, layers: list[Layer], k0: float, polarization: str) -> float:
    n0, n_last = layers[0].n, layers[-1].n
    kappa0 = boundary_kappa(n0, neff, k0)
    kappa_last = boundary_kappa(n_last, neff, k0)
    rho0 = boundary_slope_factor(n0, polarization)
    rho_last = boundary_slope_factor(n_last, polarization)

    state = np.array([1.0, kappa0 * rho0], dtype=complex)

    for layer in layers[1:-1]:
        kx = transverse_k(layer.n, neff, k0)
        gamma = coupling_factor(kx, layer.n, polarization)
        theta = kx * layer.thickness_nm

        if abs(gamma) < 1e-12:
            propagation = np.array([[1.0, layer.thickness_nm], [0.0, 1.0]], dtype=complex)
        else:
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            propagation = np.array([[cos_t, sin_t / gamma], [-gamma * sin_t, cos_t]], dtype=complex)

        state = propagation @ state

    u_end, v_end = state
    residual = v_end + kappa_last * rho_last * u_end
    return residual.real


def _scan_and_refine(f, lo: float, hi: float, n_samples: int = 4000, tol: float = 1e-9) -> list[float]:
    margin = (hi - lo) * 1e-6
    if margin <= 0:
        return []

    xs = np.linspace(lo + margin, hi - margin, n_samples)
    with np.errstate(divide="ignore", invalid="ignore"):
        ys = np.array([f(x) for x in xs])

    roots = []
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if y0 == 0.0:
            roots.append(xs[i])
        elif y0 * y1 < 0:
            roots.append(brentq(f, xs[i], xs[i + 1], xtol=tol))

    unique_roots = []
    for root in sorted(roots):
        if not unique_roots or abs(root - unique_roots[-1]) > 1e-6:
            unique_roots.append(root)
    return unique_roots
