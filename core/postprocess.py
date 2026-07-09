from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.materials import boundary_kappa, boundary_slope_factor, coupling_factor, transverse_k
from core.models import FieldProfile, ModeResult, Problem


def reconstruct_field(problem: Problem, neff: float, points_per_layer: int = 200) -> FieldProfile:
    layers = problem.layers
    inner = layers[1:-1]
    k0 = 2 * np.pi / problem.wavelength_nm
    polarization = problem.polarization.upper()

    thicknesses = [layer.thickness_nm for layer in inner]
    total_inner = sum(thicknesses)
    tail = max(total_inner, 200.0)
    offsets = np.concatenate(([0.0], np.cumsum(thicknesses)))

    n0, n_last = layers[0].n, layers[-1].n
    kappa0 = boundary_kappa(n0, neff, k0)
    kappa_last = boundary_kappa(n_last, neff, k0)
    rho0 = boundary_slope_factor(n0, polarization)
    rho_last = boundary_slope_factor(n_last, polarization)

    xs_all, us_all = [], []

    x_left = np.linspace(-tail, 0.0, points_per_layer, endpoint=False)
    xs_all.append(x_left)
    us_all.append(np.exp(kappa0 * x_left))

    state = np.array([1.0, kappa0 * rho0], dtype=complex)
    for layer, d, x_start in zip(inner, thicknesses, offsets[:-1]):
        kx = transverse_k(layer.n, neff, k0)
        gamma = coupling_factor(kx, layer.n, polarization)
        xs_local = np.linspace(0.0, d, points_per_layer, endpoint=False)

        if abs(gamma) < 1e-12:
            u_local = state[0] + state[1] * xs_local
            u_end = state[0] + state[1] * d
            v_end = state[1]
        else:
            theta_local = kx * xs_local
            u_local = state[0] * np.cos(theta_local) + state[1] * np.sin(theta_local) / gamma
            theta = kx * d
            u_end = state[0] * np.cos(theta) + state[1] * np.sin(theta) / gamma
            v_end = -gamma * state[0] * np.sin(theta) + state[1] * np.cos(theta)

        xs_all.append(x_start + xs_local)
        us_all.append(u_local.real)
        state = np.array([u_end, v_end], dtype=complex)

    x_right = np.linspace(0.0, tail, points_per_layer)
    xs_all.append(total_inner + x_right)
    us_all.append(state[0].real * np.exp(-kappa_last * x_right))

    x_nm = np.concatenate(xs_all)
    amplitude = np.concatenate(us_all)
    peak = np.max(np.abs(amplitude))
    if peak > 0:
        amplitude = amplitude / peak

    return FieldProfile(x_nm=x_nm, amplitude=amplitude)


def save_mode_plot(problem: Problem, results: list[ModeResult], out_dir: str = "outputs") -> list[Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    inner = problem.layers[1:-1]
    thicknesses = [layer.thickness_nm for layer in inner]
    offsets = np.concatenate(([0.0], np.cumsum(thicknesses)))
    for start, d in zip(offsets[:-1], thicknesses):
        ax.axvspan(start, start + d, color="0.85", zorder=0, label="core")

    field_label = "Ey" if problem.polarization.upper() == "TE" else "Hy"
    for result in results:
        if result.field is None:
            continue
        ax.plot(
            result.field.x_nm,
            result.field.amplitude,
            label=f"Modo {result.mode_index} (neff={result.neff:.4f})",
        )

    ax.set_xlabel("x (nm)")
    ax.set_ylabel(f"{field_label} normalizado")
    ax.set_title(f"Perfil de modo - {problem.polarization.upper()} @ {problem.wavelength_nm} nm")
    ax.legend()
    fig.tight_layout()

    filename = out_path / f"modes_{problem.polarization.upper()}_{int(problem.wavelength_nm)}nm.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    return [filename]
