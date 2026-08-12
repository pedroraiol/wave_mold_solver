import numpy as np
from scipy.linalg import eigh

from core.materials import guided_index_bounds, validate_stack
from core.models import FieldProfile, ModeResult, Problem
from solvers import BaseSolver

# Padding das claddings (semi-infinitas) truncadas em caixa finita, em
# múltiplos do comprimento de onda, com piso mínimo em nm.
PADDING_FACTOR = 6.0
PADDING_FLOOR_NM = 300.0

# Resolução da malha: pelo menos esse número de elementos na camada interna
# mais fina; nas claddings (padding), passo baseado em wavelength/divisor
# (só precisam resolver o decaimento evanescente, não o núcleo).
POINTS_PER_LAYER_MIN = 60
CLADDING_DX_DIVISOR = 20.0
MESH_DX_MIN_NM = 0.2
MESH_DX_MAX_NM = 10.0


class FEMSolver(BaseSolver):
    def solve(self, problem: Problem) -> list[ModeResult]:
        validate_stack(problem.layers)
        polarization = problem.polarization.upper()
        if polarization not in ("TE", "TM"):
            raise ValueError(f"Polarização inválida: {problem.polarization}")

        k0 = 2 * np.pi / problem.wavelength_nm  # 1/nm
        wavelength_m = problem.wavelength_nm * 1e-9
        n_lower, n_upper = guided_index_bounds(problem.layers)

        regions = _mesh_regions(problem)
        x_nm, n_elem = _build_mesh(regions)
        K, Kp, M, Mn, Mp = _assemble(x_nm, n_elem)

        if polarization == "TE":
            A, B = k0**2 * Mn - K, M
        else:
            A, B = k0**2 * M - Kp, Mp

        # Dirichlet: psi=0 nas bordas da caixa -> descarta 1o/ultimo grau de liberdade
        A_int, B_int = A[1:-1, 1:-1], B[1:-1, 1:-1]
        eigvals, eigvecs = eigh(A_int, B_int)

        margin = (n_upper - n_lower) * 1e-6
        candidates = []
        for w, vec in zip(eigvals, eigvecs.T):
            if w <= 0:
                continue
            neff = np.sqrt(w) / k0
            if n_lower + margin < neff < n_upper - margin:
                candidates.append((neff, vec))
        candidates.sort(key=lambda item: item[0], reverse=True)

        results = []
        for mode_index, (neff, vec_int) in enumerate(candidates):
            beta = 2 * np.pi * neff / wavelength_m  # rad/m
            field = _build_field(x_nm, vec_int)
            results.append(ModeResult(neff=neff, beta=beta, mode_index=mode_index, field=field))
        return results


def _mesh_regions(problem: Problem) -> list[tuple[float, float, float, float]]:
    """Lista de regiões (start_nm, stop_nm, n, dx_nm) em ordem crescente de x."""
    layers = problem.layers
    inner = layers[1:-1]
    thicknesses = [layer.thickness_nm for layer in inner]
    total_inner = sum(thicknesses)
    offsets = np.concatenate(([0.0], np.cumsum(thicknesses)))

    wavelength = problem.wavelength_nm
    padding = max(PADDING_FACTOR * wavelength, PADDING_FLOOR_NM)
    dx_clad = min(max(wavelength / CLADDING_DX_DIVISOR, MESH_DX_MIN_NM), MESH_DX_MAX_NM)

    regions = [(-padding, 0.0, layers[0].n, dx_clad)]
    for layer, d, start in zip(inner, thicknesses, offsets[:-1]):
        dx_layer = min(max(d / POINTS_PER_LAYER_MIN, MESH_DX_MIN_NM), MESH_DX_MAX_NM)
        regions.append((start, start + d, layer.n, dx_layer))
    regions.append((total_inner, total_inner + padding, layers[-1].n, dx_clad))
    return regions


def _build_mesh(regions: list[tuple[float, float, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Constrói nós globais (x_nm) e o índice de refração por elemento (n_elem).

    Cada região gera seus próprios nós com linspace inclusivo; o nó de
    fronteira compartilhado com a região anterior é descartado antes de
    concatenar, garantindo que nenhum elemento atravesse duas regiões com
    índices de refração diferentes.
    """
    xs_parts = []
    n_elem_parts = []
    for i, (start, stop, n, dx) in enumerate(regions):
        n_points = max(2, round((stop - start) / dx) + 1)
        nodes = np.linspace(start, stop, n_points)
        xs_parts.append(nodes if i == 0 else nodes[1:])
        n_elem_parts.append(np.full(n_points - 1, n))

    x_nm = np.concatenate(xs_parts)
    n_elem = np.concatenate(n_elem_parts)
    return x_nm, n_elem


def _assemble(x_nm: np.ndarray, n_elem: np.ndarray) -> tuple[np.ndarray, ...]:
    """Monta as matrizes tridiagonais de rigidez/massa (elementos P1 lineares)."""
    h = np.diff(x_nm)
    n_nodes = len(x_nm)
    n_elements = len(h)
    n_sq = n_elem**2
    inv_n_sq = 1.0 / n_sq

    def tridiag(diag_contrib: np.ndarray, off_contrib: np.ndarray) -> np.ndarray:
        main = np.zeros(n_nodes)
        np.add.at(main, np.arange(n_elements), diag_contrib)
        np.add.at(main, np.arange(1, n_elements + 1), diag_contrib)
        return np.diag(main) + np.diag(off_contrib, 1) + np.diag(off_contrib, -1)

    K = tridiag(1.0 / h, -1.0 / h)
    Kp = tridiag(inv_n_sq / h, -inv_n_sq / h)
    M = tridiag(h / 3.0, h / 6.0)
    Mn = tridiag(n_sq * h / 3.0, n_sq * h / 6.0)
    Mp = tridiag(inv_n_sq * h / 3.0, inv_n_sq * h / 6.0)
    return K, Kp, M, Mn, Mp


def _build_field(x_nm: np.ndarray, vec_int: np.ndarray) -> FieldProfile:
    full_vec = np.zeros(len(x_nm))
    full_vec[1:-1] = vec_int

    peak_idx = np.argmax(np.abs(full_vec))
    if full_vec[peak_idx] < 0:
        full_vec = -full_vec
    peak = np.abs(full_vec[peak_idx])
    amplitude = full_vec / peak if peak > 0 else full_vec

    return FieldProfile(x_nm=x_nm, amplitude=amplitude)
