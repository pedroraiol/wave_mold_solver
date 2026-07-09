import numpy as np

from core.models import Layer


def validate_stack(layers: list[Layer]) -> None:
    if len(layers) < 3:
        raise ValueError("A pilha precisa de ao menos 3 camadas (clad, core, clad)")

    if layers[0].thickness_nm is not None or layers[-1].thickness_nm is not None:
        raise ValueError("A primeira e a última camada devem ser semi-infinitas (thickness_nm=None)")

    for layer in layers[1:-1]:
        if layer.thickness_nm is None:
            raise ValueError("Camadas internas precisam ter thickness_nm definido")
        if layer.thickness_nm <= 0:
            raise ValueError("thickness_nm das camadas internas deve ser positivo")


def guided_index_bounds(layers: list[Layer]) -> tuple[float, float]:
    n_lower = max(layers[0].n, layers[-1].n)
    n_upper = max(layer.n for layer in layers)
    return n_lower, n_upper


def transverse_k(n: float, neff: float, k0: float) -> complex:
    """kx da camada interna (complexo: real se propagante, imaginário se evanescente)."""
    return k0 * np.sqrt(n**2 - neff**2 + 0j)


def coupling_factor(kx: complex, n: float, polarization: str) -> complex:
    """Fator de acoplamento gamma usado na matriz de propagação: kx (TE) ou kx/n^2 (TM)."""
    if polarization == "TE":
        return kx
    return kx / n**2


def boundary_kappa(n: float, neff: float, k0: float) -> float:
    """Constante de decaimento (real, positiva) na camada semi-infinita de índice n."""
    return k0 * np.sqrt(neff**2 - n**2)


def boundary_slope_factor(n: float, polarization: str) -> float:
    """Fator rho aplicado ao kappa da camada semi-infinita: 1 (TE) ou 1/n^2 (TM)."""
    if polarization == "TE":
        return 1.0
    return 1.0 / n**2
