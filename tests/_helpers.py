import numpy as np


def symmetric_slab_mode_count(n_core: float, n_clad: float, thickness_nm: float, wavelength_nm: float) -> int:
    """Numero de modos TE/TM guiados em um slab simetrico via V-number.

    V = k0 * (d/2) * sqrt(n_core^2 - n_clad^2); o modo m (m=0,1,2,...) e guiado
    se V > m*pi/2 (mesmo limiar de corte para TE e TM em slab simetrico, ja
    que a diferenca entre polarizacoes desaparece exatamente no corte).
    """
    k0 = 2 * np.pi / wavelength_nm
    na = np.sqrt(n_core**2 - n_clad**2)
    v_number = k0 * (thickness_nm / 2) * na
    return int(np.floor(2 * v_number / np.pi)) + 1
