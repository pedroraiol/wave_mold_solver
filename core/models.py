from dataclasses import dataclass

import numpy as np


@dataclass
class Layer:
    n: float
    thickness_nm: float | None = None  # None = semi-infinite


@dataclass
class FieldProfile:
    x_nm: np.ndarray
    amplitude: np.ndarray  # normalized transverse field component (Ey for TE, Hy for TM)


@dataclass
class ModeResult:
    neff: float
    beta: float  # rad/m
    mode_index: int  # 0 = fundamental, 1 = first order, ...
    field: FieldProfile | None = None


@dataclass
class Problem:
    wavelength_nm: float
    polarization: str
    method: str
    layers: list[Layer]
