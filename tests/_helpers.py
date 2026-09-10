"""Analytic fixtures shared across test packages."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def series_rc_admittance(
    freq_hz: ArrayLike, r_ohm_m: float | ArrayLike, c_f_per_m: float | ArrayLike
) -> NDArray[np.complex128]:
    """Admittance of a series-RC branch, straight from Z = R + 1/(jwC)."""
    omega = 2 * np.pi * np.asarray(freq_hz, dtype=np.float64)
    return np.asarray(
        1.0 / (np.asarray(r_ohm_m) + 1.0 / (1j * omega * np.asarray(c_f_per_m))),
        dtype=np.complex128,
    )
