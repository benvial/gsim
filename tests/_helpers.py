"""Analytic fixtures shared across test packages."""

from __future__ import annotations

from types import SimpleNamespace

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


def fake_coupling(n_cm3: ArrayLike, p_cm3: ArrayLike) -> SimpleNamespace:
    """A stand-in carrier coupling: linear in the concentrations, no physics.

    Shaped like the carriers Stage's response — index shift, absorption
    and conductivity per sample — so a Staircase test can inject it in
    place of a plasma-dispersion model. Electrons conduct twice as well
    as holes, so a profile's asymmetry survives into the conductivity.
    """
    n = np.asarray(n_cm3, dtype=np.float64)
    p = np.asarray(p_cm3, dtype=np.float64)
    return SimpleNamespace(
        index_shift=-1e-20 * (n + p),
        absorption_cm=1e-17 * (n + p),
        conductivity_s_per_m=2e-15 * n + 1e-15 * p,
    )
