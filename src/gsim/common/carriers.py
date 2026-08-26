"""Carrier-map to material-response coupling (solver-agnostic, pure functions).

This module converts charge-transport results — free-carrier concentrations
n(x, y), p(x, y) in cm^-3 — into the material responses the EM solvers
consume:

- RF: Drude conductivity ``sigma = q (mu_n n + mu_p p)`` in S/m
  (:func:`carrier_conductivity`).
- Optics: refractive-index shift ``Delta n`` and free-carrier absorption
  ``Delta alpha`` through the plasma-dispersion power-law fits of
  Soref-Bennett (1987) and Nedeljkovic-Soref-Mashanovich (2011)
  (:class:`PlasmaDispersionModel`, :func:`carrier_index_shift`,
  :func:`carrier_absorption_cm`), plus the complex permittivity they imply
  (:func:`permittivity_perturbation`).
- Staircase discretization: binning a continuously varying 1D carrier
  profile into N piecewise-constant strips for solvers that only accept
  piecewise-constant materials, e.g. Palace (:func:`staircase_profile`).

All coefficients are explicit and overridable so foundry-calibrated values
can be substituted for the published fits.

References:
    R. Soref and B. Bennett, "Electrooptical effects in silicon,"
    IEEE J. Quantum Electron. 23, 123-129 (1987).

    M. Nedeljkovic, R. Soref, and G. Z. Mashanovich, "Free-Carrier
    Electrorefraction and Electroabsorption Modulation Predictions for
    Silicon Over the 1-14 um Infrared Wavelength Range,"
    IEEE Photonics J. 3, 1171-1180 (2011).
"""

from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.constants import elementary_charge as Q  # noqa: N812

__all__ = [
    "DEFAULT_MU_N_CM2",
    "DEFAULT_MU_P_CM2",
    "PlasmaDispersionModel",
    "carrier_absorption_cm",
    "carrier_conductivity",
    "carrier_index_shift",
    "permittivity_perturbation",
    "staircase_profile",
]

#: Low-field electron mobility of lightly doped silicon at 300 K (cm^2/Vs).
DEFAULT_MU_N_CM2: float = 1417.0

#: Low-field hole mobility of lightly doped silicon at 300 K (cm^2/Vs).
DEFAULT_MU_P_CM2: float = 470.5


class PlasmaDispersionModel(BaseModel):
    """Power-law plasma-dispersion coefficients at one wavelength.

    The model evaluates::

        Delta n     = -(a_n N^b_n + a_p P^b_p)
        Delta alpha =   c_n N^d_n + c_p P^d_p      [cm^-1]

    with N, P the electron/hole concentrations in cm^-3. The presets carry
    the published silicon fits; construct the model directly (or
    ``model_copy(update=...)`` a preset) to substitute foundry-calibrated
    coefficients.

    Attributes:
        wavelength_um: Wavelength the coefficients are valid at (um).
        dn_electron_coeff: ``a_n`` in the Delta-n electron term.
        dn_electron_exp: ``b_n`` exponent of the Delta-n electron term.
        dn_hole_coeff: ``a_p`` in the Delta-n hole term.
        dn_hole_exp: ``b_p`` exponent of the Delta-n hole term.
        dalpha_electron_coeff: ``c_n`` in the Delta-alpha electron term.
        dalpha_electron_exp: ``d_n`` exponent of the Delta-alpha electron term.
        dalpha_hole_coeff: ``c_p`` in the Delta-alpha hole term.
        dalpha_hole_exp: ``d_p`` exponent of the Delta-alpha hole term.
    """

    model_config = ConfigDict(validate_assignment=True)

    wavelength_um: float = Field(gt=0, description="Validity wavelength (um)")
    dn_electron_coeff: float = Field(ge=0)
    dn_electron_exp: float = Field(gt=0)
    dn_hole_coeff: float = Field(ge=0)
    dn_hole_exp: float = Field(gt=0)
    dalpha_electron_coeff: float = Field(ge=0)
    dalpha_electron_exp: float = Field(gt=0)
    dalpha_hole_coeff: float = Field(ge=0)
    dalpha_hole_exp: float = Field(gt=0)

    @classmethod
    def nedeljkovic_1550(cls) -> PlasmaDispersionModel:
        """Nedeljkovic et al. (2011) power-law fit at 1.55 um."""
        return cls(
            wavelength_um=1.55,
            dn_electron_coeff=5.4e-22,
            dn_electron_exp=1.011,
            dn_hole_coeff=1.53e-18,
            dn_hole_exp=0.838,
            dalpha_electron_coeff=8.88e-21,
            dalpha_electron_exp=1.167,
            dalpha_hole_coeff=5.84e-20,
            dalpha_hole_exp=1.109,
        )

    @classmethod
    def nedeljkovic_1310(cls) -> PlasmaDispersionModel:
        """Nedeljkovic et al. (2011) power-law fit at 1.31 um."""
        return cls(
            wavelength_um=1.31,
            dn_electron_coeff=2.98e-22,
            dn_electron_exp=1.016,
            dn_hole_coeff=1.25e-18,
            dn_hole_exp=0.835,
            dalpha_electron_coeff=3.48e-22,
            dalpha_electron_exp=1.229,
            dalpha_hole_coeff=1.02e-19,
            dalpha_hole_exp=1.089,
        )

    @classmethod
    def soref_1550(cls) -> PlasmaDispersionModel:
        """Soref-Bennett (1987) linearized fit at 1.55 um."""
        return cls(
            wavelength_um=1.55,
            dn_electron_coeff=8.8e-22,
            dn_electron_exp=1.0,
            dn_hole_coeff=8.5e-18,
            dn_hole_exp=0.8,
            dalpha_electron_coeff=8.5e-18,
            dalpha_electron_exp=1.0,
            dalpha_hole_coeff=6.0e-18,
            dalpha_hole_exp=1.0,
        )


def _validated_carriers(
    n_cm3: ArrayLike, p_cm3: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert carrier inputs to arrays, rejecting negative concentrations."""
    n = np.asarray(n_cm3, dtype=np.float64)
    p = np.asarray(p_cm3, dtype=np.float64)
    if np.any(n < 0) or np.any(p < 0):
        raise ValueError("Carrier concentrations must be non-negative (cm^-3).")
    return n, p


def _power_term(
    x: NDArray[np.float64], coeff: float, exp: float
) -> NDArray[np.float64]:
    """Evaluate ``coeff * x**exp`` with 0**exp = 0 (no 0-division warnings)."""
    out = np.zeros_like(x)
    mask = x > 0
    out[mask] = coeff * x[mask] ** exp
    return out


@overload
def carrier_index_shift(
    n_cm3: NDArray[np.floating],
    p_cm3: ArrayLike,
    *,
    model: PlasmaDispersionModel,
) -> NDArray[np.float64]: ...
@overload
def carrier_index_shift(
    n_cm3: float,
    p_cm3: float,
    *,
    model: PlasmaDispersionModel,
) -> float: ...
def carrier_index_shift(
    n_cm3: ArrayLike,
    p_cm3: ArrayLike,
    *,
    model: PlasmaDispersionModel,
) -> NDArray[np.float64] | float:
    """Refractive-index shift ``Delta n`` from free carriers.

    Args:
        n_cm3: Electron concentration(s) in cm^-3 (scalar or array).
        p_cm3: Hole concentration(s) in cm^-3 (scalar or array).
        model: Plasma-dispersion coefficients at the target wavelength.

    Returns:
        ``Delta n`` (negative for positive carrier densities); scalar in,
        scalar out.
    """
    n, p = _validated_carriers(n_cm3, p_cm3)
    dn = np.asarray(
        -(
            _power_term(n, model.dn_electron_coeff, model.dn_electron_exp)
            + _power_term(p, model.dn_hole_coeff, model.dn_hole_exp)
        ),
        dtype=np.float64,
    )
    return dn if dn.ndim else float(dn)


@overload
def carrier_absorption_cm(
    n_cm3: NDArray[np.floating],
    p_cm3: ArrayLike,
    *,
    model: PlasmaDispersionModel,
) -> NDArray[np.float64]: ...
@overload
def carrier_absorption_cm(
    n_cm3: float,
    p_cm3: float,
    *,
    model: PlasmaDispersionModel,
) -> float: ...
def carrier_absorption_cm(
    n_cm3: ArrayLike,
    p_cm3: ArrayLike,
    *,
    model: PlasmaDispersionModel,
) -> NDArray[np.float64] | float:
    """Free-carrier absorption ``Delta alpha`` in cm^-1.

    Args:
        n_cm3: Electron concentration(s) in cm^-3 (scalar or array).
        p_cm3: Hole concentration(s) in cm^-3 (scalar or array).
        model: Plasma-dispersion coefficients at the target wavelength.

    Returns:
        ``Delta alpha`` in cm^-1 (non-negative); scalar in, scalar out.
    """
    n, p = _validated_carriers(n_cm3, p_cm3)
    dalpha = np.asarray(
        _power_term(n, model.dalpha_electron_coeff, model.dalpha_electron_exp)
        + _power_term(p, model.dalpha_hole_coeff, model.dalpha_hole_exp),
        dtype=np.float64,
    )
    return dalpha if dalpha.ndim else float(dalpha)


@overload
def carrier_conductivity(
    n_cm3: NDArray[np.floating],
    p_cm3: ArrayLike,
    *,
    mu_n_cm2: float = ...,
    mu_p_cm2: float = ...,
) -> NDArray[np.float64]: ...
@overload
def carrier_conductivity(
    n_cm3: float,
    p_cm3: float,
    *,
    mu_n_cm2: float = ...,
    mu_p_cm2: float = ...,
) -> float: ...
def carrier_conductivity(
    n_cm3: ArrayLike,
    p_cm3: ArrayLike,
    *,
    mu_n_cm2: float = DEFAULT_MU_N_CM2,
    mu_p_cm2: float = DEFAULT_MU_P_CM2,
) -> NDArray[np.float64] | float:
    """Drude conductivity ``sigma = q (mu_n n + mu_p p)`` in S/m.

    Args:
        n_cm3: Electron concentration(s) in cm^-3 (scalar or array).
        p_cm3: Hole concentration(s) in cm^-3 (scalar or array).
        mu_n_cm2: Electron mobility in cm^2/Vs.
        mu_p_cm2: Hole mobility in cm^2/Vs.

    Returns:
        Conductivity in S/m; scalar in, scalar out.
    """
    if mu_n_cm2 < 0 or mu_p_cm2 < 0:
        raise ValueError("Mobilities must be non-negative (cm^2/Vs).")
    n, p = _validated_carriers(n_cm3, p_cm3)
    # q [C] * mu [cm^2/Vs] * n [cm^-3] = sigma [S/cm]; * 100 -> S/m.
    sigma = Q * (mu_n_cm2 * n + mu_p_cm2 * p) * 100.0
    return sigma if sigma.ndim else float(sigma)


def permittivity_perturbation(
    *,
    n0: float,
    dn: float,
    dalpha_cm: float,
    wavelength_um: float,
) -> complex:
    """Complex relative permittivity of a carrier-perturbed dielectric.

    Builds ``eps = (n0 + dn - i kappa)^2`` with the extinction coefficient
    ``kappa = alpha * lambda / (4 pi)`` from the absorption change, using the
    ``exp(+i omega t)`` convention (lossy medium: ``Im(eps) < 0``).

    Args:
        n0: Unperturbed refractive index.
        dn: Carrier-induced index shift (from :func:`carrier_index_shift`).
        dalpha_cm: Carrier-induced absorption in cm^-1 (non-negative).
        wavelength_um: Vacuum wavelength in um.

    Returns:
        Complex relative permittivity.
    """
    if n0 <= 0:
        raise ValueError("n0 must be positive.")
    if dalpha_cm < 0:
        raise ValueError("dalpha_cm must be non-negative.")
    if wavelength_um <= 0:
        raise ValueError("wavelength_um must be positive.")
    alpha_m = dalpha_cm * 100.0
    kappa = alpha_m * wavelength_um * 1e-6 / (4.0 * np.pi)
    n_complex = (n0 + dn) - 1j * kappa
    return complex(n_complex * n_complex)


def staircase_profile(
    h: ArrayLike,
    values: ArrayLike,
    *,
    n_bins: int,
    h_min: float | None = None,
    h_max: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bin a sampled 1D profile into N equal-width piecewise-constant strips.

    The samples are interpreted as a piecewise-linear function of ``h``; each
    strip value is the exact average of that interpolant over the strip, so
    ``n_bins=1`` recovers the exact mean of the profile and increasing
    ``n_bins`` converges to the continuous profile.

    Args:
        h: Sample coordinates along the binning axis (um), any order.
        values: Sample values (e.g. carrier concentration, sigma, Delta n).
        n_bins: Number of strips (>= 1).
        h_min: Window start; defaults to ``min(h)``.
        h_max: Window end; defaults to ``max(h)``.

    Returns:
        ``(edges, means)`` — strip edges of length ``n_bins + 1`` and the
        per-strip averages of length ``n_bins``.
    """
    h_arr = np.asarray(h, dtype=np.float64).ravel()
    v_arr = np.asarray(values, dtype=np.float64).ravel()
    if h_arr.size != v_arr.size:
        raise ValueError("h and values must have the same length.")
    if h_arr.size < 2:
        raise ValueError("At least two samples are required.")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")

    order = np.argsort(h_arr)
    h_arr = h_arr[order]
    v_arr = v_arr[order]

    lo = float(h_arr[0]) if h_min is None else float(h_min)
    hi = float(h_arr[-1]) if h_max is None else float(h_max)
    if hi <= lo:
        raise ValueError("h_max must exceed h_min.")

    edges = np.asarray(np.linspace(lo, hi, n_bins + 1), dtype=np.float64)

    # Exact average of the piecewise-linear interpolant over each strip via
    # its antiderivative sampled with cumulative trapezoids.
    dense = np.union1d(edges, h_arr[(h_arr > lo) & (h_arr < hi)])
    dense_v = np.interp(dense, h_arr, v_arr)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (dense_v[1:] + dense_v[:-1]) * np.diff(dense)))
    )
    edge_integrals = np.interp(edges, dense, cumulative)
    means = np.asarray(np.diff(edge_integrals) / np.diff(edges), dtype=np.float64)
    return edges, means
