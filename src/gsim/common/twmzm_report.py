"""One-call wiring from mode-solver outputs to TW-MZM figures of merit.

The mode solvers (Palace BoundaryMode or the femwell adapter) produce
complex effective indices and characteristic impedances versus frequency
and bias; the charge/optics side produces the effective-index shift along
the bias sweep. This module packages those into typed containers and one
entry point, :func:`twmzm_figures_of_merit`, that returns the full device
report: EO frequency response and 3 dB bandwidth, velocity mismatch and
the analytic walk-off limit, RLGC line parameters, and V_pi·L — all
computed with the pure analysis functions of :mod:`gsim.common.twmzm`.

Sign convention for complex effective indices is ``exp(+i omega t)``
(lossy: ``Im(n_eff) < 0``); the extraction accepts either sign of the
imaginary part and treats its magnitude as loss.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.common.twmzm import (
    eo_bandwidth,
    eo_response,
    rlgc_from_line_params,
    vpi_length_vcm,
    walkoff_bandwidth,
)

__all__ = [
    "OpticalPhaseSweep",
    "RFLineParams",
    "TWMZMReport",
    "line_params_from_neff",
    "twmzm_figures_of_merit",
]


class RFLineParams(BaseModel):
    """RF transmission-line parameters versus frequency at one bias.

    Attributes:
        freq_hz: RF frequencies in Hz (ascending).
        n_rf: RF effective (phase) index per frequency.
        alpha_rf_np_m: RF amplitude loss in Np/m per frequency.
        z0_ohm: Complex characteristic impedance in ohms per frequency.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    freq_hz: NDArray[np.float64]
    n_rf: NDArray[np.float64]
    alpha_rf_np_m: NDArray[np.float64]
    z0_ohm: NDArray[np.complex128]

    @model_validator(mode="after")
    def validate_shapes(self) -> Self:
        """All arrays share the frequency axis; frequencies are positive."""
        shape = self.freq_hz.shape
        for name in ("n_rf", "alpha_rf_np_m", "z0_ohm"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have the same shape as freq_hz.")
        if self.freq_hz.ndim != 1 or self.freq_hz.size == 0:
            raise ValueError("freq_hz must be a non-empty 1D array.")
        if np.any(self.freq_hz <= 0):
            raise ValueError("Frequencies must be positive.")
        return self

    @property
    def gamma_per_m(self) -> NDArray[np.complex128]:
        """Complex propagation constant ``alpha + j beta`` in 1/m."""
        omega = 2.0 * np.pi * self.freq_hz
        return np.asarray(
            self.alpha_rf_np_m + 1j * omega * self.n_rf / C0, dtype=np.complex128
        )

    @property
    def rlgc(self) -> dict[str, NDArray[np.float64]]:
        """RLGC per-unit-length parameters (Marks-Williams relations)."""
        return rlgc_from_line_params(
            self.freq_hz, gamma_per_m=self.gamma_per_m, z0_ohm=self.z0_ohm
        )


def line_params_from_neff(
    freq_hz: ArrayLike,
    n_eff: ArrayLike,
    *,
    z0_ohm: ArrayLike,
) -> RFLineParams:
    """Build :class:`RFLineParams` from complex mode effective indices.

    This is the extraction step shared by both solver routes: Palace
    BoundaryMode (``PalaceTextResults.modes[m]["n_eff"]`` per frequency)
    and the femwell adapter (``mode.n_eff``) both deliver a complex
    effective index; ``beta = omega Re(n_eff) / c0`` and
    ``alpha = omega |Im(n_eff)| / c0``.

    Args:
        freq_hz: RF frequencies in Hz.
        n_eff: Complex effective index per frequency (either sign
            convention for the imaginary part).
        z0_ohm: Characteristic impedance per frequency (complex allowed),
            e.g. from Palace's impedance postprocessing or a
            Marks-Williams extraction.

    Returns:
        The RF line parameters.
    """
    freq = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
    n_arr = np.broadcast_to(
        np.atleast_1d(np.asarray(n_eff, dtype=np.complex128)), freq.shape
    )
    z0 = np.broadcast_to(
        np.atleast_1d(np.asarray(z0_ohm, dtype=np.complex128)), freq.shape
    )
    omega = 2.0 * np.pi * freq
    return RFLineParams(
        freq_hz=freq,
        n_rf=np.asarray(n_arr.real, dtype=np.float64),
        alpha_rf_np_m=np.asarray(np.abs(n_arr.imag) * omega / C0, dtype=np.float64),
        z0_ohm=np.asarray(z0, dtype=np.complex128),
    )


class OpticalPhaseSweep(BaseModel):
    """Optical response of the phase shifter along a bias sweep.

    Attributes:
        voltages_v: Bias voltages in volts (>= 2 points).
        dn_eff: Effective-index shift at each bias.
        alpha_opt_db_cm: Optional optical loss in dB/cm at each bias.
        wavelength_um: Vacuum wavelength in um.
        n_group: Optical group index used for the velocity-mismatch
            analysis.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    voltages_v: NDArray[np.float64]
    dn_eff: NDArray[np.float64]
    alpha_opt_db_cm: NDArray[np.float64] | None = None
    wavelength_um: float = Field(gt=0.0)
    n_group: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_shapes(self) -> Self:
        """Bias sweep arrays share one axis with at least two points."""
        if self.voltages_v.ndim != 1 or self.voltages_v.size < 2:
            raise ValueError("voltages_v must be 1D with >= 2 points.")
        if self.dn_eff.shape != self.voltages_v.shape:
            raise ValueError("dn_eff must have the same shape as voltages_v.")
        if (
            self.alpha_opt_db_cm is not None
            and self.alpha_opt_db_cm.shape != self.voltages_v.shape
        ):
            raise ValueError("alpha_opt_db_cm must match voltages_v.")
        return self


class TWMZMReport(BaseModel):
    """Full traveling-wave modulator device report.

    Attributes:
        freq_hz: RF frequencies of the response (Hz).
        response: Normalized complex EO response per frequency.
        bandwidth_3db_hz: 3 dB EO bandwidth, or None when the response
            stays above the threshold over the swept range.
        walkoff_bandwidth_hz: Analytic walk-off-limited bandwidth of a
            lossless matched line, or None when velocity matched.
        velocity_mismatch: ``n_rf - n_group`` per frequency.
        z0_ohm: Characteristic impedance per frequency.
        z_load_ohm: Termination impedance used.
        z_gen_ohm: Generator impedance used.
        rlgc: RLGC per-unit-length parameters per frequency.
        vpi_l_vcm: V_pi·L in V*cm at each bias point.
        voltages_v: Bias voltages of the V_pi·L sweep.
        length_m: Electrode length (m).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    freq_hz: NDArray[np.float64]
    response: NDArray[np.complex128]
    bandwidth_3db_hz: float | None
    walkoff_bandwidth_hz: float | None
    velocity_mismatch: NDArray[np.float64]
    z0_ohm: NDArray[np.complex128]
    z_load_ohm: complex
    z_gen_ohm: complex
    rlgc: dict[str, NDArray[np.float64]]
    vpi_l_vcm: NDArray[np.float64]
    voltages_v: NDArray[np.float64]
    length_m: float


def twmzm_figures_of_merit(
    rf: RFLineParams,
    optical: OpticalPhaseSweep,
    *,
    length_m: float,
    z_load_ohm: complex = 50.0,
    z_gen_ohm: complex = 50.0,
) -> TWMZMReport:
    """Combine RF line parameters and the optical sweep into the report.

    Args:
        rf: RF line parameters versus frequency (one bias point).
        optical: Optical phase response along the bias sweep.
        length_m: Electrode length in meters (> 0).
        z_load_ohm: Termination impedance in ohms.
        z_gen_ohm: Generator impedance in ohms.

    Returns:
        The assembled :class:`TWMZMReport`.
    """
    if length_m <= 0:
        raise ValueError("length_m must be positive.")

    response = eo_response(
        rf.freq_hz,
        length_m=length_m,
        n_rf=rf.n_rf,
        n_opt=optical.n_group,
        alpha_rf_np_m=rf.alpha_rf_np_m,
        z0_ohm=rf.z0_ohm,
        z_load_ohm=z_load_ohm,
        z_gen_ohm=z_gen_ohm,
    )
    bandwidth = eo_bandwidth(rf.freq_hz, response)

    n_rf_repr = float(np.mean(rf.n_rf))
    if n_rf_repr == optical.n_group:
        walkoff: float | None = None
    else:
        walkoff = walkoff_bandwidth(
            length_m=length_m, n_rf=n_rf_repr, n_opt=optical.n_group
        )

    return TWMZMReport(
        freq_hz=rf.freq_hz,
        response=np.asarray(response, dtype=np.complex128),
        bandwidth_3db_hz=bandwidth,
        walkoff_bandwidth_hz=walkoff,
        velocity_mismatch=np.asarray(rf.n_rf - optical.n_group, dtype=np.float64),
        z0_ohm=rf.z0_ohm,
        z_load_ohm=complex(z_load_ohm),
        z_gen_ohm=complex(z_gen_ohm),
        rlgc=rf.rlgc,
        vpi_l_vcm=vpi_length_vcm(
            optical.voltages_v,
            optical.dn_eff,
            wavelength_um=optical.wavelength_um,
        ),
        voltages_v=optical.voltages_v,
        length_m=length_m,
    )
