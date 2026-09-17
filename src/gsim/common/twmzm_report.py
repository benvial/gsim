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
    "LoadedLineComparison",
    "OpticalPhaseSweep",
    "RFLineParams",
    "TWMZMReport",
    "line_params_from_gamma",
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
        unloaded: Whether these are the bare electrode's parameters —
            the cross-section solved with every carrier switched off —
            rather than a Bias point's answer.
        bias_v: The Bias the Cross-section was built at (V), when it was
            built from a Bias point; ``None`` for parameters that came
            from nowhere in particular (a hand-assembled line).
        signal_contact: Name of the Contact the RF drive is applied to,
            which names the conductor the impedance was read over;
            ``None`` when no Contact was involved.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    freq_hz: NDArray[np.float64]
    n_rf: NDArray[np.float64]
    alpha_rf_np_m: NDArray[np.float64]
    z0_ohm: NDArray[np.complex128]
    unloaded: bool = False
    bias_v: float | None = None
    signal_contact: str | None = None

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

    def resampled(self, freq_hz: ArrayLike) -> RFLineParams:
        """These parameters on another frequency grid.

        Linear interpolation of the index, the loss and the complex
        impedance — its real and imaginary parts separately — onto
        ``freq_hz``, with the end values held outside the solved range
        rather than extrapolated. What the line Stage's response grid and
        the SAX line model both read, so a frequency axis interpolates
        one way everywhere.

        Args:
            freq_hz: Frequencies to sample at (Hz, ascending, 1D).

        Returns:
            The same line on the new grid, carrying the same Bias, signal
            Contact and loaded/unloaded flag.

        Raises:
            ValueError: When the grid is not ascending.
        """
        grid = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
        if grid.ndim != 1 or np.any(np.diff(grid) < 0.0):
            raise ValueError("freq_hz must be a 1D ascending frequency grid.")
        return RFLineParams(
            freq_hz=grid,
            n_rf=np.interp(grid, self.freq_hz, self.n_rf),
            alpha_rf_np_m=np.interp(grid, self.freq_hz, self.alpha_rf_np_m),
            z0_ohm=np.asarray(
                np.interp(grid, self.freq_hz, self.z0_ohm.real)
                + 1j * np.interp(grid, self.freq_hz, self.z0_ohm.imag),
                dtype=np.complex128,
            ),
            unloaded=self.unloaded,
            bias_v=self.bias_v,
            signal_contact=self.signal_contact,
        )


def line_params_from_neff(
    freq_hz: ArrayLike,
    n_eff: ArrayLike,
    *,
    z0_ohm: ArrayLike,
    unloaded: bool = False,
    bias_v: float | None = None,
    signal_contact: str | None = None,
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
        unloaded: Flag the result as the bare electrode's — solved with
            the carriers switched off — rather than a Bias point's.
        bias_v: The Bias the Cross-section was built at (V), if any.
        signal_contact: The Contact the impedance was read over, if any.

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
        freq_hz=freq.copy(),
        n_rf=np.array(n_arr.real, dtype=np.float64),
        alpha_rf_np_m=np.asarray(np.abs(n_arr.imag) * omega / C0, dtype=np.float64),
        z0_ohm=np.array(z0, dtype=np.complex128),
        unloaded=unloaded,
        bias_v=bias_v,
        signal_contact=signal_contact,
    )


def line_params_from_gamma(
    freq_hz: ArrayLike,
    gamma_per_m: ArrayLike,
    *,
    z0_ohm: ArrayLike,
    unloaded: bool = False,
    bias_v: float | None = None,
    signal_contact: str | None = None,
) -> RFLineParams:
    """Build :class:`RFLineParams` from complex propagation constants.

    The inverse of :attr:`RFLineParams.gamma_per_m`, for routes that
    produce ``gamma`` directly — the loaded-line assembly
    (:func:`gsim.common.twmzm.loaded_line_params`) rather than a mode
    solve: ``n_RF = |Im(gamma)| c0 / omega`` and
    ``alpha = |Re(gamma)|``, so either sign convention is read as loss.

    Args:
        freq_hz: RF frequencies in Hz.
        gamma_per_m: Complex propagation constant per frequency (1/m).
        z0_ohm: Characteristic impedance per frequency (complex allowed).
        unloaded: Flag the result as the bare electrode's.
        bias_v: The Bias the Cross-section was built at (V), if any.
        signal_contact: The Contact the impedance was read over, if any.

    Returns:
        The RF line parameters.
    """
    freq = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
    gamma = np.broadcast_to(
        np.atleast_1d(np.asarray(gamma_per_m, dtype=np.complex128)), freq.shape
    )
    z0 = np.broadcast_to(
        np.atleast_1d(np.asarray(z0_ohm, dtype=np.complex128)), freq.shape
    )
    omega = 2.0 * np.pi * freq
    return RFLineParams(
        freq_hz=freq.copy(),
        n_rf=np.asarray(np.abs(gamma.imag) * C0 / omega, dtype=np.float64),
        alpha_rf_np_m=np.array(np.abs(gamma.real), dtype=np.float64),
        z0_ohm=np.array(z0, dtype=np.complex128),
        unloaded=unloaded,
        bias_v=bias_v,
        signal_contact=signal_contact,
    )


class LoadedLineComparison(BaseModel):
    """The two loaded-line routes side by side, at one Bias point.

    The direct route solves the carrier-loaded Staircase as one
    cross-section; the assembled route combines the unloaded (bare
    electrode) RLGC with the charge solve's series-RC junction branch per
    unit length. Both are the same compact model assembled two ways, so
    their n_RF, loss and Z0 should agree — up to what the lumped branch
    cannot capture of the distributed junction. The known systematic gap:
    the undepleted slab conducts, so in the direct solve it extends the
    electrode plates toward the junction and every field path (oxide,
    substrate, air) sees the narrowed gap, while the assembly keeps the
    bare electrode's shunt parameters and adds only the junction's own
    R_s/C_j — the assembled route therefore reads consistently light. On
    the demo device it sits ~20% low on n_RF, 20-45% low on the loss and
    20-35% high on |Z0| (the finer the charge mesh resolves the slab,
    the lossier the direct solve), flat across 10-30 GHz. The default
    tolerances of :meth:`check` are set just outside that gap; a route
    bug (a dropped conductivity, a wrong-branch mode, a unit slip)
    overshoots them by multiples.

    Attributes:
        direct: The direct loaded solve's line parameters.
        assembled: The loaded-line assembly's parameters, from the
            unloaded RLGC plus the junction branch.
    """

    direct: RFLineParams
    assembled: RFLineParams

    @model_validator(mode="after")
    def validate_axes(self) -> Self:
        """Both routes answer on the same frequency axis."""
        if self.direct.freq_hz.shape != self.assembled.freq_hz.shape or np.any(
            self.direct.freq_hz != self.assembled.freq_hz
        ):
            raise ValueError("The two routes must share one freq_hz axis.")
        return self

    @property
    def freq_hz(self) -> NDArray[np.float64]:
        """The shared frequency axis (Hz)."""
        return self.direct.freq_hz

    @property
    def delta_n_rf(self) -> NDArray[np.float64]:
        """Relative n_RF difference, assembled against direct."""
        return np.asarray(
            (self.assembled.n_rf - self.direct.n_rf) / self.direct.n_rf,
            dtype=np.float64,
        )

    @property
    def delta_alpha(self) -> NDArray[np.float64]:
        """Relative RF-loss difference, assembled against direct."""
        return np.asarray(
            (self.assembled.alpha_rf_np_m - self.direct.alpha_rf_np_m)
            / self.direct.alpha_rf_np_m,
            dtype=np.float64,
        )

    @property
    def delta_z0(self) -> NDArray[np.float64]:
        """Relative |Z0| deviation, assembled against direct."""
        return np.asarray(
            np.abs(self.assembled.z0_ohm - self.direct.z0_ohm)
            / np.abs(self.direct.z0_ohm),
            dtype=np.float64,
        )

    def check(
        self,
        *,
        rtol_n_rf: float = 0.3,
        rtol_alpha: float = 0.55,
        rtol_z0: float = 0.45,
    ) -> None:
        """Fail loudly where the two routes disagree.

        The defaults sit just outside the systematic gap the class
        docstring describes (the assembled route reading ~20% light on
        the demo device), so they pass an honest assembly and fail a
        broken route, which misses by multiples.

        Args:
            rtol_n_rf: Relative tolerance on n_RF.
            rtol_alpha: Relative tolerance on the RF loss.
            rtol_z0: Relative tolerance on |Z0|.

        Raises:
            ValueError: Naming the quantity that diverged and the
                frequency it diverged at.
        """
        for name, delta, rtol in (
            ("n_RF", self.delta_n_rf, rtol_n_rf),
            ("the RF loss", self.delta_alpha, rtol_alpha),
            ("Z0", self.delta_z0, rtol_z0),
        ):
            excess = np.abs(delta) > rtol
            if np.any(excess):
                where = int(np.argmax(np.abs(np.where(excess, delta, 0.0))))
                raise ValueError(
                    f"The two loaded-line routes disagree on {name}: "
                    f"{delta[where]:+.1%} relative at "
                    f"{self.freq_hz[where] / 1e9:g} GHz "
                    f"(tolerance {rtol:.0%}). Direct "
                    f"{self.direct.n_rf[where]:g}/"
                    f"{self.direct.alpha_rf_np_m[where]:g}/"
                    f"{self.direct.z0_ohm[where]:.3g} vs assembled "
                    f"{self.assembled.n_rf[where]:g}/"
                    f"{self.assembled.alpha_rf_np_m[where]:g}/"
                    f"{self.assembled.z0_ohm[where]:.3g} "
                    "(n_RF/loss Np/m/Z0 ohm)."
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
