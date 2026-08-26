"""Traveling-wave Mach-Zehnder modulator assembly (pure analysis functions).

Combines RF transmission-line parameters — effective RF index, loss, and
characteristic impedance versus frequency — with the optical phase response
into the standard traveling-wave modulator figures of merit:

- small-signal electro-optic frequency response including velocity mismatch,
  RF loss, and impedance mismatch with source/load reflections
  (:func:`eo_response`, :func:`eo_bandwidth`);
- the analytic walk-off-limited bandwidth of a lossless matched line
  (:func:`walkoff_bandwidth`);
- modulation efficiency ``V_pi L`` from a bias sweep of the effective-index
  shift (:func:`vpi_length_vcm`);
- RLGC line parameters from the propagation constant and characteristic
  impedance (:func:`rlgc_from_line_params`, Marks-Williams relations).

The response model follows the classic single-drive analysis (e.g. Ghione,
*Semiconductor Devices for High-Speed Optoelectronics*, ch. 6): the voltage
wave on a lossy line of length L terminated in ``Z_L`` and driven through
``Z_g`` is averaged over the co-propagating optical group delay.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import speed_of_light as C0  # noqa: N812

__all__ = [
    "SINC_3DB_ARGUMENT",
    "eo_bandwidth",
    "eo_response",
    "rlgc_from_line_params",
    "vpi_length_vcm",
    "walkoff_bandwidth",
]

#: Argument where ``|sin(u)/u|`` falls to 1/sqrt(2) (walk-off 3 dB point).
SINC_3DB_ARGUMENT: float = 1.3915573782515105


def _f_avg(u: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Evaluate ``F(u) = (exp(u) - 1) / u`` with the ``F(0) = 1`` limit."""
    out = np.ones_like(u)
    mask = np.abs(u) > 1e-12
    out[mask] = np.expm1(u[mask]) / u[mask]
    return out


def _effective_voltage(
    freq_hz: NDArray[np.float64],
    *,
    length_m: float,
    n_rf: NDArray[np.float64],
    n_opt: float,
    alpha_rf_np_m: NDArray[np.float64],
    z0_ohm: NDArray[np.complex128],
    z_load_ohm: complex,
    z_gen_ohm: complex,
) -> NDArray[np.complex128]:
    """Optically averaged line voltage per volt of generator amplitude."""
    omega = 2.0 * np.pi * freq_hz
    gamma = alpha_rf_np_m + 1j * omega * n_rf / C0
    beta_opt = omega * n_opt / C0

    gamma_l = gamma * length_m
    reflect_load = (z_load_ohm - z0_ohm) / (z_load_ohm + z0_ohm)
    round_trip = reflect_load * np.exp(-2.0 * gamma_l)

    tanh_gl = np.tanh(gamma_l)
    z_in = z0_ohm * (z_load_ohm + z0_ohm * tanh_gl) / (z0_ohm + z_load_ohm * tanh_gl)
    v_input = z_in / (z_in + z_gen_ohm)
    v_forward = v_input / (1.0 + round_trip)

    u_forward = (1j * beta_opt - gamma) * length_m
    u_backward = (1j * beta_opt + gamma) * length_m
    return v_forward * (_f_avg(u_forward) + round_trip * _f_avg(u_backward))


def eo_response(
    freq_hz: ArrayLike,
    *,
    length_m: float,
    n_rf: ArrayLike,
    n_opt: float,
    alpha_rf_np_m: ArrayLike,
    z0_ohm: ArrayLike,
    z_load_ohm: complex,
    z_gen_ohm: complex,
    normalize: bool = True,
) -> NDArray[np.complex128]:
    """Small-signal electro-optic frequency response of a TW modulator.

    Averages the RF voltage wave — including load/source reflections — over
    the optical group propagation, capturing velocity mismatch, RF loss, and
    impedance mismatch simultaneously.

    Args:
        freq_hz: RF frequencies in Hz (array).
        length_m: Electrode length in meters (> 0).
        n_rf: RF effective (phase) index; scalar or per-frequency array.
        n_opt: Optical group index.
        alpha_rf_np_m: RF amplitude loss in Np/m; scalar or per-frequency.
        z0_ohm: Characteristic impedance in ohms (complex allowed); scalar
            or per-frequency.
        z_load_ohm: Termination impedance in ohms.
        z_gen_ohm: Generator impedance in ohms.
        normalize: Divide by the DC value so the response tends to 1 at low
            frequency (uses the lowest-frequency line parameters at f = 0).

    Returns:
        Complex response, same shape as ``freq_hz``.
    """
    freq = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
    if length_m <= 0:
        raise ValueError("length_m must be positive.")
    if np.any(freq <= 0):
        raise ValueError("Frequencies must be positive (use normalize for DC).")

    n_rf_arr = np.broadcast_to(np.asarray(n_rf, dtype=np.float64), freq.shape)
    alpha_arr = np.broadcast_to(np.asarray(alpha_rf_np_m, dtype=np.float64), freq.shape)
    z0_arr = np.broadcast_to(np.asarray(z0_ohm, dtype=np.complex128), freq.shape)

    response = _effective_voltage(
        freq,
        length_m=length_m,
        n_rf=n_rf_arr,
        n_opt=n_opt,
        alpha_rf_np_m=alpha_arr,
        z0_ohm=z0_arr,
        z_load_ohm=complex(z_load_ohm),
        z_gen_ohm=complex(z_gen_ohm),
    )

    if normalize:
        dc = _effective_voltage(
            np.array([0.0]),
            length_m=length_m,
            n_rf=n_rf_arr[:1],
            n_opt=n_opt,
            alpha_rf_np_m=alpha_arr[:1],
            z0_ohm=z0_arr[:1],
            z_load_ohm=complex(z_load_ohm),
            z_gen_ohm=complex(z_gen_ohm),
        )[0]
        response = response / dc
    return response


def eo_bandwidth(
    freq_hz: ArrayLike,
    response: ArrayLike,
    *,
    threshold: float = 1.0 / np.sqrt(2.0),
) -> float | None:
    """First frequency where the normalized |response| crosses *threshold*.

    Args:
        freq_hz: Frequencies in Hz, ascending.
        response: Complex (or magnitude) response, normalized to 1 at DC.
        threshold: Magnitude threshold; the default ``1/sqrt(2)`` is the
            3 dB electro-optic point.

    Returns:
        Linearly interpolated crossing frequency in Hz, or ``None`` when the
        response never falls below the threshold.
    """
    freq = np.asarray(freq_hz, dtype=np.float64)
    mag = np.abs(np.asarray(response))
    if freq.shape != mag.shape:
        raise ValueError("freq_hz and response must have the same shape.")
    below = np.nonzero(mag < threshold)[0]
    if below.size == 0:
        return None
    i = int(below[0])
    if i == 0:
        return float(freq[0])
    f0, f1 = freq[i - 1], freq[i]
    m0, m1 = mag[i - 1], mag[i]
    return float(f0 + (threshold - m0) * (f1 - f0) / (m1 - m0))


def walkoff_bandwidth(*, length_m: float, n_rf: float, n_opt: float) -> float:
    """Walk-off-limited 3 dB bandwidth of a lossless matched line in Hz.

    Solves ``|sin(u)/u| = 1/sqrt(2)`` with ``u = pi f L |n_rf - n_opt| / c``.

    Args:
        length_m: Electrode length in meters (> 0).
        n_rf: RF effective index.
        n_opt: Optical group index (different from ``n_rf``).

    Returns:
        3 dB frequency in Hz.
    """
    if length_m <= 0:
        raise ValueError("length_m must be positive.")
    mismatch = abs(n_rf - n_opt)
    if mismatch == 0:
        raise ValueError("n_rf equals n_opt: walk-off bandwidth is unbounded.")
    return SINC_3DB_ARGUMENT * C0 / (np.pi * length_m * mismatch)


def vpi_length_vcm(
    voltages: ArrayLike,
    dn_eff: ArrayLike,
    *,
    wavelength_um: float,
) -> NDArray[np.float64]:
    """Modulation efficiency ``V_pi L`` in V*cm along a bias sweep.

    Uses the local slope of the effective-index shift:
    ``V_pi L = lambda / (2 |d(dn_eff)/dV|)``.

    Args:
        voltages: Bias voltages in volts (>= 2 points, ascending).
        dn_eff: Effective-index shift at each bias.
        wavelength_um: Vacuum wavelength in um.

    Returns:
        ``V_pi L`` in V*cm at each bias point.
    """
    v = np.asarray(voltages, dtype=np.float64)
    dn = np.asarray(dn_eff, dtype=np.float64)
    if v.shape != dn.shape or v.size < 2:
        raise ValueError("voltages and dn_eff must be equal-length with >= 2 points.")
    if wavelength_um <= 0:
        raise ValueError("wavelength_um must be positive.")
    slope = np.gradient(dn, v)
    if np.any(slope == 0):
        raise ValueError("dn_eff slope vanishes; V_pi L is unbounded there.")
    # lambda[um] / (2 |slope|) is in V*um; 1 V*cm = 1e4 V*um.
    return wavelength_um / (2.0 * np.abs(slope)) / 1e4


def rlgc_from_line_params(
    freq_hz: ArrayLike,
    *,
    gamma_per_m: ArrayLike,
    z0_ohm: ArrayLike,
) -> dict[str, NDArray[np.float64]]:
    """RLGC per-unit-length parameters from ``gamma`` and ``Z_0``.

    Uses the telegrapher relations ``R + j omega L = gamma Z_0`` and
    ``G + j omega C = gamma / Z_0`` (Marks & Williams).

    Args:
        freq_hz: Frequencies in Hz.
        gamma_per_m: Complex propagation constant ``alpha + j beta`` in 1/m.
        z0_ohm: Complex characteristic impedance in ohms.

    Returns:
        Dict with arrays ``R`` (ohm/m), ``L`` (H/m), ``G`` (S/m), ``C`` (F/m).
    """
    freq = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
    if np.any(freq <= 0):
        raise ValueError("Frequencies must be positive.")
    gamma = np.broadcast_to(np.asarray(gamma_per_m, dtype=np.complex128), freq.shape)
    z0 = np.broadcast_to(np.asarray(z0_ohm, dtype=np.complex128), freq.shape)
    omega = 2.0 * np.pi * freq
    series = gamma * z0
    shunt = gamma / z0
    return {
        "R": series.real.copy(),
        "L": series.imag / omega,
        "G": shunt.real.copy(),
        "C": shunt.imag / omega,
    }
