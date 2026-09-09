"""Compact-model export of a uniform line (pure circuit functions).

The Traveling-wave electrode leaves gsim as a two-port: the solved line
parameters ``gamma(f)`` and ``Z0(f)`` plus a chosen length become the
uniform line's S-matrix (:func:`line_smatrix`), written to a Touchstone
``.s2p`` file (:func:`write_touchstone`) or wrapped as a callable
following the SAX model convention (:func:`sax_line_model`) — a plain
function over numpy arrays returning a dict of S-matrix entries, with no
sax import anywhere. This is gsim's half of the compact-model handoff to
a circuit simulator such as circulax: the physics is solved here, the
assembly happens there.

The S-matrix referenced to a real ``Z_ref`` follows the standard
telegrapher's two-port (e.g. Pozar, *Microwave Engineering*, ch. 4)::

    S11 = S22 = (Zc^2 - Zr^2) sinh(gl) / D
    S21 = S12 = 2 Zc Zr / D
    D = 2 Zc Zr cosh(gl) + (Zc^2 + Zr^2) sinh(gl)

with ``Zc`` the line's characteristic impedance, ``Zr`` the reference,
and ``gl = gamma * length`` the complex electrical length. Sign
convention matches the rest of gsim: ``gamma = alpha + j beta`` with
``alpha >= 0`` for a lossy line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "SaxLineModel",
    "line_smatrix",
    "sax_line_model",
    "write_touchstone",
]


class SaxLineModel(Protocol):
    """A SAX-convention model: keyword arguments in, S-dict out."""

    def __call__(
        self, *, f: ArrayLike | None = None
    ) -> dict[tuple[str, str], NDArray[np.complex128]]:
        """Evaluate the S-matrix entries at frequencies ``f`` (Hz)."""
        ...


def _require_positive_length(length_m: float) -> None:
    """Refuse a line of zero or negative length."""
    if length_m <= 0.0:
        raise ValueError("length_m must be a positive line length in meters.")


def _require_ascending(freq_hz: NDArray[np.float64], where: str) -> None:
    """Refuse a frequency axis ``np.interp`` (or Touchstone) would misread.

    Args:
        freq_hz: The frequency axis (1D).
        where: Argument name for the error message.
    """
    if freq_hz.ndim != 1 or freq_hz.size == 0:
        raise ValueError(f"{where} must be a non-empty 1D array.")
    if np.any(np.diff(freq_hz) <= 0.0):
        raise ValueError(
            f"{where} must be strictly ascending: a descending or repeated "
            "frequency axis interpolates and reads back silently wrong."
        )


def _interp_complex(
    grid: NDArray[np.float64],
    freq_hz: NDArray[np.float64],
    values: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Linear interpolation of a complex array, ends held."""
    return np.asarray(
        np.interp(grid, freq_hz, values.real)
        + 1j * np.interp(grid, freq_hz, values.imag),
        dtype=np.complex128,
    )


def line_smatrix(
    gamma_per_m: ArrayLike,
    z0_ohm: ArrayLike,
    *,
    length_m: float,
    z_ref_ohm: complex = 50.0,
) -> NDArray[np.complex128]:
    """The two-port S-matrix of a uniform line, referenced to ``z_ref_ohm``.

    Args:
        gamma_per_m: Complex propagation constant ``alpha + j beta`` per
            frequency (1/m).
        z0_ohm: Complex characteristic impedance per frequency (ohm); a
            scalar broadcasts over ``gamma_per_m``.
        length_m: Line length in meters (> 0).
        z_ref_ohm: Port reference impedance (ohm).

    Returns:
        The S-matrix, shaped ``gamma_per_m.shape + (2, 2)``.

    Raises:
        ValueError: When the length is not positive, or the two line
            parameter arrays do not broadcast against each other.
    """
    _require_positive_length(length_m)
    gamma = np.asarray(gamma_per_m, dtype=np.complex128)
    z_c = np.asarray(z0_ohm, dtype=np.complex128)
    try:
        gamma, z_c = np.broadcast_arrays(gamma, z_c)
    except ValueError as error:
        raise ValueError(
            f"gamma_per_m (shape {np.shape(gamma_per_m)}) and z0_ohm (shape "
            f"{np.shape(z0_ohm)}) must share one frequency axis."
        ) from error
    z_r = complex(z_ref_ohm)

    gl = gamma * length_m
    sinh, cosh = np.sinh(gl), np.cosh(gl)
    denom = 2.0 * z_c * z_r * cosh + (z_c**2 + z_r**2) * sinh
    s11 = (z_c**2 - z_r**2) * sinh / denom
    s21 = 2.0 * z_c * z_r / denom

    s = np.empty((*gamma.shape, 2, 2), dtype=np.complex128)
    s[..., 0, 0] = s11
    s[..., 1, 1] = s11
    s[..., 0, 1] = s21
    s[..., 1, 0] = s21
    return s


def write_touchstone(
    path: str | Path,
    *,
    freq_hz: ArrayLike,
    s: ArrayLike,
    z_ref_ohm: float = 50.0,
    comments: list[str] | None = None,
) -> Path:
    """Write a two-port S-matrix as a Touchstone v1 ``.s2p`` file.

    The file says ``# Hz S RI R <z_ref>`` and carries one row per
    frequency in the Touchstone two-port order (S11, S21, S12, S22),
    real and imaginary columns — what scikit-rf, ADS or any Touchstone
    consumer reads back without conversion.

    Args:
        path: Output file; the ``.s2p`` suffix is added when missing.
        freq_hz: Frequencies in Hz (ascending, 1D).
        s: The S-matrix, shaped ``(len(freq_hz), 2, 2)``.
        z_ref_ohm: Port reference impedance (ohm). Touchstone references
            are real, so an imaginary part is refused rather than
            silently dropped.
        comments: Extra provenance lines written as ``!`` comments.

    Returns:
        The written path.

    Raises:
        ValueError: On a complex reference impedance, a non-ascending
            frequency axis, or mismatched array shapes.
    """
    # complex() rather than trusting the annotation: a complex reference
    # passed at runtime must be refused, not truncated.
    z_r = complex(z_ref_ohm)
    if z_r.imag != 0.0 or z_r.real <= 0.0:
        raise ValueError(
            "A Touchstone reference impedance is a positive real number; got "
            f"{z_r}. Renormalize the S-matrix to a real reference instead."
        )
    freq = np.asarray(freq_hz, dtype=np.float64)
    _require_ascending(freq, "freq_hz")
    matrix = np.asarray(s, dtype=np.complex128)
    if matrix.shape != (freq.size, 2, 2):
        raise ValueError(
            f"s (shape {matrix.shape}) must be (len(freq_hz), 2, 2) with "
            f"freq_hz 1D (shape {freq.shape})."
        )

    target = Path(path)
    if target.suffix.lower() != ".s2p":
        target = target.with_suffix(".s2p")

    lines = [f"! {comment}" for comment in ("gsim line two-port", *(comments or []))]
    lines.append(f"# Hz S RI R {z_r.real:g}")
    for row, entries in zip(freq, matrix, strict=True):
        # Touchstone two-port order: S11, S21, S12, S22.
        ordered = (
            entries[0, 0],
            entries[1, 0],
            entries[0, 1],
            entries[1, 1],
        )
        values = " ".join(f"{v.real:.12e} {v.imag:.12e}" for v in ordered)
        lines.append(f"{row:.12e} {values}")
    target.write_text("\n".join(lines) + "\n")
    return target


def sax_line_model(
    freq_hz: ArrayLike,
    gamma_per_m: ArrayLike,
    z0_ohm: ArrayLike,
    *,
    length_m: float,
    z_ref_ohm: complex = 50.0,
) -> SaxLineModel:
    """Wrap the solved line as a SAX-convention S-model over frequency.

    The returned callable is a plain function needing only numpy: called
    with ``f`` (Hz, scalar or array; the solved frequencies when
    omitted), it linearly interpolates ``gamma`` and ``Z0`` onto ``f``
    — held at the end values outside the solved range — and returns the
    dict of S-matrix entries keyed by port pairs ``("o1", "o1")`` ...
    ``("o2", "o2")``, the sdict convention SAX composes circuits from.
    The closure copies its inputs, so it stays valid after the arrays it
    was built from are mutated or garbage collected.

    Args:
        freq_hz: Frequencies the line was solved at (Hz, ascending, 1D).
        gamma_per_m: Complex propagation constant per frequency (1/m).
        z0_ohm: Complex characteristic impedance per frequency (ohm).
        length_m: Line length in meters (> 0).
        z_ref_ohm: Port reference impedance (ohm).

    Returns:
        The model callable.

    Raises:
        ValueError: When the length is not positive, the frequency axis
            is not ascending, or the arrays do not share it.
    """
    _require_positive_length(length_m)
    freq = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64)).copy()
    _require_ascending(freq, "freq_hz")
    gamma = np.broadcast_to(
        np.asarray(gamma_per_m, dtype=np.complex128), freq.shape
    ).copy()
    z_c = np.broadcast_to(np.asarray(z0_ohm, dtype=np.complex128), freq.shape).copy()

    def model(
        *, f: ArrayLike | None = None
    ) -> dict[tuple[str, str], NDArray[np.complex128]]:
        """The line's S-matrix entries at frequencies ``f`` (Hz)."""
        grid = freq if f is None else np.asarray(f, dtype=np.float64)
        gamma_f = _interp_complex(grid, freq, gamma)
        z0_f = _interp_complex(grid, freq, z_c)
        s = line_smatrix(gamma_f, z0_f, length_m=length_m, z_ref_ohm=z_ref_ohm)
        return {
            ("o1", "o1"): s[..., 0, 0],
            ("o1", "o2"): s[..., 0, 1],
            ("o2", "o1"): s[..., 1, 0],
            ("o2", "o2"): s[..., 1, 1],
        }

    return model
