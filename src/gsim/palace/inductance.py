"""Inductance matrix from a magnetostatic Palace run.

For a problem with N current sources, Palace drives each source in turn with a
unit surface current and writes the inductance matrix (``terminal-M.csv``), its
inverse, the excitation currents and the stored energies. This module reads
them, labels them with the source names and checks that they hold together.

The conventions, in Palace's own outputs:

- Units are SI: H for the matrix, 1/H for its inverse, A for the currents and
  J for the energies. Conductors are perfect, so the matrix is the external
  inductance, without the internal inductance of the metal.
- ``M[i][j]`` is signed. It is positive when unit currents along each source's
  direction link flux in the same sense; reversing one source flips the sign of
  its row and column. Either sign is physical.
- Each excitation drives ``I_inc[i] = 1/sqrt(Z0)``, about 0.0515 A, not 1 A.
  The matrix does not depend on it; the stored magnetic energy of excitation
  ``i`` is ``E_mag = 1/2 M[i][i] I_inc[i]**2``.
- Palace also writes ``terminal-Mm.csv``, a current-difference form fully
  derived from ``M``; it is not read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gsim.palace._terminal_csv import _read_column, _read_table, _result_file

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd
    from numpy.typing import NDArray

    from gsim.palace.results import PalaceTextResults


@dataclass(frozen=True, eq=False)
class InductanceMatrix:
    """The inductance matrix of a magnetostatic run, labelled by current source.

    Attributes:
        sources: Current-source names, in Palace's index order.
        matrix: Inductance matrix, in H. Off-diagonal entries are signed.
        inverse: Inverse of the matrix, in 1/H, when Palace wrote it.
        excitation_current: Current of each excitation, in A.
        stored_energy: Magnetic energy of each excitation, in J, at its
            excitation current.
    """

    sources: tuple[str, ...]
    matrix: NDArray[np.float64]
    inverse: NDArray[np.float64] | None = None
    excitation_current: NDArray[np.float64] | None = None
    stored_energy: NDArray[np.float64] | None = None

    def _index(self, name: str) -> int:
        """Position of a source in the matrix, KeyError if there is none."""
        try:
            return self.sources.index(name)
        except ValueError:
            msg = f"unknown source {name!r}; the sources are {list(self.sources)}"
            raise KeyError(msg) from None

    def self_inductance(self, name: str) -> float:
        """Self inductance of a current source, in H."""
        index = self._index(name)
        return float(self.matrix[index, index])

    def mutual_inductance(self, first: str, second: str) -> float:
        """Mutual inductance of two current sources, in H, signed."""
        if first == second:
            msg = (
                "mutual_inductance() needs two different sources; "
                "use self_inductance() for one"
            )
            raise ValueError(msg)
        return float(self.matrix[self._index(first), self._index(second)])

    def coupling(self, first: str, second: str) -> float:
        """Coupling coefficient ``k = M_ij / sqrt(M_ii M_jj)``, signed.

        NaN when either self inductance is not positive.
        """
        return float(self._coupling_matrix()[self._index(first), self._index(second)])

    def inductance_frame(self) -> pd.DataFrame:
        """The inductance matrix, in H, as a DataFrame indexed by source name."""
        import pandas as pd

        return pd.DataFrame(self.matrix, index=self.sources, columns=self.sources)

    def coupling_frame(self) -> pd.DataFrame:
        """The coupling coefficients as a DataFrame indexed by source name.

        The diagonal is 1; see :meth:`coupling`.
        """
        import pandas as pd

        return pd.DataFrame(
            self._coupling_matrix(), index=self.sources, columns=self.sources
        )

    def pairs(self) -> pd.DataFrame:
        """Every self and mutual inductance, with its coupling coefficient.

        One row per source against itself (its self inductance, coupling 1),
        then one row per pair of sources (their mutual inductance).

        Returns:
            A table with the columns ``first``, ``second``, ``inductance_H``
            and ``coupling``.
        """
        import pandas as pd

        coupling = self._coupling_matrix()
        count = len(self.sources)
        indices = [(i, i) for i in range(count)]
        indices += [(i, j) for i in range(count) for j in range(i + 1, count)]
        rows = [
            (
                self.sources[i],
                self.sources[j],
                float(self.matrix[i, j]),
                float(coupling[i, j]),
            )
            for i, j in indices
        ]
        return pd.DataFrame(
            rows, columns=["first", "second", "inductance_H", "coupling"]
        )

    def problems(self, rtol: float = 1e-3) -> list[str]:
        """List what is inconsistent in the matrix; empty when it holds together.

        The checks, in order: every value is finite; every self inductance is
        positive; the matrix is reciprocal; the magnetic energy is not negative
        for any set of currents; no coupling coefficient exceeds 1 in
        magnitude; and, when Palace wrote them, the inverse inverts the matrix
        and each excitation's stored energy is ``1/2 M_ii I_i**2``. The first
        two stop the checks, since the others are meaningless without them.
        A coupling above 1 is the visible sign of a negative energy, so the
        energy is only reported when every coupling is within bounds.

        These are physics sanity checks; a run that did not converge can
        still pass them.

        Args:
            rtol: Tolerance. Reciprocity, energy sign and coupling are judged
                on the coupling matrix ``k_ij = M_ij / sqrt(M_ii M_jj)``, so a
                weak mutual next to a large self inductance is judged on its
                own scale; the stored energy is compared per source.
        """
        values = {
            "the inductance matrix": self.matrix,
            "the inverse": self.inverse,
            "the excitation currents": self.excitation_current,
            "the stored energies": self.stored_energy,
        }
        bad = [
            name
            for name, v in values.items()
            if v is not None and not np.isfinite(v).all()
        ]
        if bad:
            return [
                f"these are not finite (NaN or infinity): {', '.join(bad)}. A NaN "
                f"off-diagonal entry usually means the inactive current sources "
                f"were shorted (Palace InactivePorts 'Short') instead of left open"
            ]

        diagonal = np.diag(self.matrix)
        if diagonal.min() <= 0:
            name = self.sources[int(diagonal.argmin())]
            return [
                f"source {name!r} has no positive self inductance "
                f"({diagonal.min():.3g} H): its current loop may not close, or "
                f"this is not an inductance matrix"
            ]

        coupling = self._coupling_matrix()
        found: list[str] = []

        asymmetry = float(np.abs(coupling - coupling.T).max())
        if asymmetry > rtol:
            found.append(
                f"the inductance matrix is not reciprocal: "
                f"max |k_ij - k_ji| = {asymmetry:.3g}"
            )

        largest = float(np.abs(coupling - np.eye(len(coupling))).max())
        smallest = float(np.linalg.eigvalsh((coupling + coupling.T) / 2).min())
        if largest > 1 + rtol:
            found.append(
                f"a coupling coefficient above 1 in magnitude: max |k_ij| = "
                f"{largest:.4g}, which no passive pair of loops can have"
            )
        elif smallest < -rtol:
            found.append(
                f"the magnetic energy is negative for some currents: the coupling "
                f"matrix has an eigenvalue of {smallest:.3g}"
            )

        if self.inverse is not None:
            error = float(
                np.abs(self.matrix @ self.inverse - np.eye(len(self.matrix))).max()
            )
            if error > rtol:
                found.append(
                    f"the inverse does not invert the inductance matrix: "
                    f"max |M M^-1 - I| = {error:.3g}"
                )

        if self.excitation_current is not None and self.stored_energy is not None:
            expected = 0.5 * diagonal * self.excitation_current**2
            error = np.abs(self.stored_energy - expected) / expected
            if error.max() > rtol:
                name = self.sources[int(error.argmax())]
                found.append(
                    f"the stored energy does not match 1/2 M_ii I_i^2: off by "
                    f"{error.max():.3g} (relative) for source {name!r}"
                )
        return found

    def _coupling_matrix(self) -> NDArray[np.float64]:
        """``D^-1/2 M D^-1/2`` with ``D`` the diagonal; NaN where it has no meaning."""
        diagonal = np.diag(self.matrix)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(diagonal > 0, 1.0 / np.sqrt(diagonal), np.nan)
        coupling = self.matrix * np.outer(scale, scale)
        np.fill_diagonal(coupling, np.where(diagonal > 0, 1.0, np.nan))
        return coupling


def load_inductance(
    source: str | Path | dict | PalaceTextResults,
    *,
    source_names: Sequence[str] | None = None,
) -> InductanceMatrix:
    """Load the inductance matrix of a magnetostatic Palace run.

    Args:
        source: The results dict that ``run()`` returns, the
            ``PalaceTextResults`` that ``run_local()`` returns, or the
            simulation or Palace output directory.
        source_names: Names for the current sources, in Palace's index order,
            which is the order they were added to the simulation. Defaults to
            ``S1``, ``S2``, ...

    Returns:
        The matrix, labelled by current source.

    Raises:
        FileNotFoundError: If ``terminal-M.csv`` is missing.
        ValueError: If the matrix is not square, an optional table does not
            have one row per source, or the names do not match the matrix.
    """
    matrix_path = _result_file(source, "terminal-M.csv")
    if matrix_path is None:
        where = str(source) if isinstance(source, (str, Path)) else "the results"
        msg = f"terminal-M.csv not found in {where}"
        raise FileNotFoundError(msg)
    matrix = _read_table(matrix_path)[:, 1:]
    count = len(matrix)
    if matrix.shape != (count, count):
        msg = f"the inductance matrix in terminal-M.csv is not square: {matrix.shape}"
        raise ValueError(msg)

    names = (
        tuple(f"S{index}" for index in range(1, count + 1))
        if source_names is None
        else tuple(source_names)
    )
    if len(names) != count:
        msg = f"{len(names)} names given for {count} sources"
        raise ValueError(msg)
    if len(set(names)) != count:
        msg = f"the source names must be different: {list(names)}"
        raise ValueError(msg)

    def optional(
        name: str, read: Callable[[Path], NDArray[np.float64] | None], shape: tuple
    ) -> NDArray[np.float64] | None:
        path = _result_file(source, name)
        values = None if path is None else read(path)
        if values is not None and values.shape != shape:
            msg = (
                f"{name} has shape {values.shape}, expected {shape} for {count} sources"
            )
            raise ValueError(msg)
        return values

    return InductanceMatrix(
        sources=names,
        matrix=matrix,
        inverse=optional(
            "terminal-Minv.csv", lambda p: _read_table(p)[:, 1:], (count, count)
        ),
        excitation_current=optional(
            "terminal-I.csv", lambda p: _read_table(p)[:, 1], (count,)
        ),
        stored_energy=optional(
            "domain-E.csv", lambda p: _read_column(p, "E_mag (J)"), (count,)
        ),
    )


__all__ = ["InductanceMatrix", "load_inductance"]
