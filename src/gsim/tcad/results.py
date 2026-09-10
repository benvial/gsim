"""Result containers for the charge-transport solve.

A 2D DEVSIM device is one cm deep, so extensive quantities (currents,
charges, capacitances) are per cm of device depth. Coordinates are
converted back to the gsim mesh unit (um).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["BiasPoint", "BiasSweepResult", "CarrierMap"]


class CarrierMap(BaseModel):
    """Node-wise solution fields on the cross-section mesh.

    Attributes:
        x_um: In-plane node coordinates (um).
        y_um: Vertical node coordinates (um).
        region: Per-node mesh region name.
        electrons_cm3: Electron concentration n(x, y) (cm^-3).
        holes_cm3: Hole concentration p(x, y) (cm^-3).
        potential_v: Electrostatic potential (V).
        net_doping_cm3: Net doping (donors - acceptors, cm^-3).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x_um: NDArray[np.float64]
    y_um: NDArray[np.float64]
    region: list[str]
    electrons_cm3: NDArray[np.float64]
    holes_cm3: NDArray[np.float64]
    potential_v: NDArray[np.float64]
    net_doping_cm3: NDArray[np.float64]


class BiasPoint(BaseModel):
    """Solved state of the device at one bias voltage.

    Attributes:
        bias_v: Applied bias on the swept contact (V).
        carriers: Node-wise carrier maps.
        currents_a_per_cm: Total terminal current per contact
            (electron + hole, A per cm of depth).
        charge_c_per_cm: Contact charge on the swept contact
            (C per cm of depth).
        capacitance_f_per_cm: Small-signal capacitance |dQ/dV| at this
            bias (F per cm of depth).
        admittance_s_per_cm: Complex small-signal terminal admittance of
            the swept contact at :attr:`admittance_freq_hz`
            (S per cm of depth).
        admittance_freq_hz: Frequency the admittance's AC solve ran at
            (Hz) — above the quasi-static one, so the capacitive current
            dominates the junction leakage in ``Re(Y)``; zero when no AC
            solve produced this point.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bias_v: float
    carriers: CarrierMap
    currents_a_per_cm: dict[str, float] = Field(default_factory=dict)
    charge_c_per_cm: float = 0.0
    capacitance_f_per_cm: float = 0.0
    admittance_s_per_cm: complex = 0.0j
    admittance_freq_hz: float = 0.0

    @property
    def capacitance_f_per_m(self) -> float:
        """Small-signal capacitance per meter of device length (F/m)."""
        return self.capacitance_f_per_cm * 1e2

    @property
    def admittance_s_per_m(self) -> complex:
        """Small-signal admittance per meter of device length (S/m)."""
        return self.admittance_s_per_cm * 1e2

    def junction_branch(self) -> tuple[float, float]:
        """Fit the series-RC junction branch to this point's admittance.

        The lumped shunt model the standard loaded-line workflow inserts
        per unit length of the Traveling-wave electrode: the junction
        capacitance behind the series resistance of the doped slab.

        Returns:
            ``(r_s_ohm_m, c_j_f_per_m)`` — series resistance (ohm*m) and
            junction capacitance (F/m).

        Raises:
            ValueError: When this point holds no small-signal admittance,
                or one a series RC cannot represent.
        """
        from gsim.common.twmzm import series_rc_from_admittance

        if self.admittance_freq_hz <= 0 or self.admittance_s_per_cm == 0:
            raise ValueError(
                f"The bias point at {self.bias_v:g} V holds no small-signal "
                "admittance; re-solve it with a charge backend recent enough "
                "to report one."
            )
        r_s, c_j = series_rc_from_admittance(
            self.admittance_s_per_m, freq_hz=self.admittance_freq_hz
        )
        return float(r_s), float(c_j)


class BiasSweepResult(BaseModel):
    """Ordered collection of solved bias points from a voltage sweep."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    contact: str
    points: list[BiasPoint] = Field(default_factory=list)

    @property
    def voltages(self) -> NDArray[np.float64]:
        """Applied biases (V) in sweep order."""
        return np.asarray([p.bias_v for p in self.points], dtype=np.float64)

    @property
    def capacitance_f_per_cm(self) -> NDArray[np.float64]:
        """Small-signal C(V) per cm of depth in sweep order."""
        return np.asarray(
            [p.capacitance_f_per_cm for p in self.points], dtype=np.float64
        )

    @property
    def capacitance_f_per_m(self) -> NDArray[np.float64]:
        """Small-signal C(V) per meter of device length in sweep order."""
        return np.asarray(self.capacitance_f_per_cm * 1e2, dtype=np.float64)

    @property
    def admittance_s_per_cm(self) -> NDArray[np.complex128]:
        """Small-signal terminal admittance per cm of depth in sweep order."""
        return np.asarray(
            [p.admittance_s_per_cm for p in self.points], dtype=np.complex128
        )

    def junction_branch(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The series-RC junction branch fitted at every Bias point.

        Returns:
            ``(r_s_ohm_m, c_j_f_per_m)`` arrays in sweep order — series
            resistance (ohm*m) and junction capacitance (F/m) per meter
            of Traveling-wave electrode.
        """
        fitted = [p.junction_branch() for p in self.points]
        return (
            np.asarray([r for r, _ in fitted], dtype=np.float64),
            np.asarray([c for _, c in fitted], dtype=np.float64),
        )
