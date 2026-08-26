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
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bias_v: float
    carriers: CarrierMap
    currents_a_per_cm: dict[str, float] = Field(default_factory=dict)
    charge_c_per_cm: float = 0.0
    capacitance_f_per_cm: float = 0.0

    @property
    def capacitance_f_per_m(self) -> float:
        """Small-signal capacitance per meter of device length (F/m)."""
        return self.capacitance_f_per_cm * 1e2


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
