"""Analytic doping profiles for the charge-transport solve.

Doping is specified analytically per mesh region — no process simulation.
Three profile shapes cover the common cases:

- :class:`StepDoping`: uniform concentration inside an optional box.
- :class:`GaussianDoping`: separable Gaussian around a center point.
- :class:`ImplantDoping`: implant-like Gaussian in depth (projected range
  and straggle below a surface) with a hard lateral window.

All profiles are pure functions of the cross-section coordinates: ``x`` is
the in-plane (transverse) coordinate and ``y`` the vertical coordinate of
the native-2D mesh, both in um. Concentrations are in cm^-3. The evaluated
node values are what enters DEVSIM as the ``Donors`` / ``Acceptors`` node
solutions behind the ``NetDoping`` node model.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "DopingProfile",
    "GaussianDoping",
    "ImplantDoping",
    "StepDoping",
    "acceptor_donor_concentrations",
    "net_doping_cm3",
]


def _validate_interval(name: str, interval: tuple[float, float] | None) -> None:
    """Reject non-finite or descending (min, max) intervals."""
    if interval is None:
        return
    lo, hi = interval
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"{name} bounds must be finite")
    if hi <= lo:
        raise ValueError(f"{name} must be an ascending (min, max) interval")


def _window_mask(
    values: NDArray[np.float64], interval: tuple[float, float] | None
) -> NDArray[np.float64]:
    """Return a 0/1 mask selecting values inside the closed interval."""
    if interval is None:
        return np.ones_like(values)
    lo, hi = interval
    return ((values >= lo) & (values <= hi)).astype(np.float64)


class _DopingBase(BaseModel):
    """Common fields shared by all doping profile shapes.

    Attributes:
        region: Mesh volume physical-group (layer) name the profile applies
            to. The region must exist on the generated cross-section mesh.
        dopant_type: ``"donor"`` (n-type) or ``"acceptor"`` (p-type).
    """

    model_config = ConfigDict(validate_assignment=True)

    region: str = Field(min_length=1, description="Mesh region (layer) name")
    dopant_type: Literal["donor", "acceptor"]

    def concentration(
        self, x: ArrayLike, y: ArrayLike
    ) -> NDArray[np.float64]:  # pragma: no cover - abstract
        """Evaluate the dopant concentration (cm^-3) at coordinates in um."""
        raise NotImplementedError


class StepDoping(_DopingBase):
    """Uniform doping inside an optional (x, y) box.

    Attributes:
        concentration_cm3: Dopant concentration inside the box (cm^-3).
        x_range: Optional in-plane ``(min, max)`` window in um; unbounded
            when omitted.
        y_range: Optional vertical ``(min, max)`` window in um; unbounded
            when omitted.
    """

    kind: Literal["step"] = "step"
    concentration_cm3: float = Field(gt=0.0)
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        """Windows must be ascending finite intervals."""
        _validate_interval("x_range", self.x_range)
        _validate_interval("y_range", self.y_range)
        return self

    def concentration(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        """Uniform ``concentration_cm3`` inside the box, zero outside."""
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        return np.asarray(
            self.concentration_cm3
            * _window_mask(xa, self.x_range)
            * _window_mask(ya, self.y_range),
            dtype=np.float64,
        )


class GaussianDoping(_DopingBase):
    """Separable Gaussian doping around a center point.

    Each axis with a finite ``sigma`` contributes a Gaussian factor
    ``exp(-(u - u0)^2 / (2 sigma^2))``; an omitted sigma leaves the profile
    uniform along that axis. Optional hard windows clip the tails.

    Attributes:
        peak_cm3: Peak concentration at the center (cm^-3).
        center: ``(x0, y0)`` center in um.
        sigma_x: Optional in-plane standard deviation in um.
        sigma_y: Optional vertical standard deviation in um.
        x_range: Optional in-plane hard window in um.
        y_range: Optional vertical hard window in um.
    """

    kind: Literal["gaussian"] = "gaussian"
    peak_cm3: float = Field(gt=0.0)
    center: tuple[float, float]
    sigma_x: float | None = Field(default=None, gt=0.0)
    sigma_y: float | None = Field(default=None, gt=0.0)
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """At least one sigma must be given; windows must be ascending."""
        if self.sigma_x is None and self.sigma_y is None:
            raise ValueError("GaussianDoping needs sigma_x and/or sigma_y")
        _validate_interval("x_range", self.x_range)
        _validate_interval("y_range", self.y_range)
        return self

    def concentration(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        """Gaussian falloff from the peak, clipped to the hard windows."""
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        x0, y0 = self.center
        result = np.full(np.broadcast(xa, ya).shape, self.peak_cm3)
        if self.sigma_x is not None:
            result = result * np.exp(-((xa - x0) ** 2) / (2.0 * self.sigma_x**2))
        if self.sigma_y is not None:
            result = result * np.exp(-((ya - y0) ** 2) / (2.0 * self.sigma_y**2))
        return np.asarray(
            result * _window_mask(xa, self.x_range) * _window_mask(ya, self.y_range),
            dtype=np.float64,
        )


class ImplantDoping(_DopingBase):
    """Implant-like Gaussian-in-depth profile below a surface.

    The concentration peaks at the projected range ``range_um`` below
    ``surface_y`` and falls off as a Gaussian with standard deviation
    ``straggle_um``: ``N(d) = peak * exp(-(d - Rp)^2 / (2 dRp^2))`` with
    depth ``d = surface_y - y``. Above the surface (``d < 0``) the
    concentration is zero. The lateral extent is a hard window.

    Attributes:
        peak_cm3: Peak concentration at the projected range (cm^-3).
        surface_y: Implant surface y coordinate in um (implant goes to -y).
        range_um: Projected range Rp below the surface in um (>= 0).
        straggle_um: Straggle (depth standard deviation) in um.
        x_range: Optional in-plane hard window in um.
    """

    kind: Literal["implant"] = "implant"
    peak_cm3: float = Field(gt=0.0)
    surface_y: float
    range_um: float = Field(ge=0.0)
    straggle_um: float = Field(gt=0.0)
    x_range: tuple[float, float] | None = None

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        """Lateral window must be an ascending finite interval."""
        _validate_interval("x_range", self.x_range)
        return self

    def concentration(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        """Gaussian in depth below the surface, zero above it."""
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        depth = self.surface_y - ya
        gauss = self.peak_cm3 * np.exp(
            -((depth - self.range_um) ** 2) / (2.0 * self.straggle_um**2)
        )
        inside = (depth >= 0.0).astype(np.float64)
        return np.asarray(
            gauss * inside * _window_mask(xa, self.x_range), dtype=np.float64
        )


DopingProfile = Annotated[
    Union[StepDoping, GaussianDoping, ImplantDoping],  # noqa: UP007
    Field(discriminator="kind"),
]


def acceptor_donor_concentrations(
    profiles: list[StepDoping | GaussianDoping | ImplantDoping],
    x: ArrayLike,
    y: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sum profiles into total (acceptors, donors) concentrations in cm^-3.

    Args:
        profiles: Doping profiles to superpose (any mix of regions —
            filter by region before calling when needed).
        x: In-plane coordinates in um.
        y: Vertical coordinates in um.

    Returns:
        Tuple ``(acceptors, donors)`` of arrays broadcast to the coordinate
        shape.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    shape = np.broadcast(xa, ya).shape
    acceptors = np.zeros(shape)
    donors = np.zeros(shape)
    for profile in profiles:
        values = profile.concentration(xa, ya)
        if profile.dopant_type == "acceptor":
            acceptors = np.asarray(acceptors + values, dtype=np.float64)
        else:
            donors = np.asarray(donors + values, dtype=np.float64)
    return (
        np.asarray(acceptors, dtype=np.float64),
        np.asarray(donors, dtype=np.float64),
    )


def net_doping_cm3(
    profiles: list[StepDoping | GaussianDoping | ImplantDoping],
    x: ArrayLike,
    y: ArrayLike,
) -> NDArray[np.float64]:
    """Net doping ``donors - acceptors`` in cm^-3 at coordinates in um."""
    acceptors, donors = acceptor_donor_concentrations(profiles, x, y)
    return np.asarray(donors - acceptors, dtype=np.float64)
