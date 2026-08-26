"""Cross-section specification models for Palace 2D mode simulations."""

from __future__ import annotations

import math
import re
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

_PLANE_SPEC_RE = re.compile(
    r"^\s*([xXyYzZ])\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


class CrossSectionPlaneConfig(BaseModel):
    """Axis-aligned cross-section plane for 2D mode extraction.

    An optional window clips the meshed 2D domain to a sub-region of the
    component cross-section instead of the full bounding box plus margins.
    This lets one component feed differently sized per-solver domains (full
    extent for RF, a small box around the rib for optics, the doped slab for
    charge transport).

    Attributes:
        axis: Plane normal axis ("x", "y", or "z").
        value: Plane coordinate in microns.
        window: Optional in-plane interval ``(min, max)`` in um clipping the
            transverse extent (y for an x-plane, x for a y-plane).
        window_z: Optional z interval ``(min, max)`` in um clipping the
            vertical extent.
    """

    model_config = ConfigDict(validate_assignment=True)

    axis: Literal["x", "y", "z"]
    value: float = Field(description="Plane coordinate in um")
    window: tuple[float, float] | None = Field(
        default=None, description="In-plane clip interval (min, max) in um"
    )
    window_z: tuple[float, float] | None = Field(
        default=None, description="Vertical clip interval (min, max) in um"
    )

    @model_validator(mode="after")
    def validate_value(self) -> Self:
        """Ensure the plane coordinate is finite and windows are ascending."""
        if not math.isfinite(self.value):
            raise ValueError("cross-section value must be finite")
        for name, interval in (("window", self.window), ("window_z", self.window_z)):
            if interval is None:
                continue
            lo, hi = interval
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError(f"{name} bounds must be finite")
            if hi <= lo:
                raise ValueError(f"{name} must be an ascending (min, max) interval")
        return self

    @classmethod
    def from_spec(cls, spec: str) -> Self:
        """Parse a string specification like ``x=0`` or ``y=100``."""
        match = _PLANE_SPEC_RE.match(spec)
        if match is None:
            raise ValueError(
                "Invalid plane spec. Use 'x=<value>', 'y=<value>', or 'z=<value>'"
            )
        axis = cast(Literal["x", "y", "z"], match.group(1).lower())
        value = float(match.group(2))
        return cls(axis=axis, value=value)

    @property
    def spec(self) -> str:
        """Return the normalized string representation (e.g., ``x=0.0``)."""
        return f"{self.axis}={self.value}"


__all__ = ["CrossSectionPlaneConfig"]
