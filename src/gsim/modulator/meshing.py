"""What every meshing Stage asks of the mesh pipeline, in one place.

The charge, optical and RF Stages all mesh a Cross-section, and all three
want the same background region and much the same element sizes. Kept as
three literals they drifted apart silently; kept here, a Stage overrides
only what its own physics needs — the charge Stage its finer elements —
and a user still overrides anything, per Stage, through the Stage's own
``mesh=`` and ``airbox=`` settings.
"""

from __future__ import annotations

from typing import Any

__all__ = ["STAGE_AIRBOX", "STAGE_MESH", "stage_airbox", "stage_mesh"]

#: Mesh-pipeline arguments a Stage meshes with unless it says otherwise.
STAGE_MESH: dict[str, Any] = {
    "preset": "coarse",
    "refined_mesh_size": 0.05,
    "max_mesh_size": 40.0,
    "verbose": False,
}

#: Background region a Stage puts around its clipped domain.
STAGE_AIRBOX: dict[str, Any] = {
    "margin_x": 2.0,
    "margin_y": 2.0,
    "z_above": 1.5,
    "z_below": 1.0,
    "material": "sio2",
}


def stage_mesh(**overrides: Any) -> dict[str, Any]:
    """The shared mesh settings, with a Stage's own changes applied.

    Args:
        **overrides: Settings this Stage differs on.

    Returns:
        A fresh dict, so no Stage can mutate another's defaults.
    """
    return {**STAGE_MESH, **overrides}


def stage_airbox(**overrides: Any) -> dict[str, Any]:
    """The shared airbox settings, with a Stage's own changes applied.

    Args:
        **overrides: Settings this Stage differs on.

    Returns:
        A fresh dict, so no Stage can mutate another's defaults.
    """
    return {**STAGE_AIRBOX, **overrides}
