"""Mesh transfer from the shared native-2D pipeline to DEVSIM.

The BoundaryMode pipeline writes the cross-section mesh in um (msh v2.2).
DEVSIM's silicon physics parameters are cm-based, so the same mesh is
rewritten with coordinates scaled to cm before ``create_gmsh_mesh`` loads
it. Physical groups (region and contact names) are preserved verbatim —
this is a unit conversion, not a second meshing path.
"""

from __future__ import annotations

from pathlib import Path

import meshio

#: Coordinate scale from the gsim mesh unit (um) to DEVSIM's cm.
UM_TO_CM: float = 1e-4


def write_scaled_msh(
    src: str | Path,
    dst: str | Path,
    *,
    scale: float = UM_TO_CM,
) -> Path:
    """Write a copy of a gmsh v2.2 mesh with scaled node coordinates.

    Args:
        src: Source mesh path (msh v2.2, coordinates in um).
        dst: Destination mesh path.
        scale: Multiplicative coordinate scale (default um -> cm).

    Returns:
        The destination path.
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    mesh = meshio.read(str(src))
    mesh.points = mesh.points * scale
    meshio.write(str(dst), mesh, file_format="gmsh22", binary=False)
    return Path(dst)


__all__ = ["UM_TO_CM", "write_scaled_msh"]
