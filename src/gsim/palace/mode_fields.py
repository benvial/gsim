"""The fields of a solved Palace boundary Mode, read back off disk.

Palace's ``BoundaryMode`` text results carry effective indices and
nothing else, so everything a line parameter needs beyond the
propagation constant — the power the Mode carries, the current on its
signal conductor, and therefore its characteristic impedance — used to be
a femwell-Route capability. It need not be: asked to save its Modes,
Palace writes each one's transverse and longitudinal fields to ParaView,
and the Marks-Williams integrals are the same integrals on either
solver's fields.

The fields arrive as nodal values on second-order Lagrange triangles
covering the meshed Cross-section, in um coordinates and SI units. The
integrals here therefore land in the same scale the femwell adapter's do
(:func:`gsim.femwell.adapter.z0_power_current`), which is what lets the
two Routes be compared as numbers rather than as pictures.

One thing has to be undone on the way in. Palace writes the two
transverse components of a boundary Mode in the opposite order to the
coordinates of the points it writes them at, so a field read straight
out of the file comes out mirrored. That is not a matter of taste: on a
perfect conductor the electric field is normal to the surface and the
magnetic field tangential to it, and the raw arrays have both the wrong
way round on every face. :func:`load_boundary_mode_field` swaps them
back, and the boundary conditions come out right on faces of either
orientation.

The sign convention is ``exp(+i omega t)``, matching the rest of gsim.

:class:`BoundaryModeField` and :func:`load_boundary_mode_field` are
re-exported from :mod:`gsim.palace`; the three integrals are not, because
they share a unit contract — um coordinates against SI fields, which
cancels only in the ratio :func:`z0_power_current` takes — and are meant
to be read together with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.constants import mu_0 as MU0  # noqa: N812

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "DEFAULT_PERIMETER_TOL_UM",
    "BoundaryModeField",
    "contour_current",
    "load_boundary_mode_field",
    "power_flux",
    "z0_power_current",
]

#: How far a mesh edge may sit off a conductor's perimeter and still
#: count as on it (um). Palace writes its ParaView coordinates in single
#: precision, which rounds a coordinate tens of um from the origin by
#: about a nanometre; this is loose enough to absorb that and far tighter
#: than any drawn feature.
DEFAULT_PERIMETER_TOL_UM: float = 1e-4

#: Barycentric points and weights of a triangle rule exact to degree 4,
#: which is what the product of two second-order fields asks for.
_QUADRATURE: tuple[tuple[float, float, float, float], ...] = (
    (0.445948490915965, 0.445948490915965, 0.108103018168070, 0.223381589678011),
    (0.445948490915965, 0.108103018168070, 0.445948490915965, 0.223381589678011),
    (0.108103018168070, 0.445948490915965, 0.445948490915965, 0.223381589678011),
    (0.091576213509771, 0.091576213509771, 0.816847572980459, 0.109951743655322),
    (0.091576213509771, 0.816847572980459, 0.091576213509771, 0.109951743655322),
    (0.816847572980459, 0.091576213509771, 0.091576213509771, 0.109951743655322),
)


@dataclass(frozen=True)
class BoundaryModeField:
    """One saved Palace boundary Mode, as fields on its own mesh.

    Attributes:
        points_um: ``(N, 2)`` node coordinates of the Cross-section (um).
        cells: ``(M, 6)`` second-order Lagrange triangles — three corner
            nodes, then the midside node of each edge in turn.
        attribute: ``(M,)`` Palace domain attribute of each triangle,
            which is the mesh Region it belongs to.
        e_t: ``(N, 2)`` transverse electric field (V/m).
        e_n: ``(N,)`` longitudinal electric field (V/m).
        h_t: ``(N, 2)`` transverse magnetic field (A/m).
        h_n: ``(N,)`` longitudinal magnetic field (A/m).
    """

    points_um: NDArray[np.float64]
    cells: NDArray[np.int64]
    attribute: NDArray[np.int64]
    e_t: NDArray[np.complex128]
    e_n: NDArray[np.complex128]
    h_t: NDArray[np.complex128]
    h_n: NDArray[np.complex128]


def _complex_array(grid: Any, name: str) -> NDArray[np.complex128]:
    """Join Palace's ``<name>_real`` / ``<name>_imag`` pair into one array."""
    missing = [
        key for key in (f"{name}_real", f"{name}_imag") if key not in grid.point_data
    ]
    if missing:
        raise ValueError(
            f"The saved mode carries no {', '.join(missing)}; its point data is "
            f"{sorted(grid.point_data)}. Palace writes mode fields only when the "
            "boundary-mode block asks it to save them (set_boundary_mode(save=...))."
        )
    real = np.asarray(grid.point_data[f"{name}_real"], dtype=np.float64)
    imag = np.asarray(grid.point_data[f"{name}_imag"], dtype=np.float64)
    return np.asarray(real + 1j * imag, dtype=np.complex128)


def _transverse(grid: Any, name: str) -> NDArray[np.complex128]:
    """One transverse field, in the order of the points it sits on.

    Palace writes a boundary Mode's two transverse components in the
    opposite order to the coordinates of its points, so the file's first
    component belongs to the second coordinate. Reading them as written
    puts the electric field along a perfect conductor and the magnetic
    field through it, which is the wrong way round on both counts.
    """
    values = _complex_array(grid, name)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(
            f"The saved mode's {name} field has shape {values.shape}, not the "
            "two transverse components of a boundary mode."
        )
    return np.asarray(values[:, ::-1], dtype=np.complex128)


def load_boundary_mode_field(
    source: str | Path | dict[str, Any], *, mode_id: int
) -> BoundaryModeField:
    """Read one saved Mode of a ``BoundaryMode`` solve back off disk.

    Palace writes one ParaView cycle per saved Mode, in the solver's own
    mode order, so Mode ``m`` is cycle ``m``. Reading a Mode the solve
    did not save is an error rather than a silent fallback to another
    one.

    Args:
        source: What ``run_local`` returned, or the simulation
            directory.
        mode_id: Palace's own mode number, 1-based — the ``mode_id`` of
            a :class:`~gsim.modulator.route.PalaceMode`.

    Returns:
        The Mode's fields on the meshed Cross-section.

    Raises:
        FileNotFoundError: When the solve saved no fields at all.
        ValueError: When the requested Mode was not among the saved
            ones, or the mesh is not the second-order triangulation a
            boundary-mode solve writes.
    """
    from gsim.palace.results import load_fields

    grid = load_fields(source, cycle=int(mode_id))
    cell_types = np.unique(np.asarray(grid.celltypes))
    # 22 is VTK_QUADRATIC_TRIANGLE, 69 its Lagrange spelling; both carry
    # three corner nodes then three midside ones. The Lagrange spelling
    # has no fixed node count of its own, so the connectivity is read
    # through the offsets rather than through pyvista's cells_dict.
    counts = np.diff(np.asarray(grid.offset, dtype=np.int64))
    if cell_types.size != 1 or int(cell_types[0]) not in (22, 69):
        raise ValueError(
            f"The saved mode's mesh has cell types {cell_types.tolist()}, not the "
            "second-order triangles a boundary-mode solve writes."
        )
    if counts.size == 0 or not np.all(counts == 6):
        raise ValueError(
            "The saved mode's triangles carry "
            f"{sorted(set(counts.tolist()))} nodes each, not the six a "
            "second-order triangle carries."
        )
    cells = np.asarray(grid.cell_connectivity, dtype=np.int64).reshape(-1, 6)

    b_n = _complex_array(grid, "B")
    return BoundaryModeField(
        points_um=np.asarray(grid.points[:, :2], dtype=np.float64),
        cells=cells,
        attribute=np.asarray(grid.cell_data["attribute"], dtype=np.int64),
        e_t=_transverse(grid, "E"),
        e_n=_complex_array(grid, "En"),
        h_t=np.asarray(_transverse(grid, "Bt") / MU0, dtype=np.complex128),
        h_n=np.asarray(b_n / MU0, dtype=np.complex128),
    )


def _triangle_areas(field: BoundaryModeField) -> NDArray[np.float64]:
    """Signed-free area of each triangle from its corner nodes (um^2)."""
    corners = field.points_um[field.cells[:, :3]]
    edge_a = corners[:, 1] - corners[:, 0]
    edge_b = corners[:, 2] - corners[:, 0]
    cross = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    return np.asarray(0.5 * np.abs(cross), dtype=np.float64)


def _shape_values(l0: float, l1: float, l2: float) -> NDArray[np.float64]:
    """The six second-order shape functions at one barycentric point."""
    return np.asarray(
        [
            l0 * (2.0 * l0 - 1.0),
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            4.0 * l0 * l1,
            4.0 * l1 * l2,
            4.0 * l2 * l0,
        ],
        dtype=np.float64,
    )


def power_flux(field: BoundaryModeField) -> complex:
    """Complex power the Mode carries along the propagation direction.

    ``P = (1/2) integral (E_t x H_t*) . z dA`` over the whole meshed
    Cross-section — the numerator of the Marks-Williams power-current
    impedance, and the same integral the femwell Route assembles.

    Args:
        field: The Mode's saved fields.

    Returns:
        The complex power, in the um-area scale
        :func:`z0_power_current` divides out.
    """
    areas = _triangle_areas(field)
    e_t = field.e_t[field.cells]
    h_t = np.conj(field.h_t[field.cells])
    total = np.zeros(field.cells.shape[0], dtype=np.complex128)
    for l0, l1, l2, weight in _QUADRATURE:
        shape = _shape_values(l0, l1, l2)
        e_q = np.einsum("n,cnk->ck", shape, e_t)
        h_q = np.einsum("n,cnk->ck", shape, h_t)
        total += weight * (e_q[:, 0] * h_q[:, 1] - e_q[:, 1] * h_q[:, 0])
    return complex(0.5 * np.sum(total * areas))


def _rectangle_side_normals(
    midpoints: NDArray[np.float64],
    *,
    h_span: tuple[float, float],
    v_span: tuple[float, float],
    tol_um: float,
) -> NDArray[np.float64]:
    """Outward normal of the rectangle side each midpoint sits on.

    Rows for midpoints that sit on no side come back as zeros, which
    contribute nothing to the contour integral.
    """
    normals = np.zeros_like(midpoints)
    on_low_h = np.abs(midpoints[:, 0] - h_span[0]) <= tol_um
    on_high_h = np.abs(midpoints[:, 0] - h_span[1]) <= tol_um
    on_low_v = np.abs(midpoints[:, 1] - v_span[0]) <= tol_um
    on_high_v = np.abs(midpoints[:, 1] - v_span[1]) <= tol_um
    normals[on_low_h] = (-1.0, 0.0)
    normals[on_high_h] = (1.0, 0.0)
    normals[on_low_v] = (0.0, -1.0)
    normals[on_high_v] = (0.0, 1.0)
    return normals


def contour_current(
    field: BoundaryModeField,
    *,
    h_span: tuple[float, float],
    v_span: tuple[float, float],
    tol_um: float = DEFAULT_PERIMETER_TOL_UM,
) -> complex:
    """Ampere's contour integral of ``H`` around one rectangular conductor.

    ``I = contour integral of H . dl``, written with the side's outward
    normal as ``(n x H) . z``. This is the current definition a perfect
    conductor has — it carries no volume current to integrate — and it
    reads the same enclosed current for a conductor meshed as a Region.

    Each mesh edge lying on the rectangle's perimeter contributes; edges
    shared by two triangles (which is how a conductor meshed as a Region
    presents its outline) are counted once.

    Args:
        field: The Mode's saved fields.
        h_span: ``(min, max)`` of the conductor along the first
            coordinate (um).
        v_span: ``(min, max)`` along the second coordinate (um).
        tol_um: How far an edge may sit off the perimeter and still
            count as on it (um).

    Returns:
        The enclosed current, in the um-length scale
        :func:`z0_power_current` divides out.

    Raises:
        ValueError: When no mesh edge lies on the rectangle, so the
            conductor named is not on this Cross-section.
    """
    # Edge n of a second-order triangle runs between corners n and n+1
    # with its midside node at 3 + n.
    edges = np.concatenate(
        [
            field.cells[:, [0, 1, 3]],
            field.cells[:, [1, 2, 4]],
            field.cells[:, [2, 0, 5]],
        ]
    )
    ends = field.points_um[edges[:, :2]]
    midpoints = ends.mean(axis=1)
    inside = (
        (midpoints[:, 0] >= h_span[0] - tol_um)
        & (midpoints[:, 0] <= h_span[1] + tol_um)
        & (midpoints[:, 1] >= v_span[0] - tol_um)
        & (midpoints[:, 1] <= v_span[1] + tol_um)
    )
    normals = _rectangle_side_normals(
        midpoints, h_span=h_span, v_span=v_span, tol_um=tol_um
    )
    on_perimeter = inside & np.any(normals != 0.0, axis=1)
    if not np.any(on_perimeter):
        raise ValueError(
            f"No mesh edge lies on the rectangle h={h_span}, v={v_span}, so no "
            "conductor of the cross-section has that outline."
        )

    edges = edges[on_perimeter]
    normals = normals[on_perimeter]
    ends = ends[on_perimeter]
    # One conductor outline, not two: a Region's outline is an edge of
    # the triangles on both sides of it and would otherwise count twice.
    _unique, first = np.unique(np.sort(edges[:, :2], axis=1), axis=0, return_index=True)
    edges, normals, ends = edges[first], normals[first], ends[first]

    lengths = np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1)
    h_edge = field.h_t[edges]
    # Simpson over each edge's two ends and its midside node, which is
    # exact for the second-order field they interpolate.
    weights = np.asarray([1.0 / 6.0, 1.0 / 6.0, 4.0 / 6.0])
    h_mean = np.einsum("n,cnk->ck", weights, h_edge)
    tangential = normals[:, 0] * h_mean[:, 1] - normals[:, 1] * h_mean[:, 0]
    return complex(np.sum(tangential * lengths))


def z0_power_current(
    field: BoundaryModeField,
    *,
    h_span: tuple[float, float],
    v_span: tuple[float, float],
    tol_um: float = DEFAULT_PERIMETER_TOL_UM,
) -> complex:
    """Marks-Williams power-current impedance of a saved Palace Mode.

    ``Z_0 = 2 P / |I|^2`` with :func:`power_flux` over the whole
    Cross-section and :func:`contour_current` around the signal
    conductor. The ratio is invariant to the mode's field normalization,
    and the um coordinates cancel between the area and the length
    integral, so the result is in ohms.

    Args:
        field: The Mode's saved fields.
        h_span: ``(min, max)`` of the signal conductor along the first
            coordinate (um).
        v_span: ``(min, max)`` along the second coordinate (um).
        tol_um: Perimeter tolerance passed to :func:`contour_current`.

    Returns:
        The complex characteristic impedance in ohms.

    Raises:
        ValueError: When the contour integral comes out zero, which
            means the Mode carries no current on that conductor.
    """
    current = contour_current(field, h_span=h_span, v_span=v_span, tol_um=tol_um)
    if current == 0:
        raise ValueError(
            "The saved mode carries no current around the signal conductor, so "
            "it has no power-current impedance."
        )
    return complex(2.0 * power_flux(field) / (abs(current) ** 2))
