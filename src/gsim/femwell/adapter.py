"""femwell adapter on the shared native-2D cross-section mesh.

Loads the exact msh v2.2 mesh the BoundaryMode pipeline generates into
skfem/femwell, with two epsilon paths:

- **piecewise-constant**: every named 2D physical group gets the complex
  relative permittivity of its stack material at the target wavelength or
  frequency (:func:`epsilon_by_region`) — the configuration Palace can
  also express, used for cross-solver validation.
- **continuous**: carrier-derived eps(x, y) node values (e.g. from a
  :class:`gsim.tcad.results.CarrierMap` through
  :func:`gsim.common.carriers.permittivity_perturbation`) projected onto a
  piecewise-element basis (:func:`elementwise_epsilon`) — the configuration
  Palace cannot express.

Both epsilon paths are pure meshio/scipy functions testable without the
femwell runtime; only :func:`solve_modes` needs femwell/skfem installed
(``pip install 'gsim[femwell]'``).

The sign convention is ``exp(+i omega t)``: lossy media have
``Im(eps) < 0``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import meshio
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import epsilon_0 as EPS0  # noqa: N812
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.common.stack.materials import (
    MaterialProperties,
    ResolvedMaterial,
    resolve_material_at_wavelength,
)
from gsim.femwell.runtime import require_femwell, require_skfem

if TYPE_CHECKING:
    from gsim.common.stack.extractor import LayerStack

__all__ = [
    "elementwise_epsilon",
    "epsilon_by_region",
    "region_material_map",
    "solve_modes",
    "z0_power_current",
]


def _complex_permittivity(
    resolved: ResolvedMaterial, *, frequency_hz: float
) -> complex:
    """Complex relative permittivity in the exp(+i omega t) convention."""
    eps_re = resolved.permittivity_scalar
    if eps_re is None:
        eps_re = 1.0
    loss_tangent = (
        resolved.loss_tangent_scalar
        if resolved.loss_tangent_scalar is not None
        else 0.0
    )
    sigma = (
        resolved.conductivity_scalar
        if resolved.conductivity_scalar is not None
        else 0.0
    )
    omega = 2.0 * np.pi * frequency_hz
    return complex(eps_re * (1.0 - 1j * loss_tangent) - 1j * sigma / (omega * EPS0))


def _group_tags_2d(mesh: meshio.Mesh) -> dict[str, int]:
    """Map dim-2 physical-group names to their gmsh tags."""
    return {
        str(name): int(np.asarray(data)[0])
        for name, data in mesh.field_data.items()
        if int(np.asarray(data)[1]) == 2
    }


def region_material_map(stack: LayerStack, regions: list[str]) -> dict[str, str]:
    """Map mesh region (physical-group) names to stack material names.

    Regions matching a stack layer use that layer's material; other
    regions (background media like ``sio2`` / ``air``, or generated strip
    regions whose material shares the region name) map to their own name.

    Args:
        stack: The layer stack the mesh was generated from.
        regions: 2D physical-group names on the mesh.

    Returns:
        ``{region_name: material_name}``.
    """
    mapping: dict[str, str] = {}
    for region in regions:
        layer = stack.layers.get(region)
        mapping[region] = layer.material if layer is not None else region
    return mapping


def epsilon_by_region(
    mesh: meshio.Mesh | str | Path,
    stack: LayerStack,
    *,
    wavelength_um: float | None = None,
    frequency_hz: float | None = None,
    overrides: dict[str, MaterialProperties] | None = None,
) -> dict[str, complex]:
    """Resolve the complex permittivity of every 2D mesh region.

    Materials come from the stack's materials (the same database the
    Palace config generator uses), evaluated at the target wavelength or
    frequency, so both solvers see identical piecewise-constant epsilon.

    Args:
        mesh: The shared msh v2.2 mesh (path or loaded meshio mesh).
        stack: Layer stack the mesh was generated from (region-to-material
            mapping and material property source).
        wavelength_um: Target vacuum wavelength in um (optical).
        frequency_hz: Target frequency in Hz (RF). Exactly one of
            ``wavelength_um`` / ``frequency_hz`` must be given.
        overrides: Optional material-property overrides by material name.

    Returns:
        ``{region_name: complex_relative_permittivity}`` for every dim-2
        physical group (``exp(+i omega t)``: lossy means ``Im < 0``).
    """
    if (wavelength_um is None) == (frequency_hz is None):
        raise ValueError("Give exactly one of wavelength_um or frequency_hz.")
    if wavelength_um is not None:
        frequency = C0 / (wavelength_um * 1e-6)
        wavelength = wavelength_um
    else:
        assert frequency_hz is not None  # noqa: S101 - guarded above
        frequency = float(frequency_hz)
        wavelength = C0 / frequency * 1e6

    if not isinstance(mesh, meshio.Mesh):
        mesh = meshio.read(str(mesh))
    regions = list(_group_tags_2d(mesh))
    if not regions:
        raise ValueError("Mesh has no 2D physical groups.")

    materials = region_material_map(stack, regions)
    merged_overrides: dict[str, MaterialProperties] = {}
    for name, props in (stack.materials or {}).items():
        merged_overrides[name] = (
            props
            if isinstance(props, MaterialProperties)
            else MaterialProperties.model_validate(props)
        )
    merged_overrides.update(overrides or {})

    result: dict[str, complex] = {}
    for region in regions:
        material = materials[region]
        resolved = resolve_material_at_wavelength(
            material, wavelength, overrides=merged_overrides
        )
        if resolved is None:
            raise ValueError(
                f"Region '{region}' maps to material '{material}' which is "
                "not resolvable from the stack materials or the built-in "
                "database."
            )
        result[region] = _complex_permittivity(resolved, frequency_hz=frequency)
    return result


def elementwise_epsilon(
    mesh: meshio.Mesh | str | Path,
    x_um: ArrayLike,
    y_um: ArrayLike,
    eps_values: ArrayLike,
    *,
    fill: complex | None = None,
) -> NDArray[np.complex128]:
    """Project scattered eps(x, y) samples onto per-element (P0) values.

    Each triangle of the mesh gets the value of the linear interpolant of
    the samples at its centroid — the continuous-epsilon path Palace
    cannot express. Centroids outside the convex hull of the samples fall
    back to nearest-neighbour (or ``fill`` when given).

    Args:
        mesh: The shared msh v2.2 mesh (path or loaded meshio mesh).
        x_um: Sample x coordinates in um (e.g. charge-solve nodes).
        y_um: Sample y coordinates in um.
        eps_values: Complex permittivity samples at those points.
        fill: Value for centroids outside the sample hull; defaults to
            nearest-neighbour extrapolation.

    Returns:
        Complex epsilon per triangle, in the mesh's triangle order.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    if not isinstance(mesh, meshio.Mesh):
        mesh = meshio.read(str(mesh))
    triangles = [block.data for block in mesh.cells if block.type == "triangle"]
    if not triangles:
        raise ValueError("Mesh has no triangle elements.")
    tris = np.vstack(triangles)
    centroids = mesh.points[tris][:, :, :2].mean(axis=1)

    points = np.column_stack(
        [
            np.asarray(x_um, dtype=np.float64).ravel(),
            np.asarray(y_um, dtype=np.float64).ravel(),
        ]
    )
    values = np.asarray(eps_values, dtype=np.complex128).ravel()
    if points.shape[0] != values.size:
        raise ValueError("x_um, y_um and eps_values must have the same length.")
    if points.shape[0] < 3:
        raise ValueError("At least three sample points are required.")

    linear = LinearNDInterpolator(points, values)
    result = np.asarray(linear(centroids), dtype=np.complex128)
    missing = np.isnan(result.real)
    if np.any(missing):
        if fill is not None:
            result[missing] = fill
        else:
            nearest = NearestNDInterpolator(points, values)
            result[missing] = np.asarray(
                nearest(centroids[missing]), dtype=np.complex128
            )
    return result


def solve_modes(
    msh_path: str | Path,
    *,
    epsilon: dict[str, complex] | ArrayLike,
    wavelength_um: float,
    num_modes: int = 1,
    order: int = 1,
    metallic_boundaries: bool = False,
    n_guess: float | None = None,
) -> Any:
    """Solve waveguide modes with femwell on the shared mesh.

    Requires the femwell runtime (``pip install 'gsim[femwell]'``).

    Args:
        msh_path: Path of the shared msh v2.2 mesh (um coordinates).
        epsilon: Either ``{region_name: eps}`` (piecewise-constant, from
            :func:`epsilon_by_region`) or per-triangle values (continuous,
            from :func:`elementwise_epsilon`).
        wavelength_um: Vacuum wavelength in um. For RF, pass the free-space
            wavelength of the target frequency (``c0 / f`` in um).
        num_modes: Number of modes to compute.
        order: Finite element order of the mode solve.
        metallic_boundaries: Enforce PEC on the outer boundary.
        n_guess: Effective-index guess centering the eigenvalue search.
            femwell's default guess tracks the largest permittivity, which
            for RF materials with big conductive ``|Im(eps)|`` can put the
            shift far from the physical quasi-TEM mode; pass an explicit
            guess (e.g. the expected slow-wave index) there.

    Returns:
        The femwell ``Modes`` result (each mode carries ``n_eff``).

    Raises:
        ValueError: If the mesh has no triangles, if a region in ``epsilon``
            is not on the mesh, if a 2D region on the mesh is missing from
            ``epsilon``, or if per-element values do not match the element
            count.
    """
    require_femwell()
    skfem = require_skfem()
    from femwell.maxwell.waveguide import compute_modes

    # Build the skfem mesh and the region-to-element mapping directly from
    # meshio: skfem's own msh subdomain parsing is version-fragile, and the
    # explicit construction keeps the meshio triangle order == skfem element
    # order (which the per-element epsilon path relies on).
    mio = meshio.read(str(msh_path))
    tri_blocks = [
        (block.data, np.asarray(phys))
        for block, phys in zip(
            mio.cells, mio.cell_data.get("gmsh:physical", []), strict=False
        )
        if block.type == "triangle"
    ]
    if not tri_blocks:
        raise ValueError("Mesh has no triangle elements.")
    tris = np.vstack([data for data, _phys in tri_blocks])
    phys_tags = np.concatenate([phys for _data, phys in tri_blocks])

    # Drop points no triangle references (e.g. nodes only line/contact
    # groups use): they would become zero rows in the eigenproblem and make
    # the shift-invert factorization exactly singular.
    points = mio.points
    used = np.unique(tris)
    if used.size != points.shape[0]:
        remap = np.full(points.shape[0], -1, dtype=np.int64)
        remap[used] = np.arange(used.size)
        tris = remap[tris]
        points = points[used]

    mesh = skfem.MeshTri(
        np.ascontiguousarray(points[:, :2].T, dtype=np.float64),
        np.ascontiguousarray(tris.T, dtype=np.int64),
    )
    basis0 = skfem.Basis(mesh, skfem.ElementTriP0())

    if isinstance(epsilon, dict):
        group_tags = _group_tags_2d(mio)
        eps = basis0.zeros(dtype=complex)
        # cast: ty cannot narrow the dict half of the union on its own.
        eps_map = cast("dict[str, complex]", epsilon)  # type: ignore[redundant-cast]
        for region, value in eps_map.items():
            if region not in group_tags:
                raise ValueError(
                    f"Region '{region}' not found on the mesh. "
                    f"Available subdomains: {sorted(group_tags)}"
                )
            # ElementTriP0: one dof per element, in element order.
            eps[phys_tags == group_tags[region]] = value
        # Every element must get a permittivity: a region left out of the map
        # keeps eps = 0, which is not a material at all -- it either makes the
        # shift-invert factorization singular or returns modes of a structure
        # the caller never described.
        mapped = {group_tags[region] for region in eps_map}
        present = {int(tag) for tag in np.unique(phys_tags)}
        missing = sorted(present - mapped)
        if missing:
            tag_names = {tag: name for name, tag in group_tags.items()}
            named = [
                f"'{tag_names[tag]}'" if tag in tag_names else f"tag {tag}"
                for tag in missing
            ]
            raise ValueError(
                f"No permittivity given for mesh region(s) {', '.join(named)}. "
                "Every 2D region on the mesh must appear in the epsilon map."
            )
    else:
        eps = np.asarray(epsilon, dtype=np.complex128)
        if eps.size != basis0.N:
            raise ValueError(
                f"Per-element epsilon has {eps.size} values but the mesh "
                f"has {basis0.N} elements."
            )

    return compute_modes(
        basis0,
        eps,
        wavelength=wavelength_um,
        num_modes=num_modes,
        order=order,
        metallic_boundaries=metallic_boundaries,
        n_guess=n_guess,
    )


def z0_power_current(
    mode: Any,
    *,
    frequency_hz: float,
    sigma_s_per_m: ArrayLike | None = None,
    current_elements: ArrayLike | None = None,
) -> complex:
    """Marks-Williams power-current characteristic impedance of an RF mode.

    ``Z_0 = 2 P / |I|^2`` with the complex Poynting flux
    ``P = (1/2) integral (E_t x H_t*) . z dA`` over the whole cross-section
    and the longitudinal conduction current ``I = integral sigma E_z dA``
    over the signal conductor. The ratio is invariant to the mode's field
    normalization; the mesh coordinates are in um and ``sigma`` in S/m, the
    unit conversion is internal.

    Args:
        mode: A femwell ``Mode`` from :func:`solve_modes` (fields solved
            with the complex permittivity that encodes the conductivity,
            ``exp(+i omega t)``: ``Im(eps) < 0``).
        frequency_hz: RF frequency of the solve in Hz.
        sigma_s_per_m: Conductivity per mesh element in S/m. Defaults to
            the conduction profile implied by the mode's own epsilon:
            ``sigma = -Im(eps_r) omega eps_0`` where negative.
        current_elements: Element indices (or boolean mask) carrying the
            signal current. Defaults to every element with positive
            conductivity — valid only when the mesh has a single signal
            conductor; on a two-conductor line (e.g. CPS electrodes) the
            signal and return currents nearly cancel in that sum, so pass
            the elements of one conductor explicitly.

    Returns:
        Complex characteristic impedance in ohms.
    """
    skfem = require_skfem()
    from skfem.helpers import cross

    omega = 2.0 * np.pi * float(frequency_hz)
    eps = np.asarray(mode.epsilon_r, dtype=np.complex128)
    if sigma_s_per_m is None:
        sigma = np.where(eps.imag < 0.0, -eps.imag, 0.0) * omega * EPS0
    else:
        sigma = np.asarray(sigma_s_per_m, dtype=np.float64)
        if sigma.shape != eps.shape:
            raise ValueError(
                f"sigma_s_per_m has shape {sigma.shape} but the mesh has "
                f"{eps.shape[0]} elements."
            )

    if current_elements is None:
        elements = np.flatnonzero(sigma > 0.0)
    else:
        elements = np.atleast_1d(np.asarray(current_elements))
        if elements.dtype == bool:
            elements = np.flatnonzero(elements)
    if elements.size == 0:
        raise ValueError(
            "No conductive elements to integrate the current over; the mode "
            "was solved without conductive regions (all Im(eps) >= 0)."
        )

    basis = mode.basis

    @skfem.Functional(dtype=np.complex128)  # type: ignore[untyped-decorator]
    def _power_form(w: Any) -> Any:
        return cross(w["E"][0], np.conj(w["H"][0]))

    power = 0.5 * _power_form.assemble(
        basis,
        E=basis.interpolate(mode.E),
        H=basis.interpolate(mode.H),
    )

    @skfem.Functional(dtype=np.complex128)  # type: ignore[untyped-decorator]
    def _current_form(w: Any) -> Any:
        return w["sigma"] * w["E"][1]

    sub = basis.with_elements(elements)
    sub_sigma = mode.basis_epsilon_r.with_elements(elements)
    # Mesh coordinates are um: S/m -> S/um so the um^2 area integral is in A.
    current = 1e-6 * _current_form.assemble(
        sub,
        E=sub.interpolate(mode.E),
        sigma=sub_sigma.interpolate(np.asarray(sigma, dtype=np.float64)),
    )
    if current == 0:
        raise ValueError("Zero longitudinal current over the selected elements.")
    return complex(2.0 * power / (abs(current) ** 2))
