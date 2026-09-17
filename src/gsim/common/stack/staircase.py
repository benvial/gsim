"""Staircase route: carrier maps as piecewise-constant Palace domains.

Palace only accepts piecewise-constant materials per mesh domain, so a
continuously varying carrier distribution is represented as N adjacent
strips along the junction axis. Each strip becomes a patterned-dielectric
region through the same ``Layer``/``MaterialProperties`` machinery the
doped cross-section builder uses, so the result plugs straight into
``build_doped_cross_section(doping=...)`` and the native ``BoundaryMode``
solver.

Everything that turns a Carrier map into per-Strip averages lives here:

- :func:`staircase_profile` bins a sampled one-dimensional profile into N
  equal-width Strips, each carrying the exact average of the
  piecewise-linear interpolant over it.
- :func:`strip_averages_from_nodes` reduces a scattered two-dimensional
  node cloud (e.g. a :class:`gsim.tcad.results.CarrierMap`) to that
  one-dimensional profile first — the tested, reusable mesh-transfer
  step — and bins it.
- :func:`build_staircase_cross_section` does the whole job in one call: a
  Carrier map, a strip count and the Junction extent in, a meshable
  Staircase cross-section out — the strip Regions, their material
  response for *both* EM Stages, and the flanking electrodes. The
  drawing of the Strips and the material of each one are its own
  private steps.
- :func:`surroundings_from_section` supplies what the Strips are *not*:
  every other Region of the drawn Cross-section, cut against the Strip
  footprint, so the Staircase is the drawn waveguide with its doped
  silicon replaced by Strips rather than a bare silicon wire in the
  background medium. A Staircase built without them answers for a
  different guide, and the difference does not shrink with strip count.

``n_strips=1`` recovers the uniform-strip model: one rectangle spanning
the window carrying the profile average.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gsim.common.carriers import (
    DEFAULT_MU_N_CM2,
    DEFAULT_MU_P_CM2,
    PlasmaDispersionModel,
    carrier_absorption_cm,
    carrier_conductivity,
    carrier_index_shift,
    permittivity_perturbation,
)
from gsim.common.stack.materials import MaterialProperties, make_doped_materials

if TYPE_CHECKING:
    import gdsfactory as gf

    from gsim.common.stack.extractor import Layer, LayerStack

__all__ = [
    "COLUMN_TOL_FRACTION",
    "DEFAULT_ELECTRODES",
    "DEFAULT_SI_INDEX",
    "DEFAULT_STRIP_LAYER",
    "DEFAULT_SURROUND_LAYER",
    "STRIP_LENGTH_UM",
    "ConductorModel",
    "ElectrodeSpec",
    "StaircaseCrossSection",
    "SurroundingRegion",
    "build_staircase_cross_section",
    "carrier_map_extent",
    "staircase_profile",
    "strip_averages_from_nodes",
    "surroundings_from_section",
]

#: Unperturbed silicon refractive index near 1.55 um: what a Strip
#: carries before the carriers move it, when the drawn stack cannot say
#: what its own silicon is.
DEFAULT_SI_INDEX: float = 3.4757

#: ``(layer, datatype)`` Strip 0 is drawn on, Strip ``i`` taking
#: ``datatype + i``. Deliberately outside the range gdsfactory's generic
#: PDK uses: a Strip landing on a PDK metal or via layer is resolved as
#: that conductor, which a solver reading the stack (Palace) then honours
#: and one reading only the mesh regions (femwell) does not.
DEFAULT_STRIP_LAYER: tuple[int, int] = (300, 0)

#: ``(layer, datatype)`` the first surrounding Region is drawn on, the
#: next taking ``datatype + 1``. Outside the generic PDK's own layers for
#: the same reason :data:`DEFAULT_STRIP_LAYER` is, and distinct from it so
#: a Strip and a surrounding Region never share a GDS layer.
DEFAULT_SURROUND_LAYER: tuple[int, int] = (310, 0)

#: Extents closer than this count as coincident when a surrounding
#: Region is cut against the Strip footprint (um).
SURROUND_TOL_UM: float = 1e-9

#: Fraction of the sampled extent within which two nodes count as one
#: column of the mesh, when no explicit tolerance is given. Node columns
#: are what a 2D cloud is averaged over before it is binned into Strips.
COLUMN_TOL_FRACTION: float = 1e-6

#: Drawn length of a Staircase along the propagation direction (um).
#: The Cross-section is invariant along it, so it is not a setting;
#: a Stage meshing a Staircase cuts through the middle of it.
STRIP_LENGTH_UM: float = 10.0


def staircase_profile(
    h: ArrayLike,
    values: ArrayLike,
    *,
    n_bins: int,
    h_min: float | None = None,
    h_max: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bin a sampled 1D profile into N equal-width piecewise-constant strips.

    The samples are interpreted as a piecewise-linear function of ``h``; each
    strip value is the exact average of that interpolant over the strip, so
    ``n_bins=1`` recovers the exact mean of the profile and increasing
    ``n_bins`` converges to the continuous profile.

    Args:
        h: Sample coordinates along the binning axis (um), any order.
        values: Sample values (e.g. carrier concentration, sigma, Delta n).
        n_bins: Number of strips (>= 1).
        h_min: Window start; defaults to ``min(h)``.
        h_max: Window end; defaults to ``max(h)``.

    Returns:
        ``(edges, means)`` — strip edges of length ``n_bins + 1`` and the
        per-strip averages of length ``n_bins``.
    """
    h_arr = np.asarray(h, dtype=np.float64).ravel()
    v_arr = np.asarray(values, dtype=np.float64).ravel()
    if h_arr.size != v_arr.size:
        raise ValueError("h and values must have the same length.")
    if h_arr.size < 2:
        raise ValueError("At least two samples are required.")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")

    order = np.argsort(h_arr)
    h_arr = h_arr[order]
    v_arr = v_arr[order]

    lo = float(h_arr[0]) if h_min is None else float(h_min)
    hi = float(h_arr[-1]) if h_max is None else float(h_max)
    if hi <= lo:
        raise ValueError("h_max must exceed h_min.")

    edges = np.asarray(np.linspace(lo, hi, n_bins + 1), dtype=np.float64)

    # Exact average of the piecewise-linear interpolant over each strip via
    # its antiderivative sampled with cumulative trapezoids.
    dense = np.union1d(edges, h_arr[(h_arr > lo) & (h_arr < hi)])
    dense_v = np.interp(dense, h_arr, v_arr)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (dense_v[1:] + dense_v[:-1]) * np.diff(dense)))
    )
    edge_integrals = np.interp(edges, dense, cumulative)
    means = np.asarray(np.diff(edge_integrals) / np.diff(edges), dtype=np.float64)
    return edges, means


def strip_averages_from_nodes(
    h_um: ArrayLike,
    values: ArrayLike,
    *,
    n_strips: int,
    h_min: float | None = None,
    h_max: float | None = None,
    v_um: ArrayLike | None = None,
    v_range: tuple[float, float] | None = None,
    column_tol_um: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average scattered node values into N strips along the junction axis.

    Node values (e.g. carrier concentrations on the charge-solve mesh) are
    reduced to a 1D profile of ``h`` — nodes sharing a coordinate are
    averaged, so a band selected with ``v_range`` contributes all of its
    rows and not just one — and binned with
    :func:`staircase_profile`, whose strip values are
    exact averages of the piecewise-linear interpolant — so ``n_strips=1``
    recovers the profile mean and increasing N converges to the continuous
    profile.

    Args:
        h_um: Node coordinates along the junction (binning) axis in um.
        values: Node values (same length as ``h_um``).
        n_strips: Number of strips (>= 1).
        h_min: Window start along the axis; defaults to ``min(h_um)``.
        h_max: Window end along the axis; defaults to ``max(h_um)``.
        v_um: Optional node coordinates transverse to the binning axis
            (e.g. z); used with ``v_range`` to select a band of a 2D node
            cloud before binning.
        v_range: Optional ``(min, max)`` band in the ``v_um`` coordinate.
        column_tol_um: Nodes whose ``h`` agree to within this (um) are one
            column of the mesh and are averaged together; defaults to
            :data:`COLUMN_TOL_FRACTION` of the sampled extent.

    Returns:
        ``(edges, means)`` — strip edges of length ``n_strips + 1`` and
        per-strip averages of length ``n_strips``.
    """
    h_arr: NDArray[np.float64] = np.asarray(h_um, dtype=np.float64).ravel()
    v_arr: NDArray[np.float64] = np.asarray(values, dtype=np.float64).ravel()
    if h_arr.size != v_arr.size:
        raise ValueError("h_um and values must have the same length.")
    if v_range is not None:
        if v_um is None:
            raise ValueError("v_range requires v_um node coordinates.")
        band = np.asarray(v_um, dtype=np.float64).ravel()
        if band.size != h_arr.size:
            raise ValueError("v_um must have the same length as h_um.")
        lo, hi = v_range
        mask = (band >= lo) & (band <= hi)
        if not np.any(mask):
            raise ValueError("No nodes inside v_range.")
        h_arr = np.asarray(h_arr[mask], dtype=np.float64)
        v_arr = np.asarray(v_arr[mask], dtype=np.float64)
    # A 2D node cloud carries many nodes per h coordinate. staircase_profile
    # reads its samples as a piecewise-linear function of h, so a column of
    # nodes would leave one arbitrary node standing per h and discard the
    # rest of the band: average each column here instead. Columns are
    # grouped within a tolerance rather than by exact equality — a mesh
    # generator is free to place a column's nodes at coordinates agreeing
    # only to rounding.
    span = float(np.ptp(h_arr))
    tol = column_tol_um if column_tol_um is not None else COLUMN_TOL_FRACTION * span
    keys = np.round(h_arr / tol).astype(np.int64) if tol > 0.0 else h_arr
    _unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    if _unique.size != h_arr.size:
        h_arr = np.asarray(
            np.bincount(inverse, weights=h_arr) / counts, dtype=np.float64
        )
        v_arr = np.asarray(
            np.bincount(inverse, weights=v_arr) / counts, dtype=np.float64
        )
    return staircase_profile(h_arr, v_arr, n_bins=n_strips, h_min=h_min, h_max=h_max)


def _strip_response(
    edges: ArrayLike,
    n_strips_cm3: ArrayLike,
    p_strips_cm3: ArrayLike,
    *,
    n0: float = DEFAULT_SI_INDEX,
    dispersion: PlasmaDispersionModel | None = None,
    wavelength_um: float | None = None,
    mu_n_cm2: float = DEFAULT_MU_N_CM2,
    mu_p_cm2: float = DEFAULT_MU_P_CM2,
) -> dict[str, Any]:
    """Per-strip material response of a Staircase, for both EM Stages.

    Args:
        edges: Ascending strip edges (um), length N+1.
        n_strips_cm3: Per-strip average electron concentration (cm^-3).
        p_strips_cm3: Per-strip average hole concentration (cm^-3).
        n0: Unperturbed refractive index of the strips.
        dispersion: Plasma-dispersion coefficients. The optical response
            (``dn``, ``dalpha_cm``, ``eps_complex``) is only computed when
            a model is given.
        wavelength_um: Vacuum wavelength the optical Stage solves at (um),
            which sets each Strip's extinction
            ``kappa = dalpha_cm lambda / 4 pi``. The model's own
            wavelength — where its coefficients were fitted, and so what
            ``dalpha_cm`` means, not where anyone is solving — stands in
            when omitted.
        mu_n_cm2: Electron mobility (cm^2/Vs) for the RF conductivity.
        mu_p_cm2: Hole mobility (cm^2/Vs) for the RF conductivity.

    Returns:
        Dict with ``edges_um``, ``n_cm3``, ``p_cm3``, ``sigma_s_per_m``
        and — with a dispersion model — ``dn``, ``dalpha_cm`` and
        ``eps_complex``.
    """
    edge_arr = np.asarray(edges, dtype=np.float64).ravel()
    n_arr = np.asarray(n_strips_cm3, dtype=np.float64).ravel()
    p_arr = np.asarray(p_strips_cm3, dtype=np.float64).ravel()
    sigma = carrier_conductivity(n_arr, p_arr, mu_n_cm2=mu_n_cm2, mu_p_cm2=mu_p_cm2)
    strips_info: dict[str, Any] = {
        "edges_um": edge_arr,
        "n_cm3": n_arr,
        "p_cm3": p_arr,
        "sigma_s_per_m": np.asarray(sigma, dtype=np.float64),
    }
    if dispersion is not None:
        dn = np.asarray(carrier_index_shift(n_arr, p_arr, model=dispersion))
        dalpha = np.asarray(carrier_absorption_cm(n_arr, p_arr, model=dispersion))
        strips_info["dn"] = dn
        strips_info["dalpha_cm"] = dalpha
        strips_info["eps_complex"] = np.asarray(
            [
                permittivity_perturbation(
                    n0=n0,
                    dn=float(dn[i]),
                    dalpha_cm=float(dalpha[i]),
                    wavelength_um=(
                        wavelength_um
                        if wavelength_um is not None
                        else dispersion.wavelength_um
                    ),
                )
                for i in range(n_arr.size)
            ]
        )
    return strips_info


def _strip_material(
    name: str,
    strips_info: dict[str, Any],
    index: int,
    *,
    target: Literal["rf", "optical"] = "rf",
    permittivity: float = 11.9,
    fmax: float = 200e9,
) -> dict[str, MaterialProperties]:
    """Material of one Strip, for the RF or the optical Stage.

    Args:
        name: Region (and material) name of the strip.
        strips_info: Per-strip response from :func:`_strip_response`.
        index: Strip index within that response.
        target: ``"rf"`` (Drude sigma) or ``"optical"`` (perturbed eps).
        permittivity: Relative permittivity of the RF strips.
        fmax: Upper validity frequency (Hz) of the RF Drude material.

    Returns:
        ``{name: MaterialProperties}`` for that one strip.
    """
    if target == "rf":
        return make_doped_materials(
            [
                (
                    name,
                    permittivity,
                    float(strips_info["sigma_s_per_m"][index]),
                    f"carrier staircase ({name}) -- Drude sigma",
                )
            ],
            fmax=fmax,
        )
    if "eps_complex" not in strips_info:
        raise ValueError("target='optical' requires a dispersion model.")
    eps = complex(strips_info["eps_complex"][index])
    eps_re = float(eps.real)
    # exp(+i omega t) convention: lossy medium has Im(eps) < 0.
    loss_tangent = -float(eps.imag) / eps_re if eps_re > 0 else 0.0
    return {
        name: MaterialProperties(
            permittivity=eps_re,
            loss_tangent=loss_tangent,
            dispersion_models=[],
        )
    }


def _make_staircase_profile(
    comp: gf.Component,
    *,
    length: float,
    edges: ArrayLike,
    n_strips_cm3: ArrayLike,
    p_strips_cm3: ArrayLike,
    base_layer: tuple[int, int],
    zmin: float,
    zmax: float,
    name_prefix: str = "strip_",
    target: Literal["rf", "optical"] = "rf",
    permittivity: float = 11.9,
    n0: float = DEFAULT_SI_INDEX,
    dispersion: PlasmaDispersionModel | None = None,
    wavelength_um: float | None = None,
    mu_n_cm2: float = DEFAULT_MU_N_CM2,
    mu_p_cm2: float = DEFAULT_MU_P_CM2,
    fmax: float = 200e9,
    mesh_resolution: str | float = "fine",
) -> dict[str, Any]:
    """Draw N carrier strips and build their layer specs and materials.

    Strip ``i`` spans ``[edges[i], edges[i+1]]`` along y and gets a
    rectangle on GDS layer ``(base_layer[0], base_layer[1] + i)``, a
    ``Layer`` spec named ``"{name_prefix}{i}"``, and a material from its
    average carrier concentrations:

    - ``target="rf"``: Drude conductivity
      ``sigma_i = q (mu_n n_i + mu_p p_i)`` with the shared real
      *permittivity* (same machinery as the doped-slab regions).
    - ``target="optical"``: complex permittivity
      ``(n0 + Delta n_i - i kappa_i)^2`` from the plasma-dispersion
      *dispersion* model, stored as permittivity + loss tangent.

    The returned dict has the same shape as
    :func:`gsim.common.stack.pn_junction.make_doping_profile` (``layer_specs``,
    ``materials``, ``centres``) so it feeds directly into
    ``build_doped_cross_section(doping=...)``; a ``strips`` entry carries
    the per-strip numbers for inspection and convergence checks.

    Args:
        comp: gdsfactory component the strip rectangles are added to.
        length: Rectangle length along the propagation direction (um).
        edges: Ascending strip edges along y (um), length N+1 (e.g. from
            :func:`strip_averages_from_nodes`).
        n_strips_cm3: Per-strip average electron concentration (cm^-3).
        p_strips_cm3: Per-strip average hole concentration (cm^-3).
        base_layer: ``(layer, datatype)`` of strip 0; strip ``i`` uses
            ``datatype + i``.
        zmin: Bottom z of the strips (um).
        zmax: Top z of the strips (um).
        name_prefix: Region-name prefix.
        target: ``"rf"`` (Drude sigma) or ``"optical"`` (Soref eps).
        permittivity: Relative permittivity of the RF strips (and the
            unperturbed lattice for optics bookkeeping).
        n0: Unperturbed refractive index for ``target="optical"``.
        dispersion: Plasma-dispersion coefficients; required for
            ``target="optical"``.
        wavelength_um: Vacuum wavelength the optical Stage solves at (um);
            the dispersion model's own wavelength stands in when omitted.
        mu_n_cm2: Electron mobility (cm^2/Vs) for the RF conductivity.
        mu_p_cm2: Hole mobility (cm^2/Vs) for the RF conductivity.
        fmax: Upper validity frequency (Hz) of the RF Drude materials.
        mesh_resolution: Mesh resolution assigned to the strip layers.

    Returns:
        Dict with keys ``layer_specs``, ``materials``, ``centres`` and
        ``strips`` (per-strip edges/values).
    """
    from gsim.common.stack.extractor import Layer

    edge_arr = np.asarray(edges, dtype=np.float64).ravel()
    n_arr = np.asarray(n_strips_cm3, dtype=np.float64).ravel()
    p_arr = np.asarray(p_strips_cm3, dtype=np.float64).ravel()

    if edge_arr.size < 2:
        raise ValueError("edges needs at least two values (one strip).")
    if np.any(np.diff(edge_arr) <= 0):
        raise ValueError("edges must be strictly ascending.")
    n_strips = edge_arr.size - 1
    if n_arr.size != n_strips or p_arr.size != n_strips:
        raise ValueError(
            f"Expected {n_strips} per-strip values for {n_strips + 1} edges, "
            f"got {n_arr.size} electron and {p_arr.size} hole values."
        )
    if length <= 0:
        raise ValueError("length must be positive.")
    if zmax <= zmin:
        raise ValueError("zmax must exceed zmin.")
    if target == "optical" and dispersion is None:
        raise ValueError("target='optical' requires a dispersion model.")

    result: dict[str, Any] = {
        "layer_specs": {},
        "materials": {},
        "centres": {},
    }
    layer_specs = cast("dict[str, Layer]", result["layer_specs"])
    materials: dict[str, Any] = result["materials"]
    centres: dict[str, float] = result["centres"]

    strips_info = _strip_response(
        edge_arr,
        n_arr,
        p_arr,
        n0=n0,
        dispersion=dispersion,
        wavelength_um=wavelength_um,
        mu_n_cm2=mu_n_cm2,
        mu_p_cm2=mu_p_cm2,
    )

    for i in range(n_strips):
        name = f"{name_prefix}{i}"
        gds_layer = (base_layer[0], base_layer[1] + i)
        y0, y1 = float(edge_arr[i]), float(edge_arr[i + 1])

        # Drawn from its two edges rather than from a width and a centre:
        # adjacent Strips then hand the GDS grid the identical coordinate
        # for the edge they share, so no strip count can snap a sliver of
        # background between them.
        comp.add_polygon(
            [(0.0, y0), (length, y0), (length, y1), (0.0, y1)], layer=gds_layer
        )
        centres[name] = (y0 + y1) / 2

        layer_specs[name] = Layer(
            name=name,
            gds_layer=gds_layer,
            zmin=zmin,
            zmax=zmax,
            thickness=zmax - zmin,
            material=name,
            layer_type="dielectric",
            mesh_resolution=mesh_resolution,
        )

        materials.update(
            _strip_material(
                name,
                strips_info,
                i,
                target=target,
                permittivity=permittivity,
                fmax=fmax,
            )
        )

    result["strips"] = strips_info
    return result


#: How the drawn conductors of a Traveling-wave electrode are expressed
#: in the meshed Cross-section. See ADR 0003.
ConductorModel = Literal["volume", "pec"]


@dataclass(frozen=True)
class ElectrodeSpec:
    """The Traveling-wave electrodes flanking a Staircase.

    Attributes:
        width_um: Width of each electrode along the junction axis (um).
        gap_um: Gap between the Junction extent and the electrode edge (um).
        thickness_um: Electrode thickness (um).
        conductor_model: How the metal is expressed in the mesh (ADR 0003).
            ``"volume"`` meshes each electrode as a Region of lossy metal
            carrying :attr:`sigma_s_per_m`; ``"pec"`` leaves its interior
            out of the meshed domain and makes its outline a perfect
            conductor. Only ``"pec"`` is a Cross-section both first-class
            Routes express identically, and only ``"pec"`` keeps an
            eigenvalue search off the metal-dominated modes a
            ``|eps| ~ 1e7`` Region carries.
        sigma_s_per_m: Electrode conductivity (S/m; aluminium by default),
            used for the RF target of the ``"volume"`` model. A ``"pec"``
            electrode has no conductivity to carry.
        optical_permittivity: Complex relative permittivity of the
            electrode metal at the optical wavelength, in the
            ``exp(+i omega t)`` convention (``Im < 0`` is lossy). The RF
            Drude conductivity above is meaningless at optical
            frequencies, so an optical ``"volume"`` Staircase that
            contains the electrodes needs this value; leave it unset when
            the optical Window excludes them, or when the electrodes are
            ``"pec"`` and so carry no permittivity at all.
        zmin: Bottom z of the electrodes (um); defaults to the strip zmin.
        names: Region names of the low-side and high-side electrode.
        gds_layer: ``(layer, datatype)`` of the low-side electrode; the
            high-side one uses ``datatype + 1``.
    """

    width_um: float = 2.0
    gap_um: float = 0.0
    thickness_um: float = 0.5
    conductor_model: ConductorModel = "volume"
    sigma_s_per_m: float = 3.8e7
    optical_permittivity: complex | None = None
    zmin: float | None = None
    names: tuple[str, str] = ("electrode_low", "electrode_high")
    gds_layer: tuple[int, int] = (42, 0)


#: Default Traveling-wave electrodes: 2 um wide, touching the Junction extent.
DEFAULT_ELECTRODES = ElectrodeSpec()


@dataclass
class StaircaseCrossSection:
    """A Carrier map reduced to Strips, ready to mesh.

    The strip Regions and the electrodes are drawn once on
    :attr:`component`; :meth:`stack` resolves them to a ``LayerStack`` for
    whichever EM Stage asks, so the RF and the optical Stage consume the
    same Staircase.

    Attributes:
        component: The gdsfactory component carrying the strip and
            electrode rectangles.
        strips: Per-strip numbers from :func:`_strip_response` — edges,
            average concentrations, Drude conductivity, and the optical
            index shift, absorption and permittivity.
        strip_names: Region names of the strips, low edge first.
        electrode_names: Region names of the electrodes (empty when the
            Staircase was built without them).
        electrode_spans: ``(min, max)`` extent of each electrode along the
            junction axis (um), in the same order as the names.
        strip_span: ``(min, max)`` extent the Strips tile (um).
        surroundings: The drawn device's own Regions redrawn around the
            Strips (empty on a Staircase built without them).
    """

    component: gf.Component
    strips: dict[str, Any]
    strip_names: list[str]
    electrode_names: tuple[str, ...]
    electrode_spans: tuple[tuple[float, float], ...]
    surroundings: tuple[SurroundingRegion, ...]
    _layer_specs: dict[str, Layer]
    _centres: dict[str, float]
    _electrodes: ElectrodeSpec | None
    _axis: Literal["x", "y", "z"]
    _value: float
    _substrate_thickness: float
    _permittivity: float
    _fmax: float
    _stacks: dict[str, LayerStack] = field(default_factory=dict)

    @property
    def strip_span(self) -> tuple[float, float]:
        """``(min, max)`` extent the Strips actually tile (um)."""
        edges = np.asarray(self.strips["edges_um"], dtype=np.float64).ravel()
        return (float(edges[0]), float(edges[-1]))

    @property
    def conductor_model(self) -> ConductorModel | None:
        """How the electrodes are expressed in the mesh, or None.

        Returns:
            The :class:`ElectrodeSpec`'s model, and ``None`` for a
            Staircase drawn without electrodes.
        """
        return self._electrodes.conductor_model if self._electrodes else None

    def electrode_extent(self, name: str) -> tuple[tuple[float, float], ...]:
        """The rectangle one electrode occupies on the Cross-section.

        A downstream integral over a conductor — the line current the
        characteristic impedance divides by — needs the conductor's
        outline, and under the ``"pec"`` model the mesh no longer carries
        it as a Region to look up.

        Args:
            name: Region name of the electrode.

        Returns:
            ``((h_min, h_max), (v_min, v_max))`` in um: the extent along
            the junction axis, then the vertical one.

        Raises:
            ValueError: When the Staircase has no electrode of that name.
        """
        if name not in self.electrode_names:
            raise ValueError(
                f"The staircase has no electrode named '{name}'; it drew "
                f"{list(self.electrode_names)}."
            )
        index = self.electrode_names.index(name)
        layer = self._layer_specs[name]
        return (self.electrode_spans[index], (layer.zmin, layer.zmax))

    def doping(self, target: Literal["rf", "optical"] = "rf") -> dict[str, Any]:
        """Layer specs, materials and centres for ``build_doped_cross_section``.

        Args:
            target: ``"rf"`` (Drude sigma per strip) or ``"optical"``
                (carrier-perturbed permittivity per strip).

        Returns:
            The doping mapping the cross-section builder consumes.
        """
        materials: dict[str, MaterialProperties] = {}
        for index, name in enumerate(self.strip_names):
            materials.update(
                _strip_material(
                    name,
                    self.strips,
                    index,
                    target=target,
                    permittivity=self._permittivity,
                    fmax=self._fmax,
                )
            )
        materials.update(self._electrode_materials(target))
        surrounding: dict[str, Any] = {
            region.material: region.properties
            for region in self.surroundings
            if region.properties is not None
        }
        return {
            "layer_specs": dict(self._layer_specs),
            # The drawn stack's own material entries first, so a Strip or
            # an electrode sharing a name still wins: the Staircase is
            # what the Carrier map speaks for.
            "materials": surrounding | materials,
            "centres": dict(self._centres),
        }

    def _electrode_materials(
        self, target: Literal["rf", "optical"]
    ) -> dict[str, MaterialProperties]:
        """Electrode materials for one EM Stage.

        Args:
            target: ``"rf"`` (Drude conductor) or ``"optical"``.

        Returns:
            ``{name: MaterialProperties}`` for every electrode.

        Raises:
            ValueError: For an optical ``"volume"`` Staircase whose
                electrodes have no optical permittivity — the RF
                conductivity would model them as a near-transparent
                dielectric.
        """
        spec = self._electrodes
        if spec is None or not self.electrode_names:
            return {}
        if spec.conductor_model == "pec":
            # No conductivity and no loss: the native-2D mesher reads the
            # material to decide whether an electrode's outline carries a
            # finite-conductivity surface impedance or is a perfect
            # conductor, and a perfect conductor is the one both Routes
            # express identically (ADR 0003).
            return {
                name: MaterialProperties(
                    permittivity=1.0, loss_tangent=0.0, dispersion_models=[]
                )
                for name in self.electrode_names
            }
        if target == "rf":
            return make_doped_materials(
                [(name, spec.sigma_s_per_m) for name in self.electrode_names],
                permittivity=1.0,
                source_prefix="electrode",
            )
        if spec.optical_permittivity is None:
            raise ValueError(
                "The staircase electrodes only carry an RF Drude "
                "conductivity, which is meaningless at optical "
                "frequencies. Give the metal its optical permittivity "
                "(ElectrodeSpec(optical_permittivity=...)), or build the "
                "staircase with electrodes=None when the optical window "
                "excludes them."
            )
        eps = complex(spec.optical_permittivity)
        eps_re = float(eps.real)
        # exp(+i omega t) convention: lossy medium has Im(eps) < 0.
        loss_tangent = -float(eps.imag) / eps_re if eps_re != 0.0 else 0.0
        return {
            name: MaterialProperties(
                permittivity=eps_re,
                loss_tangent=loss_tangent,
                dispersion_models=[],
            )
            for name in self.electrode_names
        }

    def stack(self, target: Literal["rf", "optical"] = "rf") -> LayerStack:
        """Resolve the Staircase into a meshable layer stack.

        Args:
            target: Which EM Stage the materials are for — ``"rf"`` uses
                the Drude conductivity of each Strip, ``"optical"`` its
                carrier-perturbed permittivity.

        Returns:
            The ``LayerStack``, built once per target and cached.
        """
        if target not in ("rf", "optical"):
            raise ValueError(f"Unknown target {target!r}; use 'rf' or 'optical'.")
        if target not in self._stacks:
            from gsim.common.cross_section import build_doped_cross_section

            stack, _section = build_doped_cross_section(
                self.component,
                axis=self._axis,
                value=self._value,
                substrate_thickness=self._substrate_thickness,
                doping=self.doping(target),
                verbose=False,
            )
            self._stacks[target] = stack
        return self._stacks[target]


def _electrode_layers(
    comp: gf.Component,
    spec: ElectrodeSpec,
    *,
    junction: tuple[float, float],
    length: float,
    zmin: float,
    mesh_resolution: str | float,
) -> tuple[
    dict[str, Layer],
    dict[str, float],
    tuple[tuple[float, float], ...],
]:
    """Draw the flanking electrodes and build their layer specs."""
    from gsim.common.stack.extractor import Layer

    if spec.width_um <= 0:
        raise ValueError("Electrode width_um must be positive.")
    if spec.gap_um < 0:
        raise ValueError("Electrode gap_um must be non-negative.")
    if spec.thickness_um <= 0:
        raise ValueError("Electrode thickness_um must be positive.")

    h_min, h_max = junction
    spans = (
        (h_min - spec.gap_um - spec.width_um, h_min - spec.gap_um),
        (h_max + spec.gap_um, h_max + spec.gap_um + spec.width_um),
    )
    base_z = spec.zmin if spec.zmin is not None else zmin

    layer_specs: dict[str, Layer] = {}
    centres: dict[str, float] = {}
    for index, (name, (y0, y1)) in enumerate(zip(spec.names, spans, strict=True)):
        gds_layer = (spec.gds_layer[0], spec.gds_layer[1] + index)
        # From its edges, like the Strips: an electrode is meant to touch
        # the Strip lattice, not to sit a grid rounding away from it.
        comp.add_polygon(
            [(0.0, y0), (length, y0), (length, y1), (0.0, y1)], layer=gds_layer
        )
        centres[name] = (y0 + y1) / 2
        layer_specs[name] = Layer(
            name=name,
            gds_layer=gds_layer,
            zmin=base_z,
            zmax=base_z + spec.thickness_um,
            thickness=spec.thickness_um,
            material=name,
            # A conductor layer is what the native-2D mesher meshes as an
            # outline rather than as a domain, which is what makes the
            # "pec" model a boundary condition instead of a Region
            # (ADR 0003).
            layer_type="conductor" if spec.conductor_model == "pec" else "dielectric",
            mesh_resolution=mesh_resolution,
        )
    return layer_specs, centres, spans


def carrier_map_extent(
    carriers: Any, band: tuple[float, float] | None = None
) -> tuple[float, float]:
    """The extent a Carrier map covers along the junction axis (um).

    Strips average the Carrier map, so they cannot reach past it: the
    extent here is the widest one
    :func:`build_staircase_cross_section` will accept.

    Args:
        carriers: The Carrier map (anything exposing ``x_um`` and
            ``y_um``).
        band: ``(min, max)`` vertical band of samples to measure across;
            the whole map when omitted.

    Returns:
        ``(min, max)`` along the junction axis.

    Raises:
        ValueError: When the band holds no sample of the map.
    """
    h_um = np.asarray(carriers.x_um, dtype=np.float64).ravel()
    v_um = np.asarray(carriers.y_um, dtype=np.float64).ravel()
    if band is None:
        inside = np.ones(v_um.shape, dtype=bool)
    else:
        inside = (v_um >= band[0]) & (v_um <= band[1])
    if not np.any(inside):
        raise ValueError(
            f"No carrier samples inside the vertical band {band}; "
            f"the map spans z in [{v_um.min():.3g}, {v_um.max():.3g}] um."
        )
    return (float(h_um[inside].min()), float(h_um[inside].max()))


@dataclass(frozen=True)
class SurroundingRegion:
    """A Region of the drawn device redrawn beside the Strips.

    The Strips carry the Carrier map, and nothing else. Everything the
    drawn Cross-section has around them — the undoped silicon the guide
    slab is made of, the Traveling-wave metal landing on the pads, an
    implant the charge solve never covered — guides the Mode just as much,
    and a Staircase that omits it solves a different waveguide. Each such
    Region reaches the Staircase as one of these, cut against the Strip
    footprint so the two never overlap.

    Attributes:
        name: Region name on the meshed Cross-section.
        h: ``(min, max)`` extent along the junction axis (um).
        z: ``(min, max)`` vertical extent (um).
        material: Material name, as the drawn stack names it.
        layer_type: How the mesher expresses it — ``"dielectric"`` for a
            meshed domain, ``"conductor"`` for metal meshed as an outline
            (ADR 0003).
        properties: The material's own entry from the drawn stack, for a
            material the base materials database does not already carry
            (a doped-silicon material, say). ``None`` leaves the lookup to
            the database.
        mesh_resolution: Mesh resolution assigned to the Region.
    """

    name: str
    h: tuple[float, float]
    z: tuple[float, float]
    material: str
    layer_type: Literal["conductor", "via", "dielectric", "substrate"] = "dielectric"
    properties: Any | None = None
    mesh_resolution: str | float = "fine"


def _cut_against(
    h: tuple[float, float],
    z: tuple[float, float],
    *,
    box_h: tuple[float, float],
    box_z: tuple[float, float],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """The parts of one axis-aligned rectangle outside another.

    Args:
        h: ``(min, max)`` in-plane extent of the rectangle (um).
        z: ``(min, max)`` vertical extent of the rectangle (um).
        box_h: In-plane extent of the rectangle cut out of it.
        box_z: Vertical extent of the rectangle cut out of it.

    Returns:
        Up to four disjoint rectangles covering exactly the part of the
        first that the second does not; the rectangle itself when the two
        do not overlap, and nothing when it is entirely inside.
    """
    overlap_h = (max(h[0], box_h[0]), min(h[1], box_h[1]))
    overlap_z = (max(z[0], box_z[0]), min(z[1], box_z[1]))
    if (
        overlap_h[1] - overlap_h[0] <= SURROUND_TOL_UM
        or overlap_z[1] - overlap_z[0] <= SURROUND_TOL_UM
    ):
        return [(h, z)]

    pieces: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if overlap_h[0] - h[0] > SURROUND_TOL_UM:
        pieces.append(((h[0], overlap_h[0]), z))
    if h[1] - overlap_h[1] > SURROUND_TOL_UM:
        pieces.append(((overlap_h[1], h[1]), z))
    if overlap_z[0] - z[0] > SURROUND_TOL_UM:
        pieces.append((overlap_h, (z[0], overlap_z[0])))
    if z[1] - overlap_z[1] > SURROUND_TOL_UM:
        pieces.append((overlap_h, (overlap_z[1], z[1])))
    return pieces


def surroundings_from_section(
    section: Iterable[Any],
    *,
    strip_span: tuple[float, float],
    strip_z: tuple[float, float],
    stack: LayerStack | None = None,
) -> tuple[SurroundingRegion, ...]:
    """Everything a drawn Cross-section has around the Strips.

    Each rectangle of the drawn section is cut against the Strip
    footprint: the part the Strips replace is dropped, and the rest
    becomes a :class:`SurroundingRegion` carrying the drawn material.
    That is one rule for every Region. The doped silicon the Carrier map
    speaks for disappears wherever the Strips cover it and survives
    wherever they do not, so a Strip extent narrower than the doped slab
    leaves unperturbed silicon rather than a hole.

    Args:
        section: Rectangles of the drawn Cross-section, each exposing
            ``layer_name``, ``material``, ``y0``, ``y1``, ``zmin`` and
            ``zmax`` (the output of
            :func:`gsim.common.cross_section.extract_plane_section` on a
            vertical plane).
        strip_span: ``(min, max)`` extent the Strips tile (um).
        strip_z: ``(min, max)`` vertical extent of the Strips (um).
        stack: The drawn layer stack, read for two things the section
            rectangles do not carry: how each Region is meshed
            (``"conductor"`` metal becomes an outline rather than a
            domain, ADR 0003), and the material entry of a material the
            base database does not know.

    Returns:
        The surrounding Regions, in section order, each named after the
        Region it came from (suffixed when one rectangle cuts into
        several pieces).
    """
    material_map = dict(stack.materials) if stack is not None else {}
    layers = dict(stack.layers) if stack is not None else {}
    regions: list[SurroundingRegion] = []
    for rect in section:
        name = str(rect.layer_name)
        h = (float(rect.y0), float(rect.y1))
        z = (float(rect.zmin), float(rect.zmax))
        if h[1] - h[0] <= SURROUND_TOL_UM or z[1] - z[0] <= SURROUND_TOL_UM:
            continue
        pieces = _cut_against(h, z, box_h=strip_span, box_z=strip_z)
        layer = layers.get(name)
        layer_type = layer.layer_type if layer is not None else "dielectric"
        for index, (piece_h, piece_z) in enumerate(pieces):
            regions.append(
                SurroundingRegion(
                    name=name if len(pieces) == 1 else f"{name}_{index}",
                    h=piece_h,
                    z=piece_z,
                    material=str(rect.material),
                    layer_type=layer_type,
                    properties=material_map.get(str(rect.material)),
                )
            )
    return tuple(regions)


def _surrounding_layers(
    comp: gf.Component,
    surroundings: Sequence[SurroundingRegion],
    *,
    length: float,
    base_layer: tuple[int, int],
) -> tuple[dict[str, Layer], dict[str, float]]:
    """Draw the surrounding Regions and build their layer specs."""
    from gsim.common.stack.extractor import Layer

    layer_specs: dict[str, Layer] = {}
    centres: dict[str, float] = {}
    for index, region in enumerate(surroundings):
        if region.h[1] <= region.h[0]:
            raise ValueError(
                f"Surrounding region '{region.name}' has a non-ascending "
                f"in-plane extent {region.h}."
            )
        if region.z[1] <= region.z[0]:
            raise ValueError(
                f"Surrounding region '{region.name}' has a non-ascending "
                f"vertical extent {region.z}."
            )
        if region.name in layer_specs:
            raise ValueError(
                f"Two surrounding regions are both named '{region.name}'; "
                "region names have to be unique on the cross-section."
            )
        gds_layer = (base_layer[0], base_layer[1] + index)
        y0, y1 = region.h
        comp.add_polygon(
            [(0.0, y0), (length, y0), (length, y1), (0.0, y1)], layer=gds_layer
        )
        centres[region.name] = 0.5 * (y0 + y1)
        layer_specs[region.name] = Layer(
            name=region.name,
            gds_layer=gds_layer,
            zmin=region.z[0],
            zmax=region.z[1],
            thickness=region.z[1] - region.z[0],
            material=region.material,
            layer_type=region.layer_type,
            mesh_resolution=region.mesh_resolution,
        )
    return layer_specs, centres


def build_staircase_cross_section(
    carriers: Any,
    *,
    n_strips: int,
    junction: tuple[float, float],
    zmin: float,
    zmax: float,
    band: tuple[float, float] | None = None,
    length: float = STRIP_LENGTH_UM,
    electrodes: ElectrodeSpec | None = DEFAULT_ELECTRODES,
    surroundings: Sequence[SurroundingRegion] = (),
    dispersion: PlasmaDispersionModel | None = None,
    wavelength_um: float | None = None,
    n0: float = DEFAULT_SI_INDEX,
    permittivity: float = 11.9,
    mu_n_cm2: float = DEFAULT_MU_N_CM2,
    mu_p_cm2: float = DEFAULT_MU_P_CM2,
    fmax: float = 200e9,
    base_layer: tuple[int, int] = DEFAULT_STRIP_LAYER,
    surround_layer: tuple[int, int] = DEFAULT_SURROUND_LAYER,
    name_prefix: str = "strip_",
    mesh_resolution: str | float = "fine",
    axis: Literal["x", "y", "z"] = "x",
    value: float = 0.0,
    substrate_thickness: float = 2.0,
    component: gf.Component | None = None,
) -> StaircaseCrossSection:
    """Turn a Carrier map into a meshable Staircase cross-section.

    The Carrier map is binned into *n_strips* piecewise-constant Strips
    across the Junction extent, each Strip is drawn as its own Region with
    the material response of both EM Stages, and the Traveling-wave
    electrodes are placed from the device description rather than by the
    caller.

    Coordinates follow the Carrier map's own frame: ``carriers.x_um`` runs
    along the junction axis (the in-plane coordinate of the Cross-section)
    and ``carriers.y_um`` is the vertical one, which is how the
    charge-transport backend reports a Carrier map.

    Args:
        carriers: The Carrier map to staircase (anything exposing
            ``x_um``, ``y_um``, ``electrons_cm3`` and ``holes_cm3``).
        n_strips: Number of Strips; ``1`` recovers the uniform model.
        junction: ``(min, max)`` Junction extent along the junction axis
            (um) the Strips tile.
        zmin: Bottom z of the Strips (um).
        zmax: Top z of the Strips (um).
        band: ``(min, max)`` vertical band of Carrier-map samples averaged
            into the Strips; defaults to ``(zmin, zmax)``.
        length: Drawn length along the propagation direction (um).
        electrodes: Flanking electrodes; ``None`` draws none.
        surroundings: The drawn device's own Regions to redraw around the
            Strips — see :func:`surroundings_from_section`. Empty leaves
            the Staircase as Strips alone in the background medium, which
            is the right Cross-section only when the drawn device has
            nothing else inside the meshed Window.
        dispersion: Plasma-dispersion coefficients for the optical
            response; defaults to the 1.55 um fit.
        wavelength_um: Vacuum wavelength the optical Stage solves at (um).
            Each Strip's extinction is built at it, so the Staircase and
            a continuous ``eps(x, y)`` carry the same loss. The
            dispersion model's own wavelength — where its coefficients
            were fitted — stands in when omitted, which is right only
            when the solve happens to sit there.
        n0: Unperturbed refractive index of the Strips.
        permittivity: Relative permittivity of the RF Strips.
        mu_n_cm2: Electron mobility (cm^2/Vs) for the RF conductivity.
        mu_p_cm2: Hole mobility (cm^2/Vs) for the RF conductivity.
        fmax: Upper validity frequency (Hz) of the RF Drude materials.
        base_layer: ``(layer, datatype)`` of Strip 0; defaults to
            :data:`DEFAULT_STRIP_LAYER`, outside the generic PDK's
            own layers.
        surround_layer: ``(layer, datatype)`` of the first surrounding
            Region; defaults to :data:`DEFAULT_SURROUND_LAYER`.
        name_prefix: Region-name prefix of the Strips.
        mesh_resolution: Mesh resolution assigned to the Strip layers.
        axis: Cross-section normal axis of the resolved stack.
        value: Cross-section plane coordinate (um).
        substrate_thickness: Substrate thickness of the resolved stack (um).
        component: Component to draw on; a new one is created by default.

    Returns:
        The :class:`StaircaseCrossSection`.

    Raises:
        ValueError: When the Junction extent reaches outside the Carrier
            map, or the strip count or geometry is not usable.
    """
    import gdsfactory as gf

    h_min, h_max = float(junction[0]), float(junction[1])
    if h_max <= h_min:
        raise ValueError("junction must be an ascending (min, max) extent.")
    band_range = band if band is not None else (zmin, zmax)

    h_um = np.asarray(carriers.x_um, dtype=np.float64).ravel()
    v_um = np.asarray(carriers.y_um, dtype=np.float64).ravel()
    covered = carrier_map_extent(carriers, band_range)
    if h_min < covered[0] or h_max > covered[1]:
        raise ValueError(
            f"Junction extent {junction} reaches outside the carrier map, "
            f"which covers [{covered[0]:.3g}, {covered[1]:.3g}] um along the "
            "junction axis. Widen the charge Window or narrow the extent."
        )

    edges, n_means = strip_averages_from_nodes(
        h_um,
        carriers.electrons_cm3,
        n_strips=n_strips,
        h_min=h_min,
        h_max=h_max,
        v_um=v_um,
        v_range=band_range,
    )
    _edges, p_means = strip_averages_from_nodes(
        h_um,
        carriers.holes_cm3,
        n_strips=n_strips,
        h_min=h_min,
        h_max=h_max,
        v_um=v_um,
        v_range=band_range,
    )

    comp = component if component is not None else gf.Component()
    profile = _make_staircase_profile(
        comp,
        length=length,
        edges=edges,
        n_strips_cm3=n_means,
        p_strips_cm3=p_means,
        base_layer=base_layer,
        zmin=zmin,
        zmax=zmax,
        name_prefix=name_prefix,
        target="rf",
        permittivity=permittivity,
        n0=n0,
        dispersion=(
            dispersion
            if dispersion is not None
            else PlasmaDispersionModel.nedeljkovic_1550()
        ),
        wavelength_um=wavelength_um,
        mu_n_cm2=mu_n_cm2,
        mu_p_cm2=mu_p_cm2,
        fmax=fmax,
        mesh_resolution=mesh_resolution,
    )
    layer_specs = cast("dict[str, Layer]", profile["layer_specs"])
    centres = cast("dict[str, float]", profile["centres"])
    strip_names = list(layer_specs)

    electrode_names: tuple[str, ...] = ()
    electrode_spans: tuple[tuple[float, float], ...] = ()
    if electrodes is not None:
        specs, electrode_centres, electrode_spans = _electrode_layers(
            comp,
            electrodes,
            junction=(h_min, h_max),
            length=length,
            zmin=zmin,
            mesh_resolution=mesh_resolution,
        )
        layer_specs.update(specs)
        centres.update(electrode_centres)
        electrode_names = tuple(specs)

    if surroundings:
        clashes = [region.name for region in surroundings if region.name in layer_specs]
        if clashes:
            raise ValueError(
                f"Surrounding region(s) {clashes} share a name with a strip "
                "or an electrode of this staircase; rename them so every "
                "region on the cross-section is distinct."
            )
        specs, surround_centres = _surrounding_layers(
            comp,
            tuple(surroundings),
            length=length,
            base_layer=surround_layer,
        )
        layer_specs.update(specs)
        centres.update(surround_centres)

    return StaircaseCrossSection(
        component=comp,
        strips=cast("dict[str, Any]", profile["strips"]),
        strip_names=strip_names,
        electrode_names=electrode_names,
        electrode_spans=electrode_spans,
        surroundings=tuple(surroundings),
        _layer_specs=layer_specs,
        _centres=centres,
        _electrodes=electrodes,
        _axis=axis,
        _value=value,
        _substrate_thickness=substrate_thickness,
        _permittivity=permittivity,
        _fmax=fmax,
    )
