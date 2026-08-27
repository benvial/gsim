"""Staircase route: carrier maps as piecewise-constant Palace domains.

Palace only accepts piecewise-constant materials per mesh domain, so a
continuously varying carrier distribution is represented as N adjacent
strips along the junction axis. Each strip becomes a patterned-dielectric
region through the same ``Layer``/``MaterialProperties`` machinery the
PN-junction profile uses (:mod:`gsim.common.stack.doping`), so the result
plugs straight into ``build_doped_cross_section(doping=...)`` and the
native ``BoundaryMode`` solver.

- :func:`strip_averages_from_nodes` reduces scattered solver-node values
  (e.g. a :class:`gsim.tcad.results.CarrierMap`) to exact per-strip
  averages — the tested, reusable mesh-transfer step.
- :func:`make_staircase_profile` draws the strips on a component and emits
  layer specs plus per-strip materials: Drude conductivity for RF, or the
  Soref/Nedeljkovic complex permittivity for optics.
- :func:`build_staircase_cross_section` does the whole job in one call: a
  Carrier map, a strip count and the Junction extent in, a meshable
  Staircase cross-section out — the strip Regions, their material
  response for *both* EM Stages, and the flanking electrodes.

``n_strips=1`` recovers the uniform-strip model: one rectangle spanning
the window carrying the profile average.
"""

from __future__ import annotations

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
    staircase_profile,
)
from gsim.common.stack.materials import MaterialProperties, make_doped_materials

if TYPE_CHECKING:
    import gdsfactory as gf

    from gsim.common.stack.extractor import Layer, LayerStack

__all__ = [
    "DEFAULT_ELECTRODES",
    "ElectrodeSpec",
    "StaircaseCrossSection",
    "build_staircase_cross_section",
    "make_staircase_profile",
    "strip_averages_from_nodes",
]

#: Unperturbed silicon refractive index near 1.55 um.
DEFAULT_SI_INDEX: float = 3.4757


def strip_averages_from_nodes(
    h_um: ArrayLike,
    values: ArrayLike,
    *,
    n_strips: int,
    h_min: float | None = None,
    h_max: float | None = None,
    v_um: ArrayLike | None = None,
    v_range: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Average scattered node values into N strips along the junction axis.

    Node values (e.g. carrier concentrations on the charge-solve mesh) are
    reduced to a 1D profile of ``h`` and binned with
    :func:`gsim.common.carriers.staircase_profile`, whose strip values are
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
    return staircase_profile(h_arr, v_arr, n_bins=n_strips, h_min=h_min, h_max=h_max)


def strip_response(
    edges: ArrayLike,
    n_strips_cm3: ArrayLike,
    p_strips_cm3: ArrayLike,
    *,
    n0: float = DEFAULT_SI_INDEX,
    dispersion: PlasmaDispersionModel | None = None,
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
                    wavelength_um=dispersion.wavelength_um,
                )
                for i in range(n_arr.size)
            ]
        )
    return strips_info


def strip_material(
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
        strips_info: Per-strip response from :func:`strip_response`.
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


def make_staircase_profile(
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
    :func:`gsim.common.stack.doping.make_doping_profile` (``layer_specs``,
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
        mu_n_cm2: Electron mobility (cm^2/Vs) for the RF conductivity.
        mu_p_cm2: Hole mobility (cm^2/Vs) for the RF conductivity.
        fmax: Upper validity frequency (Hz) of the RF Drude materials.
        mesh_resolution: Mesh resolution assigned to the strip layers.

    Returns:
        Dict with keys ``layer_specs``, ``materials``, ``centres`` and
        ``strips`` (per-strip edges/values).
    """
    import gdsfactory as gf

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

    strips_info = strip_response(
        edge_arr,
        n_arr,
        p_arr,
        n0=n0,
        dispersion=dispersion,
        mu_n_cm2=mu_n_cm2,
        mu_p_cm2=mu_p_cm2,
    )

    for i in range(n_strips):
        name = f"{name_prefix}{i}"
        gds_layer = (base_layer[0], base_layer[1] + i)
        y0, y1 = float(edge_arr[i]), float(edge_arr[i + 1])

        rect = comp << gf.c.rectangle((length, y1 - y0), layer=gds_layer)
        rect.y = (y0 + y1) / 2
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
            strip_material(
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


@dataclass(frozen=True)
class ElectrodeSpec:
    """The Traveling-wave electrodes flanking a Staircase.

    Attributes:
        width_um: Width of each electrode along the junction axis (um).
        gap_um: Gap between the Junction extent and the electrode edge (um).
        thickness_um: Electrode thickness (um).
        sigma_s_per_m: Electrode conductivity (S/m; aluminium by default),
            used for the RF target.
        optical_permittivity: Complex relative permittivity of the
            electrode metal at the optical wavelength, in the
            ``exp(+i omega t)`` convention (``Im < 0`` is lossy). The RF
            Drude conductivity above is meaningless at optical
            frequencies, so an optical Staircase that contains the
            electrodes needs this value; leave it unset when the optical
            Window excludes them.
        zmin: Bottom z of the electrodes (um); defaults to the strip zmin.
        names: Region names of the low-side and high-side electrode.
        gds_layer: ``(layer, datatype)`` of the low-side electrode; the
            high-side one uses ``datatype + 1``.
    """

    width_um: float = 2.0
    gap_um: float = 0.0
    thickness_um: float = 0.5
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
        strips: Per-strip numbers from :func:`strip_response` — edges,
            average concentrations, Drude conductivity, and the optical
            index shift, absorption and permittivity.
        strip_names: Region names of the strips, low edge first.
        electrode_names: Region names of the electrodes (empty when the
            Staircase was built without them).
        electrode_spans: ``(min, max)`` extent of each electrode along the
            junction axis (um), in the same order as the names.
    """

    component: gf.Component
    strips: dict[str, Any]
    strip_names: list[str]
    electrode_names: tuple[str, ...]
    electrode_spans: tuple[tuple[float, float], ...]
    _layer_specs: dict[str, Layer]
    _centres: dict[str, float]
    _electrodes: ElectrodeSpec | None
    _axis: Literal["x", "y", "z"]
    _value: float
    _substrate_thickness: float
    _permittivity: float
    _fmax: float
    _stacks: dict[str, LayerStack] = field(default_factory=dict)

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
                strip_material(
                    name,
                    self.strips,
                    index,
                    target=target,
                    permittivity=self._permittivity,
                    fmax=self._fmax,
                )
            )
        materials.update(self._electrode_materials(target))
        return {
            "layer_specs": dict(self._layer_specs),
            "materials": materials,
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
            ValueError: For an optical Staircase whose electrodes have no
                optical permittivity — the RF conductivity would model
                them as a near-transparent dielectric.
        """
        spec = self._electrodes
        if spec is None or not self.electrode_names:
            return {}
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
    import gdsfactory as gf

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
        rect = comp << gf.c.rectangle((length, y1 - y0), layer=gds_layer)
        rect.y = (y0 + y1) / 2
        centres[name] = (y0 + y1) / 2
        layer_specs[name] = Layer(
            name=name,
            gds_layer=gds_layer,
            zmin=base_z,
            zmax=base_z + spec.thickness_um,
            thickness=spec.thickness_um,
            material=name,
            layer_type="dielectric",
            mesh_resolution=mesh_resolution,
        )
    return layer_specs, centres, spans


def build_staircase_cross_section(
    carriers: Any,
    *,
    n_strips: int,
    junction: tuple[float, float],
    zmin: float,
    zmax: float,
    band: tuple[float, float] | None = None,
    length: float = 10.0,
    electrodes: ElectrodeSpec | None = DEFAULT_ELECTRODES,
    dispersion: PlasmaDispersionModel | None = None,
    n0: float = DEFAULT_SI_INDEX,
    permittivity: float = 11.9,
    mu_n_cm2: float = DEFAULT_MU_N_CM2,
    mu_p_cm2: float = DEFAULT_MU_P_CM2,
    fmax: float = 200e9,
    base_layer: tuple[int, int] = (40, 0),
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
        dispersion: Plasma-dispersion coefficients for the optical
            response; defaults to the 1.55 um fit.
        n0: Unperturbed refractive index of the Strips.
        permittivity: Relative permittivity of the RF Strips.
        mu_n_cm2: Electron mobility (cm^2/Vs) for the RF conductivity.
        mu_p_cm2: Hole mobility (cm^2/Vs) for the RF conductivity.
        fmax: Upper validity frequency (Hz) of the RF Drude materials.
        base_layer: ``(layer, datatype)`` of Strip 0.
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
    inside = (v_um >= band_range[0]) & (v_um <= band_range[1])
    if not np.any(inside):
        raise ValueError(
            f"No carrier samples inside the vertical band {band_range}; "
            f"the map spans z in [{v_um.min():.3g}, {v_um.max():.3g}] um."
        )
    covered = (float(h_um[inside].min()), float(h_um[inside].max()))
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
    profile = make_staircase_profile(
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

    return StaircaseCrossSection(
        component=comp,
        strips=cast("dict[str, Any]", profile["strips"]),
        strip_names=strip_names,
        electrode_names=electrode_names,
        electrode_spans=electrode_spans,
        _layer_specs=layer_specs,
        _centres=centres,
        _electrodes=electrodes,
        _axis=axis,
        _value=value,
        _substrate_thickness=substrate_thickness,
        _permittivity=permittivity,
        _fmax=fmax,
    )
