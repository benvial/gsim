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

``n_strips=1`` recovers the uniform-strip model: one rectangle spanning
the window carrying the profile average.
"""

from __future__ import annotations

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

    from gsim.common.stack.extractor import Layer

__all__ = ["make_staircase_profile", "strip_averages_from_nodes"]

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

    sigma = carrier_conductivity(n_arr, p_arr, mu_n_cm2=mu_n_cm2, mu_p_cm2=mu_p_cm2)
    strips_info: dict[str, Any] = {
        "edges_um": edge_arr,
        "n_cm3": n_arr,
        "p_cm3": p_arr,
        "sigma_s_per_m": np.asarray(sigma, dtype=np.float64),
    }

    if target == "optical":
        model = cast(PlasmaDispersionModel, dispersion)
        dn = np.asarray(carrier_index_shift(n_arr, p_arr, model=model))
        dalpha = np.asarray(carrier_absorption_cm(n_arr, p_arr, model=model))
        eps = np.asarray(
            [
                permittivity_perturbation(
                    n0=n0,
                    dn=float(dn[i]),
                    dalpha_cm=float(dalpha[i]),
                    wavelength_um=model.wavelength_um,
                )
                for i in range(n_strips)
            ]
        )
        strips_info["dn"] = dn
        strips_info["dalpha_cm"] = dalpha
        strips_info["eps_complex"] = eps

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

        if target == "rf":
            materials.update(
                make_doped_materials(
                    [
                        (
                            name,
                            permittivity,
                            float(strips_info["sigma_s_per_m"][i]),
                            f"carrier staircase ({name}) -- Drude sigma",
                        )
                    ],
                    fmax=fmax,
                )
            )
        else:
            eps_i = complex(strips_info["eps_complex"][i])
            eps_re = float(eps_i.real)
            # exp(+i omega t) convention: lossy medium has Im(eps) < 0.
            loss_tangent = -float(eps_i.imag) / eps_re if eps_re > 0 else 0.0
            materials[name] = MaterialProperties(
                permittivity=eps_re,
                loss_tangent=loss_tangent,
                dispersion_models=[],
            )

    result["strips"] = strips_info
    return result
