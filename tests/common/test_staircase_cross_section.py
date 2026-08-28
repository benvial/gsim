"""One call from a Carrier map to a meshable Staircase cross-section."""

from __future__ import annotations

import sys
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from gsim.common.stack.staircase import (
    DEFAULT_STRIP_LAYER,
    ElectrodeSpec,
    build_staircase_cross_section,
)

CENTER = -20.0
HALF_WIDTH = 0.3
RIB_HEIGHT = 0.22
JUNCTION = (CENTER - HALF_WIDTH, CENTER + HALF_WIDTH)


def analytic_carriers(h):
    """Smooth, strictly varying electron/hole profiles across the junction."""
    u = (np.asarray(h) - CENTER) / HALF_WIDTH
    return 1e18 * np.exp(-((u - 1.0) ** 2)), 1e18 * np.exp(-((u + 1.0) ** 2))


def carrier_map(n_samples=161):
    """Carrier map sampled across the rib, in the mesh frame (x=y_layout)."""
    h = np.linspace(JUNCTION[0], JUNCTION[1], n_samples)
    z = np.linspace(0.0, RIB_HEIGHT, 5)
    hh, zz = (a.ravel() for a in np.meshgrid(h, z))
    electrons, holes = analytic_carriers(hh)
    return SimpleNamespace(x_um=hh, y_um=zz, electrons_cm3=electrons, holes_cm3=holes)


def material_of(stack, region):
    """Material properties of a region, however the stack stores them."""
    from gsim.common.stack.materials import MaterialProperties

    props = stack.materials[stack.layers[region].material]
    return (
        props
        if isinstance(props, MaterialProperties)
        else MaterialProperties.model_validate(props)
    )


def build(**kwargs):
    """Build a staircase with the shared device description."""
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    params = dict(
        n_strips=5,
        junction=JUNCTION,
        zmin=0.0,
        zmax=RIB_HEIGHT,
    )
    params.update(kwargs)
    return build_staircase_cross_section(carrier_map(), **params)


class TestOneCall:
    def test_produces_a_meshable_cross_section(self):
        staircase = build()
        stack = staircase.stack()

        assert staircase.strip_names == [f"strip_{i}" for i in range(5)]
        for name in staircase.strip_names:
            assert name in stack.layers
            assert stack.layers[name].material in stack.materials
            layer = stack.layers[name]
            assert (layer.zmin, layer.zmax) == (0.0, RIB_HEIGHT)
        drawn = {tuple(layer) for layer in staircase.component.layers}
        assert all(
            tuple(stack.layers[name].gds_layer) in drawn
            for name in staircase.strip_names
        )

    def test_strips_tile_the_junction_extent(self):
        edges = build().strips["edges_um"]
        assert edges[0] == pytest.approx(JUNCTION[0])
        assert edges[-1] == pytest.approx(JUNCTION[1])
        assert np.all(np.diff(edges) > 0)

    def test_rejects_a_junction_extent_the_carriers_do_not_cover(self):
        with pytest.raises(ValueError):
            build(junction=(CENTER + 5.0, CENTER + 6.0))


class TestBothMaterialResponses:
    def test_strips_carry_the_rf_and_the_optical_response(self):
        strips = build().strips
        for key in ("sigma_s_per_m", "dn", "dalpha_cm", "eps_complex"):
            assert len(strips[key]) == 5
        assert np.all(strips["sigma_s_per_m"] > 0)
        assert np.all(strips["dn"] < 0)  # free carriers lower the index

    def test_the_rf_stack_carries_conductivity(self):
        staircase = build()
        stack = staircase.stack("rf")
        sigmas = [
            material_of(stack, name).conductivity for name in staircase.strip_names
        ]
        assert sigmas == pytest.approx(list(staircase.strips["sigma_s_per_m"]))

    def test_the_optical_stack_carries_the_perturbed_permittivity(self):
        staircase = build(electrodes=None)
        stack = staircase.stack("optical")
        for i, name in enumerate(staircase.strip_names):
            props = material_of(stack, name)
            expected = complex(staircase.strips["eps_complex"][i])
            assert props.permittivity == pytest.approx(expected.real)
            assert props.loss_tangent > 0

    def test_the_extinction_is_built_at_the_solve_wavelength(self):
        """kappa = alpha lambda / 4 pi, and lambda is the solve's.

        The dispersion model's own wavelength is where its coefficients
        were fitted, which says what ``alpha`` is; it does not say what
        wavelength the Stage is solving at. Reading the extinction off
        the fit wavelength inflates the loss of every strip whenever the
        two differ.
        """
        fitted = build(electrodes=None).strips
        solved = build(electrodes=None, wavelength_um=1.31).strips

        # alpha is the model's answer and does not move with the solve.
        assert solved["dalpha_cm"] == pytest.approx(fitted["dalpha_cm"])
        assert solved["dn"] == pytest.approx(fitted["dn"])
        for at_fit, at_solve in zip(
            fitted["eps_complex"], solved["eps_complex"], strict=True
        ):
            assert at_solve.imag == pytest.approx(at_fit.imag * 1.31 / 1.55)

    def test_the_solve_wavelength_defaults_to_the_fitted_one(self):
        """Omitting it keeps the model's own wavelength, as before."""
        assert build(electrodes=None).strips["eps_complex"] == pytest.approx(
            build(electrodes=None, wavelength_um=1.55).strips["eps_complex"]
        )

    def test_both_stacks_share_one_component(self):
        staircase = build(electrodes=None)
        assert staircase.stack("rf") is not staircase.stack("optical")
        assert staircase.stack("rf") is staircase.stack("rf")


class TestElectrodes:
    def test_electrodes_flank_the_junction_by_default(self):
        staircase = build()
        stack = staircase.stack()

        assert len(staircase.electrode_names) == 2
        for name in staircase.electrode_names:
            assert name in stack.layers
            assert material_of(stack, name).conductivity > 1e6

    def test_electrode_geometry_follows_the_device_description(self):
        spec = ElectrodeSpec(width_um=1.5, gap_um=0.4, thickness_um=0.6)
        staircase = build(electrodes=spec)
        stack = staircase.stack()

        low, high = staircase.electrode_spans
        assert high[0] == pytest.approx(JUNCTION[1] + 0.4)
        assert high[1] == pytest.approx(JUNCTION[1] + 0.4 + 1.5)
        assert low[1] == pytest.approx(JUNCTION[0] - 0.4)
        assert low[0] == pytest.approx(JUNCTION[0] - 0.4 - 1.5)
        electrode = stack.layers[staircase.electrode_names[0]]
        assert electrode.thickness == pytest.approx(0.6)

    def test_electrodes_can_be_left_out(self):
        staircase = build(electrodes=None)
        assert staircase.electrode_names == ()
        assert staircase.electrode_spans == ()


class TestConductorModel:
    """How the electrode metal reaches the mesh (ADR 0003)."""

    def test_a_volume_electrode_is_a_region_of_lossy_metal(self):
        staircase = build()
        stack = staircase.stack("rf")

        assert staircase.conductor_model == "volume"
        for name in staircase.electrode_names:
            assert stack.layers[name].layer_type == "dielectric"
            assert material_of(stack, name).conductivity > 1e6

    def test_a_pec_electrode_is_a_conductor_layer_without_conductivity(self):
        """A conductor layer is what the native-2D mesher meshes as an
        outline, and no conductivity is what makes that outline perfect
        rather than a surface impedance."""
        staircase = build(electrodes=ElectrodeSpec(conductor_model="pec"))
        stack = staircase.stack("rf")

        assert staircase.conductor_model == "pec"
        for name in staircase.electrode_names:
            assert stack.layers[name].layer_type == "conductor"
            assert not material_of(stack, name).conductivity

    def test_a_pec_electrode_needs_no_optical_permittivity(self):
        """A perfect conductor carries no permittivity to be asked for."""
        staircase = build(electrodes=ElectrodeSpec(conductor_model="pec"))
        stack = staircase.stack("optical")
        assert stack.layers[staircase.electrode_names[0]].layer_type == "conductor"

    def test_the_model_does_not_move_the_drawn_metal(self):
        """Only how the metal is expressed changes, not where it is."""
        volume = build()
        pec = build(electrodes=ElectrodeSpec(conductor_model="pec"))
        assert pec.electrode_spans == volume.electrode_spans
        assert pec.electrode_extent(pec.electrode_names[0]) == volume.electrode_extent(
            volume.electrode_names[0]
        )


class TestElectrodeExtent:
    def test_it_reports_the_rectangle_the_electrode_occupies(self):
        spec = ElectrodeSpec(width_um=1.5, gap_um=0.4, thickness_um=0.6)
        staircase = build(electrodes=spec)

        h_span, v_span = staircase.electrode_extent("electrode_high")
        assert h_span == pytest.approx((JUNCTION[1] + 0.4, JUNCTION[1] + 0.4 + 1.5))
        assert v_span == pytest.approx((0.0, 0.6))

    def test_an_electrode_the_staircase_never_drew_is_reported(self):
        staircase = build()
        with pytest.raises(ValueError, match="no electrode named 'ground'"):
            staircase.electrode_extent("ground")

    def test_a_staircase_without_electrodes_has_no_extent(self):
        staircase = build(electrodes=None)
        assert staircase.conductor_model is None
        with pytest.raises(ValueError, match="no electrode named"):
            staircase.electrode_extent("electrode_low")


class TestOpticalElectrodes:
    def test_an_rf_electrode_is_refused_by_an_optical_stack(self):
        staircase = build()
        with pytest.raises(ValueError, match="optical_permittivity"):
            staircase.stack("optical")

    def test_the_optical_metal_permittivity_is_used_when_given(self):
        # Aluminium near 1.55 um: n = 1.44, k = 16.0.
        eps = complex((1.44 - 16.0j) ** 2)
        staircase = build(electrodes=ElectrodeSpec(optical_permittivity=eps))
        stack = staircase.stack("optical")

        props = material_of(stack, staircase.electrode_names[0])
        assert props.permittivity == pytest.approx(eps.real)
        assert props.loss_tangent == pytest.approx(-eps.imag / eps.real)

    def test_the_rf_stack_is_unaffected(self):
        staircase = build()
        stack = staircase.stack("rf")
        assert material_of(stack, staircase.electrode_names[0]).conductivity > 1e6


class TestStripCount:
    def test_one_strip_reproduces_the_uniform_model(self):
        staircase = build(n_strips=1)
        strips = staircase.strips

        assert staircase.strip_names == ["strip_0"]
        assert len(strips["edges_um"]) == 2
        h = np.linspace(*JUNCTION, 20001)
        electrons, _holes = analytic_carriers(h)
        assert strips["n_cm3"][0] == pytest.approx(
            np.trapezoid(electrons, h) / (JUNCTION[1] - JUNCTION[0]), rel=1e-3
        )

    def test_more_strips_converge_toward_the_continuous_profile(self):
        errors = []
        for n_strips in (1, 4, 16):
            strips = build(n_strips=n_strips).strips
            edges = np.asarray(strips["edges_um"])
            centres = 0.5 * (edges[1:] + edges[:-1])
            exact, _holes = analytic_carriers(centres)
            errors.append(
                float(np.mean(np.abs(strips["n_cm3"] - exact)) / np.max(exact))
            )
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 0.01


class TestDrawnGeometry:
    """What the Strips are drawn as, which is what a solver reads back.

    Both of these are the difference between the two EM Routes seeing the
    same problem and seeing two different ones: Palace resolves the drawn
    layers against the stack and honours whatever conductor it finds
    there, while femwell reads only the meshed regions.
    """

    def test_strips_are_not_drawn_on_a_generic_pdk_layer(self):
        """A Strip on a PDK metal or via layer resolves as that conductor."""
        import gdsfactory as gf

        gf.gpdk.PDK.activate()
        from gdsfactory.gpdk.layer_map import LAYER

        pdk_layers = set()
        for name in dir(LAYER):
            if name.startswith("_"):
                continue
            layer = getattr(LAYER, name)
            try:
                pdk_layers.add((int(layer.layer), int(layer.datatype)))
            except (AttributeError, TypeError, ValueError):
                continue

        staircase = build(n_strips=8)
        drawn = {
            (spec.gds_layer[0], spec.gds_layer[1])
            for name, spec in staircase._layer_specs.items()
            if name in staircase.strip_names
        }
        assert DEFAULT_STRIP_LAYER in drawn
        assert not (drawn & pdk_layers)

    @pytest.mark.parametrize("n_strips", [3, 5, 8, 16])
    def test_adjacent_strips_share_their_edge_exactly(self, n_strips):
        """No strip count may snap a sliver of background between Strips.

        Strip edges land off the GDS grid for some counts — eight strips
        across a 0.6 um junction put every centre on a half-nanometre —
        so a Strip drawn from a width and a centre can be rounded a
        nanometre away from its neighbour. Drawn from its two edges, the
        shared edge is one coordinate that rounds once.
        """
        staircase = build(n_strips=n_strips)
        component = staircase.component
        dbu = component.kcl.dbu

        spans = []
        for name in staircase.strip_names:
            spec = staircase._layer_specs[name]
            raw = component.get_polygons(layers=(tuple(spec.gds_layer),), merge=False)
            points = [
                (point.y * dbu)
                for value in raw.values()
                for polygon in (value if isinstance(value, list) else [value])
                for point in polygon.each_point_hull()
            ]
            spans.append((min(points), max(points)))

        assert len(spans) == n_strips
        spans.sort()
        for (_low, high), (next_low, _next_high) in pairwise(spans):
            assert high == next_low


def test_builds_without_any_solver_runtime(monkeypatch):
    for name in ("devsim", "femwell", "skfem", "gmsh"):
        monkeypatch.setitem(sys.modules, name, None)
    staircase = build(n_strips=3)
    assert staircase.stack("rf").layers
