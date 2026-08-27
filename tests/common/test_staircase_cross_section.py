"""One call from a Carrier map to a meshable Staircase cross-section."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gsim.common.stack.staircase import (
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


def test_builds_without_any_solver_runtime(monkeypatch):
    for name in ("devsim", "femwell", "skfem", "gmsh"):
        monkeypatch.setitem(sys.modules, name, None)
    staircase = build(n_strips=3)
    assert staircase.stack("rf").layers
