"""Hermetic tests for Strip binning and the node-to-strip transfer.

Everything that reduces a Carrier map to per-Strip averages is under
test here: the one-dimensional binning, the two-dimensional node cloud
reduced onto it, and the Strips a Staircase draws from the result.
"""

from __future__ import annotations

from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from gsim.common.carriers import (
    PlasmaDispersionModel,
    carrier_absorption_cm,
    carrier_conductivity,
    carrier_index_shift,
)
from gsim.common.stack.staircase import (
    build_staircase_cross_section,
    staircase_profile,
    strip_averages_from_nodes,
)


class TestStaircaseProfile:
    def test_single_bin_recovers_average(self):
        # N=1 must recover the exact average of the piecewise-linear profile.
        h = np.array([0.0, 1.0, 2.0])
        v = np.array([0.0, 2.0, 0.0])  # triangle, mean = 1.0
        edges, means = staircase_profile(h, v, n_bins=1)
        assert edges == pytest.approx([0.0, 2.0])
        assert means == pytest.approx([1.0])

    def test_constant_profile_any_n(self):
        h = np.linspace(0.0, 4.0, 9)
        v = np.full(9, 7.5)
        edges, means = staircase_profile(h, v, n_bins=5)
        assert len(edges) == 6
        assert means == pytest.approx(np.full(5, 7.5))

    def test_bins_partition_window(self):
        h = np.linspace(-1.0, 1.0, 21)
        v = h**2
        edges, _means = staircase_profile(h, v, n_bins=4, h_min=-0.5, h_max=0.5)
        assert edges[0] == pytest.approx(-0.5)
        assert edges[-1] == pytest.approx(0.5)
        assert np.all(np.diff(edges) > 0)

    def test_convergence_with_n(self):
        # Staircase approximation of a smooth profile converges in L2 as N grows.
        h = np.linspace(0.0, 1.0, 401)
        v = np.exp(-((h - 0.5) ** 2) / 0.01)

        def l2_error(n_bins: int) -> float:
            edges, means = staircase_profile(h, v, n_bins=n_bins)
            approx = np.interp(h, edges[:-1], means, left=means[0], right=means[-1])
            # Evaluate staircase exactly: index of bin per sample.
            idx = np.clip(np.searchsorted(edges, h, side="right") - 1, 0, n_bins - 1)
            approx = means[idx]
            return float(np.sqrt(np.trapezoid((approx - v) ** 2, h)))

        errors = [l2_error(n) for n in (2, 8, 32)]
        assert errors[0] > errors[1] > errors[2]

    def test_unsorted_input_sorted_internally(self):
        h = np.array([2.0, 0.0, 1.0])
        v = np.array([0.0, 0.0, 2.0])
        _edges, means = staircase_profile(h, v, n_bins=1)
        assert means == pytest.approx([1.0])

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            staircase_profile(np.array([0.0]), np.array([1.0]), n_bins=1)
        with pytest.raises(ValueError):
            staircase_profile(np.array([0.0, 1.0]), np.array([1.0, 1.0]), n_bins=0)
        with pytest.raises(ValueError):
            staircase_profile(
                np.array([0.0, 1.0]),
                np.array([1.0, 1.0]),
                n_bins=2,
                h_min=1.0,
                h_max=0.0,
            )


class TestStripAveragesFromNodes:
    def test_single_strip_recovers_mean_of_linear_profile(self):
        h = np.linspace(0.0, 1.0, 101)
        values = 2.0 * h  # mean 1.0
        edges, means = strip_averages_from_nodes(h, values, n_strips=1)
        np.testing.assert_allclose(edges, [0.0, 1.0])
        assert means[0] == pytest.approx(1.0)

    def test_vertical_band_filters_2d_node_cloud(self):
        # Two rows of nodes; only the y ~ 0 row carries the profile.
        h = np.concatenate([np.linspace(0.0, 1.0, 51), np.linspace(0.0, 1.0, 51)])
        v = np.concatenate([np.zeros(51), np.ones(51)])
        values = np.concatenate([np.linspace(0.0, 2.0, 51), np.full(51, 100.0)])
        _edges, means = strip_averages_from_nodes(
            h, values, n_strips=1, v_um=v, v_range=(-0.1, 0.1)
        )
        assert means[0] == pytest.approx(1.0)

    def test_error_decreases_with_strip_count(self):
        h = np.linspace(-1.0, 1.0, 401)
        values = np.tanh(5.0 * h)

        def reconstruction_error(n_strips):
            edges, means = strip_averages_from_nodes(h, values, n_strips=n_strips)
            idx = np.clip(np.searchsorted(edges, h, side="right") - 1, 0, n_strips - 1)
            return float(np.sqrt(np.mean((means[idx] - values) ** 2)))

        errors = [reconstruction_error(n) for n in (1, 4, 16, 64)]
        assert errors == sorted(errors, reverse=True)
        assert errors[-1] < 0.05 * errors[0]

    def test_values_sharing_a_coordinate_are_averaged(self):
        # Two rows inside the band carrying different fields: the strip value
        # is the average over the band, not whichever node survives dedup.
        h_row = np.linspace(0.0, 1.0, 51)
        h = np.concatenate([h_row, h_row])
        v = np.concatenate([np.zeros(51), np.full(51, 0.05)])
        values = np.concatenate([np.zeros(51), 2.0 * h_row])
        _edges, means = strip_averages_from_nodes(
            h, values, n_strips=1, v_um=v, v_range=(-0.1, 0.1)
        )
        assert means[0] == pytest.approx(0.5)

    def test_strips_track_the_band_averaged_profile(self):
        # A field varying across the band as well as along it: each strip
        # must land on the band average, not on one row.
        h_row = np.linspace(0.0, 1.0, 41)
        rows = np.linspace(0.0, 0.1, 5)
        h = np.tile(h_row, rows.size)
        v = np.repeat(rows, h_row.size)
        values = h + 10.0 * v
        _edges, means = strip_averages_from_nodes(
            h, values, n_strips=4, v_um=v, v_range=(-0.01, 0.11)
        )
        expected = np.array([0.125, 0.375, 0.625, 0.875]) + 10.0 * rows.mean()
        np.testing.assert_allclose(means, expected, rtol=1e-6)

    def test_an_irregular_cloud_lands_on_the_analytic_average(self):
        # What a charge-solve mesh actually hands over: columns of unequal
        # height, at coordinates agreeing only to rounding, carrying a
        # field that varies along the band as well as across it.
        rng = np.random.default_rng(0)
        columns = np.linspace(-0.5, 0.5, 240)
        h_parts, v_parts, value_parts = [], [], []
        for h in columns:
            rows = rng.integers(3, 12)
            z = rng.uniform(0.0, 0.22, rows)
            jitter = rng.normal(scale=1e-12, size=rows)
            h_parts.append(np.full(rows, h) + jitter)
            v_parts.append(z)
            value_parts.append(np.exp(-10.0 * h**2) + 4.0 * z)
        h = np.concatenate(h_parts)
        v = np.concatenate(v_parts)
        values = np.concatenate(value_parts)

        edges, means = strip_averages_from_nodes(
            h, values, n_strips=6, v_um=v, v_range=(0.0, 0.22)
        )

        # The band average of the field is exp(-10 h^2) + 4 * mean(z),
        # integrated over each strip.
        dense = np.linspace(-0.5, 0.5, 20001)
        profile = np.exp(-10.0 * dense**2) + 4.0 * 0.11
        expected = [
            profile[(dense >= lo) & (dense <= hi)].mean() for lo, hi in pairwise(edges)
        ]
        np.testing.assert_allclose(means, expected, rtol=0.05)

    def test_band_without_nodes_raises(self):
        h = np.linspace(0.0, 1.0, 11)
        with pytest.raises(ValueError, match="v_range"):
            strip_averages_from_nodes(
                h, h, n_strips=1, v_um=np.zeros(11), v_range=(5.0, 6.0)
            )

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            strip_averages_from_nodes([0.0, 1.0], [1.0], n_strips=1)


def _carriers(edges, n_of_h, p_of_h, *, samples=201):
    """A one-row Carrier map across ``edges``, in the mesh frame."""
    h = np.linspace(edges[0], edges[-1], samples)
    return SimpleNamespace(
        x_um=h, y_um=np.full(h.size, 0.11), electrons_cm3=n_of_h(h), holes_cm3=p_of_h(h)
    )


def _build(carriers, n_strips, **kwargs):
    import gdsfactory as gf

    gf.gpdk.PDK.activate()
    params = dict(
        n_strips=n_strips,
        junction=(float(carriers.x_um[0]), float(carriers.x_um[-1])),
        zmin=0.0,
        zmax=0.22,
        electrodes=None,
        base_layer=(40, 0),
    )
    params.update(kwargs)
    return build_staircase_cross_section(carriers, **params)


def _material(stack, name):
    from gsim.common.stack.materials import MaterialProperties

    props = stack.materials[stack.layers[name].material]
    return (
        props
        if isinstance(props, MaterialProperties)
        else MaterialProperties.model_validate(props)
    )


class TestDrawnStripsRF:
    def test_single_strip_recovers_uniform_model(self):
        carriers = _carriers(
            [-0.2, 0.2], lambda h: np.full(h.size, 1e18), np.zeros_like
        )
        staircase = _build(carriers, 1)
        stack = staircase.stack("rf")

        # One region spanning the window with the uniform-model conductivity.
        assert staircase.strip_names == ["strip_0"]
        spec = stack.layers["strip_0"]
        assert spec.gds_layer == (40, 0)
        assert spec.zmin == 0.0
        assert spec.zmax == pytest.approx(0.22)
        expected_sigma = carrier_conductivity(1e18, 0.0)
        assert staircase.strips["sigma_s_per_m"][0] == pytest.approx(expected_sigma)
        assert _material(stack, "strip_0").conductivity == pytest.approx(expected_sigma)

    def test_arbitrary_strip_count_layers_and_materials(self):
        n = 7
        edges = np.linspace(-0.35, 0.35, n + 1)
        rise = lambda h: 1e18 * (h - edges[0]) / (edges[-1] - edges[0])  # noqa: E731
        fall = lambda h: 1e18 - rise(h)  # noqa: E731
        staircase = _build(_carriers(edges, rise, fall), n)
        stack = staircase.stack("rf")

        assert len(staircase.strip_names) == n
        centres = 0.5 * (edges[1:] + edges[:-1])
        for i, name in enumerate(staircase.strip_names):
            assert stack.layers[name].gds_layer == (40, i)
            assert stack.layers[name].material in stack.materials
        # Linear profiles: each strip average is the value at its centre.
        np.testing.assert_allclose(
            staircase.strips["sigma_s_per_m"],
            carrier_conductivity(rise(centres), fall(centres)),
            rtol=1e-6,
        )

    def test_rejects_a_descending_extent(self):
        carriers = _carriers(
            [-0.2, 0.2], lambda h: np.full(h.size, 1e18), np.zeros_like
        )
        with pytest.raises(ValueError, match="ascending"):
            _build(carriers, 1, junction=(0.2, -0.2))


class TestDrawnStripsOptical:
    def test_plasma_dispersion_permittivity_and_loss(self):
        model = PlasmaDispersionModel.nedeljkovic_1550()
        n0 = 3.4757
        carriers = _carriers(
            [-0.1, 0.1],
            lambda h: np.full(h.size, 1e18),
            lambda h: np.full(h.size, 1e18),
        )
        staircase = _build(carriers, 1, dispersion=model, n0=n0)
        material = _material(staircase.stack("optical"), "strip_0")
        dn = carrier_index_shift(1e18, 1e18, model=model)
        dalpha = carrier_absorption_cm(1e18, 1e18, model=model)
        # Carrier-depressed index: eps_re < n0^2, loss tangent positive.
        assert material.permittivity == pytest.approx((n0 + dn) ** 2, rel=1e-3)
        assert material.permittivity < n0**2
        assert material.loss_tangent > 0.0
        assert dalpha > 0.0
        np.testing.assert_allclose(staircase.strips["dn"], [dn])
