"""Hermetic tests for the staircase strip builder and node-to-strip transfer."""

from __future__ import annotations

from itertools import pairwise

import gdsfactory as gf
import numpy as np
import pytest

from gsim.common.carriers import (
    PlasmaDispersionModel,
    carrier_absorption_cm,
    carrier_conductivity,
    carrier_index_shift,
)
from gsim.common.stack.staircase import (
    make_staircase_profile,
    strip_averages_from_nodes,
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


def _make_profile(comp, edges, n_vals, p_vals, **kwargs):
    return make_staircase_profile(
        comp,
        length=10.0,
        edges=edges,
        n_strips_cm3=n_vals,
        p_strips_cm3=p_vals,
        base_layer=(40, 0),
        zmin=0.0,
        zmax=0.22,
        **kwargs,
    )


class TestMakeStaircaseProfileRF:
    def test_single_strip_recovers_uniform_model(self):
        gf.gpdk.PDK.activate()
        comp = gf.Component()
        result = _make_profile(comp, [-0.2, 0.2], [1e18], [0.0])

        # One region spanning the window with the uniform-model conductivity.
        assert set(result["layer_specs"]) == {"strip_0"}
        spec = result["layer_specs"]["strip_0"]
        assert spec.gds_layer == (40, 0)
        assert spec.zmin == 0.0
        assert spec.zmax == pytest.approx(0.22)
        assert result["centres"]["strip_0"] == pytest.approx(0.0)
        expected_sigma = carrier_conductivity(1e18, 0.0)
        assert result["strips"]["sigma_s_per_m"][0] == pytest.approx(expected_sigma)
        assert "strip_0" in result["materials"]

    def test_arbitrary_strip_count_layers_and_materials(self):
        gf.gpdk.PDK.activate()
        comp = gf.Component()
        n = 7
        edges = np.linspace(-0.35, 0.35, n + 1)
        n_vals = np.linspace(0.0, 1e18, n)
        p_vals = np.linspace(1e18, 0.0, n)
        result = _make_profile(comp, edges, n_vals, p_vals)

        assert len(result["layer_specs"]) == n
        for i in range(n):
            name = f"strip_{i}"
            assert result["layer_specs"][name].gds_layer == (40, i)
            assert name in result["materials"]
            assert result["centres"][name] == pytest.approx(
                (edges[i] + edges[i + 1]) / 2
            )
        np.testing.assert_allclose(
            result["strips"]["sigma_s_per_m"],
            carrier_conductivity(n_vals, p_vals),
        )

    def test_rejects_mismatched_strip_values(self):
        comp = gf.Component()
        with pytest.raises(ValueError, match="per-strip"):
            _make_profile(comp, [-0.2, 0.0, 0.2], [1e18], [0.0])

    def test_rejects_descending_edges(self):
        comp = gf.Component()
        with pytest.raises(ValueError, match="ascending"):
            _make_profile(comp, [0.2, -0.2], [1e18], [1e18])


class TestMakeStaircaseProfileOptical:
    def test_soref_permittivity_and_loss(self):
        gf.gpdk.PDK.activate()
        comp = gf.Component()
        model = PlasmaDispersionModel.nedeljkovic_1550()
        n0 = 3.4757
        result = _make_profile(
            comp,
            [-0.1, 0.1],
            [1e18],
            [1e18],
            target="optical",
            dispersion=model,
            n0=n0,
        )
        material = result["materials"]["strip_0"]
        dn = carrier_index_shift(1e18, 1e18, model=model)
        dalpha = carrier_absorption_cm(1e18, 1e18, model=model)
        # Carrier-depressed index: eps_re < n0^2, loss tangent positive.
        assert material.permittivity == pytest.approx((n0 + dn) ** 2, rel=1e-3)
        assert material.permittivity < n0**2
        assert material.loss_tangent > 0.0
        assert dalpha > 0.0
        np.testing.assert_allclose(result["strips"]["dn"], [dn])

    def test_optical_requires_dispersion_model(self):
        comp = gf.Component()
        with pytest.raises(ValueError, match="dispersion"):
            _make_profile(comp, [-0.1, 0.1], [1e18], [1e18], target="optical")
