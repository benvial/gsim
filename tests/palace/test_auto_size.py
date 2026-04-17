"""Tests for geometry-aware mesh auto-sizing."""

from __future__ import annotations

import gdsfactory as gf
import pytest

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace.mesh.auto_size import (
    auto_refined_mesh_size,
    min_conductor_feature_size,
)


@pytest.fixture(autouse=True)
def _activate_pdk():
    """Activate the generic PDK for all tests."""
    gf.gpdk.PDK.activate()


def _conductor_stack(gds_layer: tuple[int, int] = (1, 0)) -> LayerStack:
    """Return a stack with a single conductor layer."""
    stack = LayerStack()
    stack.layers["metal"] = Layer(
        name="metal",
        gds_layer=gds_layer,
        zmin=0.0,
        zmax=0.5,
        thickness=0.5,
        material="copper",
        layer_type="conductor",
    )
    return stack


def _narrow_trace(width: float = 2.0, length: float = 100.0) -> gf.Component:
    """Create a simple rectangular trace on layer (1, 0)."""
    c = gf.Component()
    half_w = width / 2
    half_l = length / 2
    c.add_polygon(
        [(-half_l, -half_w), (half_l, -half_w), (half_l, half_w), (-half_l, half_w)],
        layer=(1, 0),
    )
    return c


class TestMinConductorFeatureSize:
    """Tests for min_conductor_feature_size."""

    def test_narrow_trace_returns_width(self):
        component = _narrow_trace(width=2.0, length=100.0)
        stack = _conductor_stack()
        assert min_conductor_feature_size(component, stack) == 2.0

    def test_returns_none_when_no_conductors(self):
        component = _narrow_trace()
        empty_stack = LayerStack()
        assert min_conductor_feature_size(component, empty_stack) is None

    def test_ignores_non_conductor_layers(self):
        # Narrow polygon on a layer that's NOT a conductor must be ignored.
        component = _narrow_trace(width=2.0)
        stack = _conductor_stack(gds_layer=(99, 0))
        assert min_conductor_feature_size(component, stack) is None


class TestAutoRefinedMeshSize:
    """Tests for auto_refined_mesh_size."""

    def test_scales_down_for_small_features(self):
        component = _narrow_trace(width=2.0)
        stack = _conductor_stack()
        # min_feature=2um, cells_per_feature=4 -> 0.5um < preset 5.0
        assert auto_refined_mesh_size(component, stack, preset_size=5.0) == 0.5

    def test_caps_at_preset_for_large_features(self):
        component = _narrow_trace(width=100.0, length=100.0)
        stack = _conductor_stack()
        # min_feature=100um, /4 = 25um, but capped at preset 5.0
        assert auto_refined_mesh_size(component, stack, preset_size=5.0) == 5.0

    def test_falls_back_to_preset_when_no_conductors(self):
        component = _narrow_trace()
        empty_stack = LayerStack()
        assert auto_refined_mesh_size(component, empty_stack, preset_size=5.0) == 5.0
