"""Auto-sizing heuristics for mesh generation."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack

logger = logging.getLogger(__name__)


def min_conductor_feature_size(component, stack: LayerStack) -> float | None:
    """Smallest polygon bbox dimension across all conductor layers (um).

    Scans every polygon on every conductor-typed layer in the stack and
    returns the smallest of its bbox width / height. Used as a proxy for
    the minimum feature that mesh elements must resolve.

    Returns None if the component has no polygons on conductor layers, or
    if the stack declares no conductors.
    """
    conductor_gds: set[tuple[int, int]] = {
        tuple(layer.gds_layer)
        for layer in stack.layers.values()
        if layer.layer_type == "conductor"
    }
    if not conductor_gds:
        return None

    layout = component.kcl.layout
    index_to_gds: dict[int, tuple[int, int]] = {}
    for layer_index in range(layout.layers()):
        if layout.is_valid_layer(layer_index):
            info = layout.get_info(layer_index)
            index_to_gds[layer_index] = (info.layer, info.datatype)

    min_dim = math.inf
    polygons_by_index = component.get_polygons()
    for layer_index, polys in polygons_by_index.items():
        gds_tuple = index_to_gds.get(layer_index)
        if gds_tuple is None or gds_tuple not in conductor_gds:
            continue
        for poly in polys:
            points = list(poly.each_point_hull())
            if len(points) < 3:
                continue
            xs = [pt.x / 1000.0 for pt in points]
            ys = [pt.y / 1000.0 for pt in points]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            if w > 0:
                min_dim = min(min_dim, w)
            if h > 0:
                min_dim = min(min_dim, h)

    return min_dim if math.isfinite(min_dim) else None


def auto_refined_mesh_size(
    component,
    stack: LayerStack,
    preset_size: float,
    cells_per_feature: int = 4,
) -> float:
    """Pick ``refined_mesh_size`` scaled to the smallest conductor feature.

    Returns ``min(preset_size, min_feature / cells_per_feature)`` so designs
    with small features get proportionally refined meshes while large designs
    keep the preset's size. Falls back to ``preset_size`` when no conductor
    polygons are found.
    """
    min_feature = min_conductor_feature_size(component, stack)
    if min_feature is None or cells_per_feature <= 0:
        return preset_size
    return min(preset_size, min_feature / cells_per_feature)


__all__ = [
    "auto_refined_mesh_size",
    "min_conductor_feature_size",
]
