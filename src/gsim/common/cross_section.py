"""Solver-agnostic XZ cross-section extractor.

Given a gdsfactory component and a LayerStack, produce a list of
axis-aligned rectangles in the XZ plane sliced at ``Y=y_cut``.
These rectangles are what an XZ 2D FDTD simulation extrudes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import gdsfactory as gf

    from gsim.common.stack import LayerStack


@dataclass(frozen=True)
class Rect2D:
    """Axis-aligned rectangle in the XZ plane.

    Attributes:
        x0: Low X extent (um).
        x1: High X extent (um).
        zmin: Low Z extent (um).
        zmax: High Z extent (um).
        layer_name: Source layer name from the LayerStack.
        material: Material name from the LayerStack layer.
    """

    x0: float
    x1: float
    zmin: float
    zmax: float
    layer_name: str
    material: str


def extract_xz_rectangles(
    component: gf.Component,
    layer_stack: LayerStack,
    y_cut: float,
    *,
    eps: float = 1e-9,
) -> list[Rect2D]:
    """Slice ``component`` at ``Y=y_cut``; return one Rect2D per layer-interval.

    For each layer in ``layer_stack`` with a GDS layer tuple:

    1. Pull polygons for that GDS layer from the component.
    2. Intersect each polygon (with holes) with the horizontal line Y=y_cut
       using shapely.
    3. Union the resulting 1D X-intervals within that layer.
    4. Emit one Rect2D per interval at the layer's (zmin, zmax).

    Args:
        component: gdsfactory Component (may contain references).
        layer_stack: LayerStack describing which layers to extract.
        y_cut: Y coordinate of the cross-section (um).
        eps: Drop intervals shorter than this (um) -- filters out zero-length
            cuts that hit a polygon edge exactly.

    Returns:
        List of Rect2D in layer-stack order, unmerged across layers.
        Layers with no intersection are skipped.
    """
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    dbu = getattr(getattr(component, "kcl", None), "dbu", 0.001)

    rects: list[Rect2D] = []

    for layer_name, layer in layer_stack.layers.items():
        gds_layer = getattr(layer, "gds_layer", None)
        if gds_layer is None:
            continue
        gds_layer_tuple = (int(gds_layer[0]), int(gds_layer[1]))

        shapely_polys = _layer_shapely_polys(component, gds_layer_tuple, dbu)
        if not shapely_polys:
            continue

        merged = unary_union(shapely_polys)
        if merged.is_empty:
            continue

        minx, miny, maxx, maxy = merged.bounds
        if y_cut < miny - eps or y_cut > maxy + eps:
            continue

        cut_line = LineString([(minx - 1.0, y_cut), (maxx + 1.0, y_cut)])
        intersection = merged.intersection(cut_line)

        intervals = _line_intervals(intersection)

        for x0, x1 in intervals:
            if x1 - x0 <= eps:
                continue
            rects.append(
                Rect2D(
                    x0=x0,
                    x1=x1,
                    zmin=layer.zmin,
                    zmax=layer.zmax,
                    layer_name=layer_name,
                    material=layer.material,
                )
            )

    return rects


def _layer_shapely_polys(component, gds_layer_tuple, dbu):
    """Return a list of shapely Polygons for one GDS layer of ``component``."""
    from shapely.geometry import Polygon

    raw = component.get_polygons(layers=(gds_layer_tuple,), merge=True)
    if not isinstance(raw, dict) or not raw:
        return []

    polys: list = []
    for value in raw.values():
        items = list(value) if isinstance(value, list) else [value]
        for obj in items:
            exterior, holes = _poly_to_coords(obj, dbu)
            if exterior is None or len(exterior) < 3:
                continue
            try:
                poly = Polygon(exterior, holes=holes)
            except (ValueError, TypeError) as err:
                logger.debug("Skipping invalid polygon: %s", err)
                continue
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            if hasattr(poly, "geoms"):
                polys.extend(poly.geoms)
            else:
                polys.append(poly)
    return polys


def _poly_to_coords(obj, dbu):
    """Return (exterior_coords, list_of_hole_coords) from a polygon-ish object.

    Handles KLayout ``PolygonWithProperties`` objects (via ``each_point_hull``
    / ``each_point_hole``) and legacy numpy-array polygons (no holes).
    """
    # KLayout polygon with optional holes.
    if hasattr(obj, "each_point_hull"):
        exterior = [(pt.x * dbu, pt.y * dbu) for pt in obj.each_point_hull()]
        holes: list = []
        try:
            n_holes = obj.holes()
        except AttributeError:
            n_holes = 0
        for i in range(n_holes):
            try:
                holes.append(
                    [(pt.x * dbu, pt.y * dbu) for pt in obj.each_point_hole(i)]
                )
            except (AttributeError, IndexError) as err:
                logger.debug("Skipping malformed hole %d: %s", i, err)
                continue
        return exterior, holes

    # Legacy numpy / iterable of (x, y) points.
    if hasattr(obj, "__iter__"):
        try:
            return [(float(p[0]), float(p[1])) for p in obj], []
        except (TypeError, IndexError):
            return None, []
    return None, []


def _line_intervals(intersection) -> list[tuple[float, float]]:
    """Extract sorted, merged (x0, x1) intervals from a shapely intersection.

    Handles LineString, MultiLineString, empty, and GeometryCollection results.
    """
    from shapely.geometry import LineString, MultiLineString

    if intersection.is_empty:
        return []

    lines: list = []
    if isinstance(intersection, LineString):
        lines = [intersection]
    elif isinstance(intersection, MultiLineString):
        lines = list(intersection.geoms)
    else:
        for geom in getattr(intersection, "geoms", []):
            if isinstance(geom, LineString):
                lines.append(geom)

    intervals: list[tuple[float, float]] = []
    for line in lines:
        xs = [coord[0] for coord in line.coords]
        intervals.append((min(xs), max(xs)))

    intervals.sort()

    merged: list[list[float]] = []
    for x0, x1 in intervals:
        if merged and x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    return [(a, b) for a, b in merged]
