"""Unit tests for gmsh utility curve-loop fitting helpers."""

from __future__ import annotations

import math

from gsim.palace.mesh.gmsh_utils import _create_wire_loop


class _FakeKernel:
    """Tiny gmsh-kernel stub implementing the methods used by _create_wire_loop."""

    def __init__(self) -> None:
        """Initialize deterministic tag allocation and captured curve primitives."""
        self._next_tag = 1
        self.points: dict[int, tuple[float, float, float, float]] = {}
        self.bsplines: list[list[int]] = []
        self.splines: list[list[int]] = []
        self.lines: list[tuple[int, int]] = []
        self.curve_loops: list[list[int]] = []

    def _new_tag(self) -> int:
        """Return a fresh integer tag."""
        tag = self._next_tag
        self._next_tag += 1
        return tag

    def addPoint(  # noqa: N802
        self,
        x: float,
        y: float,
        z: float,
        meshseed: float,
        tag: int,
    ) -> int:
        """Record a gmsh-like point entity."""
        del tag
        point_tag = self._new_tag()
        self.points[point_tag] = (x, y, z, meshseed)
        return point_tag

    def addBSpline(self, curve_pts: list[int], tag: int) -> int:  # noqa: N802
        """Record a gmsh-like B-spline entity."""
        del tag
        self.bsplines.append(list(curve_pts))
        return self._new_tag()

    def addSpline(self, curve_pts: list[int], tag: int) -> int:  # noqa: N802
        """Record a gmsh-like spline entity."""
        del tag
        self.splines.append(list(curve_pts))
        return self._new_tag()

    def addLine(self, p1: int, p2: int, tag: int) -> int:  # noqa: N802
        """Record a gmsh-like line entity."""
        del tag
        self.lines.append((p1, p2))
        return self._new_tag()

    def addCurveLoop(self, curves: list[int], tag: int) -> int:  # noqa: N802
        """Record a gmsh-like curve-loop entity."""
        del tag
        self.curve_loops.append(list(curves))
        return self._new_tag()


def test_bspline_curve_fit_splits_at_large_angles() -> None:
    """Sharp corners stay fixed while smooth spans are fitted as B-splines."""
    kernel = _FakeKernel()

    # Polygon with two sharp corners and one long smooth span in between.
    pts_x = [0.0, 4.0, 4.3, 4.2, 3.8, 3.2, 2.4, 0.0]
    pts_y = [0.0, 0.0, 0.8, 1.6, 2.3, 2.8, 3.0, 3.0]

    loop_tag = _create_wire_loop(
        kernel,
        pts_x,
        pts_y,
        z=0.0,
        loop_mode="bspline",
    )

    assert loop_tag is not None
    assert kernel.curve_loops
    assert kernel.bsplines

    # Segmented fitting keeps B-spline spans open (no duplicated closure point).
    assert all(span[0] != span[-1] for span in kernel.bsplines)

    # The long smooth span should be fitted between the sharp-corner anchors.
    # Point tags are assigned in vertex order, so vertex 1 -> tag 2, vertex 7 -> tag 8.
    assert any(span[0] == 2 and span[-1] == 8 for span in kernel.bsplines)


def test_bspline_curve_fit_keeps_closed_fit_for_smooth_loop() -> None:
    """Smooth contours should still use a single closed B-spline loop."""
    kernel = _FakeKernel()

    n = 24
    pts_x = [math.cos(2.0 * math.pi * i / n) for i in range(n)]
    pts_y = [math.sin(2.0 * math.pi * i / n) for i in range(n)]

    loop_tag = _create_wire_loop(
        kernel,
        pts_x,
        pts_y,
        z=0.0,
        loop_mode="bspline",
    )

    assert loop_tag is not None
    assert len(kernel.bsplines) == 1
    assert kernel.bsplines[0][0] == kernel.bsplines[0][-1]
    assert not kernel.lines


def test_bspline_corner_split_threshold_is_tunable() -> None:
    """Increasing threshold suppresses corner splitting for the same contour."""
    kernel = _FakeKernel()

    pts_x = [0.0, 4.0, 4.3, 4.2, 3.8, 3.2, 2.4, 0.0]
    pts_y = [0.0, 0.0, 0.8, 1.6, 2.3, 2.8, 3.0, 3.0]

    loop_tag = _create_wire_loop(
        kernel,
        pts_x,
        pts_y,
        z=0.0,
        loop_mode="bspline",
        corner_turn_threshold_deg=170.0,
    )

    assert loop_tag is not None
    assert len(kernel.bsplines) == 1
    assert kernel.bsplines[0][0] == kernel.bsplines[0][-1]
