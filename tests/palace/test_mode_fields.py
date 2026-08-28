"""The Marks-Williams integrals over a saved Palace boundary Mode.

Palace writes a Mode's fields as nodal values on second-order triangles,
so the power flux and the contour current are quadrature problems rather
than solver problems, and both have closed-form answers on a field
written down by hand. The fields here are analytic; what is under test
is the quadrature, the edge bookkeeping of the contour, and the sign
consistency that keeps a closed contour from cancelling itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.palace.mode_fields import (
    BoundaryModeField,
    contour_current,
    power_flux,
    z0_power_current,
)

SIDE = 4  # cells per axis of the unit square
AREA = 1.0


def square_mesh(n: int = SIDE):
    """A second-order triangulation of the unit square.

    Returns:
        ``(points, cells)`` with cells as three corners then the midside
        node of each edge, which is the layout Palace writes.
    """
    step = 1.0 / n
    corners = np.array(
        [(i * step, j * step) for j in range(n + 1) for i in range(n + 1)],
        dtype=np.float64,
    )

    def corner_id(i: int, j: int) -> int:
        return j * (n + 1) + i

    triangles = []
    for j in range(n):
        for i in range(n):
            a, b = corner_id(i, j), corner_id(i + 1, j)
            c, d = corner_id(i + 1, j + 1), corner_id(i, j + 1)
            triangles.append((a, b, c))
            triangles.append((a, c, d))

    midside: dict[tuple[int, int], int] = {}
    points = list(corners)
    cells = []
    for tri in triangles:
        row = list(tri)
        for start, end in ((0, 1), (1, 2), (2, 0)):
            key = (min(tri[start], tri[end]), max(tri[start], tri[end]))
            if key not in midside:
                midside[key] = len(points)
                points.append((corners[key[0]] + corners[key[1]]) / 2.0)
            row.append(midside[key])
        cells.append(row)
    return np.asarray(points, dtype=np.float64), np.asarray(cells, dtype=np.int64)


def field_from(e_t, h_t, *, n: int = SIDE) -> BoundaryModeField:
    """A saved Mode carrying the transverse fields two callables give."""
    points, cells = square_mesh(n)
    return BoundaryModeField(
        points_um=points,
        cells=cells,
        attribute=np.ones(cells.shape[0], dtype=np.int64),
        e_t=np.asarray(e_t(points), dtype=np.complex128),
        e_n=np.zeros(points.shape[0], dtype=np.complex128),
        h_t=np.asarray(h_t(points), dtype=np.complex128),
        h_n=np.zeros(points.shape[0], dtype=np.complex128),
    )


def uniform(vector):
    """A transverse field with the same value at every node."""

    def build(points):
        return np.broadcast_to(np.asarray(vector), (points.shape[0], 2)).copy()

    return build


def rotational(rate: float):
    """``H = rate * (-y, x)``: a field whose curl is ``2 * rate`` z-hat."""

    def build(points):
        return rate * np.stack([-points[:, 1], points[:, 0]], axis=1)

    return build


class TestPowerFlux:
    def test_a_uniform_tem_field_integrates_to_its_area(self):
        """``P = (1/2) (E x H*) . z A`` when neither field varies."""
        e0, h0 = 2.0 + 0.0j, 0.0 + 3.0j
        field = field_from(uniform([0.0, e0]), uniform([h0, 0.0]))
        expected = 0.5 * (0.0 * np.conj(0.0) - e0 * np.conj(h0)) * AREA
        assert power_flux(field) == pytest.approx(expected, rel=1e-12)

    def test_the_quadrature_is_exact_for_a_quadratic_product(self):
        """Two linear fields multiply to a quadratic; degree 4 covers it."""

        def linear_e(points):
            return np.stack([np.zeros(points.shape[0]), points[:, 0]], axis=1)

        def linear_h(points):
            return np.stack([points[:, 1], np.zeros(points.shape[0])], axis=1)

        field = field_from(linear_e, linear_h)
        # -(1/2) integral x y over the unit square.
        assert power_flux(field) == pytest.approx(-0.5 * 0.25, rel=1e-12)


class TestContourCurrent:
    def test_a_curl_free_field_encloses_no_current(self):
        """A closed contour of a constant field cancels — if it is closed."""
        field = field_from(uniform([0.0, 0.0]), uniform([1.0, 2.0]))
        current = contour_current(field, h_span=(0.25, 0.75), v_span=(0.25, 0.75))
        assert abs(current) < 1e-12

    def test_a_rotational_field_encloses_the_curl_it_carries(self):
        """Ampere's law: the contour integral is the enclosed curl."""
        rate = 0.7
        field = field_from(uniform([0.0, 0.0]), rotational(rate))
        span = (0.25, 0.75)
        current = contour_current(field, h_span=span, v_span=span)
        enclosed = 2.0 * rate * (span[1] - span[0]) ** 2
        assert abs(current) == pytest.approx(enclosed, rel=1e-12)

    def test_the_contour_does_not_depend_on_where_it_is_drawn(self):
        """Any contour around the same enclosed curl reads the same."""
        field = field_from(uniform([0.0, 0.0]), rotational(0.7))
        inner = abs(contour_current(field, h_span=(0.25, 0.75), v_span=(0.25, 0.75)))
        outer = abs(contour_current(field, h_span=(0.0, 1.0), v_span=(0.0, 1.0)))
        # Four times the area, four times the enclosed curl.
        assert outer == pytest.approx(4.0 * inner, rel=1e-12)

    def test_a_rectangle_off_the_mesh_is_reported(self):
        field = field_from(uniform([0.0, 0.0]), uniform([1.0, 0.0]))
        with pytest.raises(ValueError, match="No mesh edge lies on the rectangle"):
            contour_current(field, h_span=(2.0, 3.0), v_span=(2.0, 3.0))


class TestZ0PowerCurrent:
    def test_it_divides_the_power_by_the_squared_current(self):
        rate = 0.7
        field = field_from(uniform([0.0, 1.0]), rotational(rate))
        span = (0.25, 0.75)
        z0 = z0_power_current(field, h_span=span, v_span=span)
        expected = (
            2.0
            * power_flux(field)
            / abs(contour_current(field, h_span=span, v_span=span)) ** 2
        )
        assert z0 == pytest.approx(expected, rel=1e-12)

    def test_a_mode_carrying_no_current_is_reported(self):
        field = field_from(uniform([0.0, 1.0]), uniform([1.0, 2.0]))
        with pytest.raises(ValueError, match="no current"):
            z0_power_current(field, h_span=(0.25, 0.75), v_span=(0.25, 0.75))
