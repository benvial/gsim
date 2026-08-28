"""The Palace Route on the real pipeline, and its agreement with femwell.

The cross-Route gate the modulator API owes the originating spec: both
first-class Routes solve the identical Staircase on the identical mesh,
and must land on the same effective index to solver tolerance. It ships
here as a runtime-gated test rather than as a one-off script, so the
agreement is re-checked whenever either Route moves.

Gated on gmsh, the femwell runtime and a Palace binary; deselected by
default, run with ``pytest -m palace_local``. The Carrier maps are
synthetic, so nothing here needs DEVSIM.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.modulator import Device, Study

from .conftest import CENTER_Y, HALF_WIDTH, PAD_WIDTH, RIB_HEIGHT, build_demo

pytest.importorskip("gmsh")
pytest.importorskip("femwell")
pytest.importorskip("skfem")

from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap

pytestmark = pytest.mark.palace_local

DOPING_CM3 = 1e18
DEPLETED_CM3 = 1e10
SLAB = (CENTER_Y - HALF_WIDTH - PAD_WIDTH, CENTER_Y + HALF_WIDTH + PAD_WIDTH)
# Palace solves order-2 Nedelec elements against femwell's order-1
# Lagrange: the same problem on different element bases, so agreement
# is to the same 1% the shipped Palace/femwell cross-validation pins.
N_EFF_RTOL = 1e-2


def _palace_available() -> bool:
    from gsim.palace.runtime import resolve_palace_binary

    try:
        return resolve_palace_binary() is not None
    except Exception:
        return False


if not _palace_available():  # pragma: no cover - environment dependent
    pytest.skip("Palace binary not available", allow_module_level=True)


def depletion_carriers(bias_v: float) -> CarrierMap:
    """A Carrier map whose depletion region widens with reverse bias."""
    y = np.linspace(SLAB[0], SLAB[1], 121)
    z = np.linspace(0.0, RIB_HEIGHT, 9)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    yy, zz = yy.ravel(), zz.ravel()

    depleted = np.abs(yy - CENTER_Y) < 0.05 * np.sqrt(1.0 + abs(bias_v))
    n_side = yy < CENTER_Y
    return CarrierMap(
        x_um=yy,
        y_um=zz,
        region=["n_rib" if side else "p_rib" for side in n_side],
        electrons_cm3=np.where(n_side & ~depleted, DOPING_CM3, DEPLETED_CM3),
        holes_cm3=np.where(~n_side & ~depleted, DOPING_CM3, DEPLETED_CM3),
        potential_v=np.zeros(yy.size),
        net_doping_cm3=np.zeros(yy.size),
    )


def graded_carriers(bias_v: float = 0.0) -> CarrierMap:  # noqa: ARG001
    """A smoothly graded Carrier map, for measuring staircase error.

    A step-like junction is either resolved by a strip edge or not, so the
    staircase error jumps rather than shrinking; a profile that varies
    smoothly across the Junction is the one whose approximation error a
    strip count can actually be said to converge on.
    """
    y = np.linspace(SLAB[0], SLAB[1], 161)
    z = np.linspace(0.0, RIB_HEIGHT, 9)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    yy, zz = yy.ravel(), zz.ravel()

    p_fraction = 1.0 / (1.0 + np.exp(-(yy - CENTER_Y) / 0.25))
    return CarrierMap(
        x_um=yy,
        y_um=zz,
        region=["n_rib" if side else "p_rib" for side in yy < CENTER_Y],
        electrons_cm3=DOPING_CM3 * (1.0 - p_fraction) + DEPLETED_CM3,
        holes_cm3=DOPING_CM3 * p_fraction + DEPLETED_CM3,
        potential_v=np.zeros(yy.size),
        net_doping_cm3=np.zeros(yy.size),
    )


def study_at(tmp_path, *, biases=(0.0,), carriers=depletion_carriers) -> Study:
    """A Study whose charge Stage already holds a synthetic Bias sweep."""
    demo = build_demo()
    component, stack = demo.component, demo.stack
    study = Study(
        component=component,
        stack=stack,
        device=Device(
            p_regions=["p_rib", "p_pad"],
            n_regions=["n_rib", "n_pad"],
            p_doping_cm3=DOPING_CM3,
            n_doping_cm3=DOPING_CM3,
        ),
        plane="x=0",
        output_dir=tmp_path,
    )
    study.charge._result = BiasSweepResult(
        contact="cathode",
        points=[BiasPoint(bias_v=v, carriers=carriers(v)) for v in biases],
    )
    study.charge._has_run = True
    return study


def optical_n_eff(
    study, *, route: str, n_strips: int, wavelength_um: float = 1.55
) -> complex:
    """Solve the optical Staircase on one Route and return its index."""
    study.optical(
        route=route,
        n_strips=n_strips,
        num_modes=1,
        n_guess=2.5,
        wavelength_um=wavelength_um,
    )
    return complex(study.optical.run().n_eff[0])


class TestCrossRouteAgreement:
    # 1.55 um is where the default plasma-dispersion coefficients were
    # fitted; 1.31 um is not, and the Staircase used to read its
    # extinction off the fit rather than off the solve, so the Routes
    # could only be trusted to agree at the first of these.
    @pytest.mark.parametrize("wavelength_um", [1.55, 1.31])
    def test_the_optical_routes_agree_on_the_same_staircase(
        self, tmp_path, wavelength_um
    ):
        """Identical Strips, identical mesh: one effective index, two solvers."""
        study = study_at(tmp_path)
        femwell = optical_n_eff(
            study, route="femwell", n_strips=4, wavelength_um=wavelength_um
        )
        palace = optical_n_eff(
            study, route="palace", n_strips=4, wavelength_um=wavelength_um
        )

        # Both must see the guided silicon mode, not the cladding.
        assert femwell.real > 2.0
        assert palace.real > 2.0
        assert abs(palace.real - femwell.real) < N_EFF_RTOL * abs(femwell.real)
        # The staircase carries free-carrier absorption, so both are lossy
        # in the same direction (exp(+i omega t): Im{n_eff} < 0).
        assert palace.imag < 0.0
        assert femwell.imag < 0.0

    def test_the_route_does_not_change_the_result_type(self, tmp_path):
        study = study_at(tmp_path, biases=(0.0, 2.0))
        study.optical(route="palace", n_strips=3, num_modes=1, n_guess=2.5)
        sweep = study.optical.run()

        assert [point.bias_v for point in sweep.points] == [0.0, 2.0]
        assert sweep.reference_bias_v == 0.0
        assert sweep.index_shift[0] == 0.0
        # Depleting the junction removes free carriers, which raises the index.
        assert sweep.index_shift[1] > 0.0
        # Palace reports no mode fields, so containment is not measurable.
        assert np.isnan(sweep.points[0].boundary_field_ratio)

    def test_the_route_says_it_cannot_check_the_window(self, tmp_path):
        """ADR 0002's containment guard cannot run on Palace, and says so."""
        study = study_at(tmp_path)
        study.optical(route="palace", n_strips=3, num_modes=1, n_guess=2.5)
        with pytest.warns(UserWarning, match="cannot check window containment"):
            study.optical.run()


class TestStripCountConvergence:
    def test_the_palace_index_converges_in_strip_count(self, tmp_path):
        """Refining the Staircase moves the answer less and less.

        Measured against the finest Staircase solved here: each coarser
        strip count must sit further from it than the next one down, which
        is what makes a strip count a knob a user can trade accuracy
        against mesh size with.
        """
        study = study_at(tmp_path, carriers=graded_carriers)
        indices = {
            n: optical_n_eff(study, route="palace", n_strips=n) for n in (1, 2, 4, 8)
        }
        reference = indices[8].real
        errors = [abs(indices[n].real - reference) for n in (1, 2, 4)]
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]
        # And settling, not merely ordered: four strips land several times
        # closer to the reference than one does.
        assert errors[2] < 0.5 * errors[0]
