"""The Palace Route on the real pipeline, and its agreement with femwell.

The cross-Route gate the modulator API owes the originating spec, on both
EM Stages: both first-class Routes solve the identical Staircase on the
identical mesh under the identical outer-wall condition, and must land on
the same effective index — and, on the RF Stage, the same characteristic
impedance — to solver tolerance. It ships here as a runtime-gated test
rather than as a one-off script, so the agreement is re-checked whenever
either Route moves.

Gated on gmsh, the femwell runtime and a Palace binary; deselected by
default, run with ``pytest -m palace_local``. The Carrier maps are
synthetic, so nothing here needs DEVSIM.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.common.modes import NoLineModeError
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


def rf_line(study, **settings):
    """Solve the RF Stage once and return its line parameters."""
    study.rf(frequencies_hz=[10e9], n_strips=3, num_modes=4, n_guess=2.0, **settings)
    return study.rf.run()


class TestRFElectrodeModel:
    """What the Palace Route can express of the electrode metal (ADR 0003)."""

    def test_a_metal_region_leaves_palace_no_line_mode_to_find(self, tmp_path):
        """A region with ``|Im(eps)| ~ 1e7`` returns its own modes.

        This is why the Palace Route does not default to the model the
        femwell Route does. Every mode of the search comes back losing
        far more than it advances, so none of them is a line mode.
        """
        study = study_at(tmp_path)
        with pytest.raises(NoLineModeError, match="No propagating line mode"):
            rf_line(study, route="palace", conductor_model="volume")


# The Marks-Williams integral is exact to 0.4% on the analytic PEC coax,
# but here it is run on two different discretizations of the fields — an
# order-2 Nedelec solve read back off ParaView nodes against an order-2
# Lagrange curl — so the impedance gate is looser than the index one.
Z0_RTOL = 0.05


@pytest.fixture(scope="module")
def rf_gate(tmp_path_factory):
    """One Staircase, one mesh spec, both Routes, one line Mode each.

    Both Routes express the identical Cross-section: perfect-conductor
    electrodes (ADR 0003) inside a metallic Window, femwell applying its
    boundary condition and Palace putting the same wall under
    ``Boundaries.PEC``. One solve of each Route is shared across the
    gate's assertions, because Palace takes tens of seconds per
    frequency.
    """
    study = study_at(tmp_path_factory.mktemp("rf_gate"))
    femwell = rf_line(study, route="femwell", conductor_model="pec", order=2)
    palace = rf_line(study, route="palace")
    return femwell, palace


class TestRFCrossRouteAgreement:
    """The RF Stage's cross-Route numeric gate (ticket 17)."""

    def test_both_routes_land_on_the_same_line_mode(self, rf_gate):
        femwell, palace = rf_gate
        n_femwell = float(femwell.n_rf[0])
        n_palace = float(palace.n_rf[0])
        # Both must see the line mode of the loaded staircase, not a
        # cladding or box resonance.
        assert n_femwell > 1.5
        assert n_palace > 1.5
        assert abs(n_palace - n_femwell) < N_EFF_RTOL * n_femwell

    def test_both_routes_land_on_the_same_impedance(self, rf_gate):
        femwell, palace = rf_gate
        z_femwell = complex(femwell.z0_ohm[0])
        z_palace = complex(palace.z0_ohm[0])
        assert np.isfinite(z_palace.real)
        assert abs(z_palace.real - z_femwell.real) < Z0_RTOL * abs(z_femwell.real)


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
