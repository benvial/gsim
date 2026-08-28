"""The optical Stage on the real pipeline: its own mesh, and a real solve.

The solve is gated on gmsh and the femwell runtime, and stands on a
synthetic Carrier map so it needs no DEVSIM: what it proves is the Stage's
own chain — derive the Window, mesh it, carry the carriers onto that mesh,
perturb the permittivity and solve. The end-to-end run off a real charge
solve is the ``tcad_local`` test at the bottom.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.modulator import Device, OpticalSweep, Study
from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap

from .conftest import CENTER_Y, HALF_WIDTH, PAD_WIDTH, RIB_HEIGHT, build_demo

pytest.importorskip("gmsh")
pytest.importorskip("femwell")
pytest.importorskip("skfem")

DOPING_CM3 = 1e18
DEPLETED_CM3 = 1e10
SLAB = (CENTER_Y - HALF_WIDTH - PAD_WIDTH, CENTER_Y + HALF_WIDTH + PAD_WIDTH)


def depletion_carriers(bias_v: float) -> CarrierMap:
    """A Carrier map whose depletion region widens with reverse bias.

    Not a charge solve — a monotone stand-in for one, sampled across the
    doped slab so the transfer onto the optical mesh has something to
    interpolate.
    """
    y = np.linspace(SLAB[0], SLAB[1], 121)
    z = np.linspace(0.0, RIB_HEIGHT, 9)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    yy, zz = yy.ravel(), zz.ravel()

    half_width = 0.05 * np.sqrt(1.0 + abs(bias_v))
    depleted = np.abs(yy - CENTER_Y) < half_width
    n_side = yy < CENTER_Y

    electrons = np.where(n_side, DOPING_CM3, DEPLETED_CM3)
    holes = np.where(n_side, DEPLETED_CM3, DOPING_CM3)
    electrons = np.where(depleted, DEPLETED_CM3, electrons)
    holes = np.where(depleted, DEPLETED_CM3, holes)

    return CarrierMap(
        x_um=yy,
        y_um=zz,
        region=["p_rib" if side else "n_rib" for side in ~n_side],
        electrons_cm3=electrons,
        holes_cm3=holes,
        potential_v=np.zeros(yy.size),
        net_doping_cm3=np.zeros(yy.size),
    )


def canned_sweep(biases) -> BiasSweepResult:
    """A Bias sweep of synthetic Carrier maps."""
    return BiasSweepResult(
        contact="cathode",
        points=[
            BiasPoint(bias_v=bias, carriers=depletion_carriers(bias)) for bias in biases
        ],
    )


def build_study(output_dir):
    """A Study over the phase shifter, writing into ``output_dir``."""
    demo = build_demo()
    component, stack = demo.component, demo.stack
    return Study(
        component=component,
        stack=stack,
        device=Device(p_regions=["p_rib", "p_pad"], n_regions=["n_rib", "n_pad"]),
        output_dir=output_dir,
    )


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """The optical Stage run across a synthetic Bias sweep."""
    study = build_study(tmp_path_factory.mktemp("modulator-optical"))
    biases = [0.0, 2.0]
    study.charge._result = canned_sweep(biases)
    study.charge._has_run = True
    return study, study.optical.run()


class TestItsOwnMesh:
    def test_the_optical_mesh_is_not_the_charge_mesh(self, solved):
        """ADR 0002: the optical solve meshes its own Window."""
        study, _ = solved
        charge_sim = study.charge.simulation()
        charge_sim.mesh(**study.charge.mesh)

        optical_mesh = study.stage_dir("optical") / "palace.msh"

        assert optical_mesh.exists()
        assert optical_mesh != charge_sim.mesh_path

    def test_the_optical_mesh_spans_the_derived_window(self, solved):
        import meshio

        study, _ = solved
        points = np.asarray(
            meshio.read(str(study.stage_dir("optical") / "palace.msh")).points
        )
        window = study.layout.window_around_junction(
            margin_um=study.optical.mode_margin_um
        )
        window_z = study.layout.window_z_around_guide(above_um=1.0, below_um=1.0)

        assert points[:, 0].min() == pytest.approx(window[0], abs=0.05)
        assert points[:, 0].max() == pytest.approx(window[1], abs=0.05)
        assert points[:, 1].min() == pytest.approx(window_z[0], abs=0.05)
        assert points[:, 1].max() == pytest.approx(window_z[1], abs=0.05)


class TestSolvedModes:
    def test_the_sweep_reports_one_mode_per_bias_point(self, solved):
        _, sweep = solved

        assert isinstance(sweep, OpticalSweep)
        assert sweep.contact == "cathode"
        assert sweep.wavelength_um == 1.55
        assert sweep.voltages == pytest.approx([0.0, 2.0])

    def test_the_mode_is_guided_by_the_rib(self, solved):
        _, sweep = solved
        # Between the oxide cladding and bulk silicon.
        assert all(1.444 < n.real < 3.48 for n in sweep.n_eff)

    def test_the_index_shift_is_measured_from_zero_bias(self, solved):
        _, sweep = solved

        assert sweep.reference_bias_v == 0.0
        assert sweep.index_shift[0] == 0.0
        # Reverse bias depletes the rib: fewer carriers, less negative
        # plasma-dispersion shift, so the effective index rises.
        assert sweep.index_shift[1] > 0.0

    def test_depleting_the_rib_lowers_the_loss(self, solved):
        _, sweep = solved

        assert all(loss > 0.0 for loss in sweep.loss_db_cm)
        assert sweep.loss_db_cm[1] < sweep.loss_db_cm[0]

    def test_the_mode_is_contained_by_its_window(self, solved):
        _, sweep = solved
        assert all(p.boundary_field_ratio < 0.01 for p in sweep.points)

    def test_the_carriers_stage_ran_first(self, solved):
        study, _ = solved
        assert study.carriers.has_run is True

    def test_running_twice_solves_once(self, solved):
        study, sweep = solved
        assert study.optical.run() is sweep


class TestWindowTooSmall:
    def test_a_clipped_mode_warns_naming_the_stage(self, tmp_path):
        study = build_study(tmp_path)
        study.charge._result = canned_sweep([0.0])
        study.charge._has_run = True
        # A window barely wider than the rib cannot hold the mode's tails.
        study.optical(mode_margin_um=0.45, z_above_um=0.15, z_below_um=0.15)

        with pytest.warns(UserWarning, match="optical stage"):
            sweep = study.optical.run()

        assert sweep.points[0].boundary_field_ratio > 0.01


class TestRegionsOffTheWindow:
    def test_a_pad_clipped_out_of_the_window_is_not_an_error(self, tmp_path):
        """The optical Window is a box around the rib, not the doped slab."""
        study = build_study(tmp_path)
        study.charge._result = canned_sweep([0.0])
        study.charge._has_run = True
        # Tight enough that both contact pads fall outside the mesh.
        study.optical(mode_margin_um=0.25, z_above_um=0.15, z_below_um=0.15)

        with pytest.warns(UserWarning, match="optical stage"):
            sweep = study.optical.run()

        assert sweep.points[0].n_eff.real > 1.444

    def test_a_named_region_off_the_window_is_reported(self, tmp_path):
        study = build_study(tmp_path)
        study.charge._result = canned_sweep([0.0])
        study.charge._has_run = True
        study.optical(
            mode_margin_um=0.25,
            z_above_um=0.15,
            z_below_um=0.15,
            perturbed_regions=["p_rib", "p_pad"],
        )

        with pytest.raises(ValueError, match=r"p_pad"):
            study.optical.run()

    def test_a_window_holding_no_doped_region_is_reported(self, tmp_path):
        study = build_study(tmp_path)
        study.charge._result = canned_sweep([0.0])
        study.charge._has_run = True
        # A box in the cladding, well above the rib.
        study.optical(window=(CENTER_Y - 1.0, CENTER_Y + 1.0), window_z=(1.0, 2.0))

        with pytest.raises(ValueError, match="perturb nothing"):
            study.optical.run()


@pytest.mark.tcad_local
class TestEndToEnd:
    def test_a_real_charge_solve_reaches_the_optical_mode(self, tmp_path):
        pytest.importorskip("devsim")
        study = build_study(tmp_path)
        study.charge(biases=[0.0, 2.0])

        sweep = study.optical.run()

        assert study.charge.has_run is True
        assert len(sweep.points) == 2
        assert all(1.444 < n.real < 3.48 for n in sweep.n_eff)
        assert sweep.index_shift[1] > 0.0


class TestStaircaseWavelength:
    """The Staircase reads its loss at the wavelength being solved at.

    A plasma-dispersion model's wavelength is where its coefficients were
    fitted: it says what ``dalpha_cm`` a carrier concentration means, not
    where anyone is solving. Turning that absorption into an extinction
    coefficient — ``kappa = alpha lambda / 4 pi`` — is the step that needs
    the solve's own wavelength, and the continuous path has always used
    it. Building the Strips at the fit wavelength instead inflated every
    Strip's loss by ``1.55 / 1.31``, about 18%, at 1.31 um.
    """

    @staticmethod
    def _staircase_loss(output_dir, wavelength_um: float) -> float:
        """Modal loss (dB/cm) of the Staircase solved at one wavelength."""
        study = build_study(output_dir)
        study.charge._result = canned_sweep([0.0])
        study.charge._has_run = True
        study.optical(
            route="femwell",
            wavelength_um=wavelength_um,
            n_strips=8,
            num_modes=1,
            n_guess=2.5,
        )
        return float(study.optical.run().points[0].loss_db_cm)

    def test_the_staircase_loss_is_the_carriers_not_the_wavelengths(self, tmp_path):
        """Fixed coefficients, so dB/cm is a property of the carriers.

        The material absorption does not move between 1.31 um and 1.55 um
        here — the same 1.55 um fit answers both — so the modal loss may
        only move by the confinement change, a couple of percent. The
        defect made it move by 17%.
        """
        dispersion_um = 1.55
        at_fit = self._staircase_loss(tmp_path / "at-fit", dispersion_um)
        away = self._staircase_loss(tmp_path / "away", 1.31)

        assert at_fit > 0.0
        assert away == pytest.approx(at_fit, rel=0.05)
