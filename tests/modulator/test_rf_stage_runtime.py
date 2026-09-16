"""The RF Stage on the real pipeline: its own Staircase mesh, real solves.

Gated on gmsh and the femwell runtime, and standing on a synthetic
Carrier map so it needs no DEVSIM: what it proves is the Stage's own
chain — pick the Bias point, staircase its Carrier map, mesh the
electrode-loaded Cross-section, select the line Mode at every frequency
and extract the impedance over the signal conductor. The end-to-end run
off a real charge solve is the ``tcad_local`` test at the bottom.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from gsim.common.modes import NoLineModeError
from gsim.common.twmzm_report import RFLineParams
from gsim.modulator import Device, Study

from .conftest import CENTER_Y, HALF_WIDTH, PAD_WIDTH, RIB_HEIGHT, build_demo

pytest.importorskip("gmsh")
pytest.importorskip("femwell")
pytest.importorskip("skfem")

from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap

DOPING_CM3 = 1e18
DEPLETED_CM3 = 1e10
SLAB = (CENTER_Y - HALF_WIDTH - PAD_WIDTH, CENTER_Y + HALF_WIDTH + PAD_WIDTH)
FREQS_HZ = [10e9, 40e9]


def depletion_carriers(bias_v: float) -> CarrierMap:
    """A Carrier map whose depletion region widens with reverse bias."""
    y = np.linspace(SLAB[0], SLAB[1], 121)
    z = np.linspace(0.0, RIB_HEIGHT, 9)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    yy, zz = yy.ravel(), zz.ravel()

    depleted = np.abs(yy - CENTER_Y) < 0.05 * np.sqrt(1.0 + abs(bias_v))
    n_side = yy < CENTER_Y
    electrons = np.where(n_side & ~depleted, DOPING_CM3, DEPLETED_CM3)
    holes = np.where(~n_side & ~depleted, DOPING_CM3, DEPLETED_CM3)

    return CarrierMap(
        x_um=yy,
        y_um=zz,
        region=["n_rib" if side else "p_rib" for side in n_side],
        electrons_cm3=electrons,
        holes_cm3=holes,
        potential_v=np.zeros(yy.size),
        net_doping_cm3=np.zeros(yy.size),
    )


#: Frequency the synthetic small-signal admittances are fitted at (Hz).
JUNCTION_FREQ_HZ = 1e9


def junction_admittance_per_cm(bias_v: float) -> complex:
    """A series-RC admittance whose capacitance falls with reverse bias."""
    r_s_ohm_m = 1.1e-4
    c_j_f_per_m = 3.0e-10 / np.sqrt(1.0 + abs(bias_v))
    omega = 2.0 * np.pi * JUNCTION_FREQ_HZ
    return 1.0 / (r_s_ohm_m - 1j / (omega * c_j_f_per_m)) / 1e2


def canned_sweep(biases) -> BiasSweepResult:
    """A Bias sweep of synthetic Carrier maps and series-RC admittances."""
    return BiasSweepResult(
        contact="cathode",
        points=[
            BiasPoint(
                bias_v=bias,
                carriers=depletion_carriers(bias),
                admittance_s_per_cm=junction_admittance_per_cm(bias),
                admittance_freq_hz=JUNCTION_FREQ_HZ,
            )
            for bias in biases
        ],
    )


def build_study(output_dir, biases=(0.0, 2.0)):
    """A Study whose charge Stage already holds a synthetic Bias sweep."""
    demo = build_demo()
    component, stack = demo.component, demo.stack
    study = Study(
        component=component,
        stack=stack,
        device=Device(p_regions=["p_rib", "p_pad"], n_regions=["n_rib", "n_pad"]),
        output_dir=output_dir,
    )
    study.charge._result = canned_sweep(list(biases))
    study.charge._has_run = True
    return study


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    """The RF Stage run across two frequencies."""
    study = build_study(tmp_path_factory.mktemp("modulator-rf"))
    study.rf(frequencies_hz=FREQS_HZ, n_strips=3)
    return study, study.rf.run()


class TestLineParameters:
    def test_the_result_is_the_line_parameter_type(self, solved):
        _, line = solved

        assert isinstance(line, RFLineParams)
        assert line.freq_hz == pytest.approx(FREQS_HZ)

    def test_the_loaded_line_is_slow_and_lossy(self, solved):
        _, line = solved

        # The carrier-loaded junction slows the wave well past the light
        # line, and the conductive strips make it lossy.
        assert np.all(line.n_rf > 1.0)
        assert np.all(line.alpha_rf_np_m > 0.0)

    def test_the_impedance_is_physical(self, solved):
        _, line = solved

        assert np.all(line.z0_ohm.real > 0.0)
        assert np.all(line.z0_ohm.real < 1e3)

    def test_the_line_parameters_reach_the_rlgc_relations(self, solved):
        _, line = solved

        rlgc = line.rlgc
        assert np.all(np.isfinite(rlgc["C"]))
        assert np.all(rlgc["C"] > 0.0)

    def test_the_bias_it_was_solved_at_is_reported(self, solved):
        study, _ = solved

        assert study.rf.solved_bias_v == 2.0

    def test_the_carriers_stage_ran_first(self, solved):
        study, _ = solved
        assert study.carriers.has_run is True

    def test_running_twice_solves_once(self, solved):
        study, line = solved
        assert study.rf.run() is line


class TestItsOwnStaircase:
    def test_the_rf_mesh_carries_the_strips_and_the_electrodes(self, solved):
        import meshio

        study, _ = solved
        mesh = meshio.read(str(study.stage_dir("rf") / "palace.msh"))
        regions = {
            str(name)
            for name, data in mesh.field_data.items()
            if int(np.asarray(data)[1]) == 2
        }

        assert {"strip_0", "strip_1", "strip_2"} <= regions
        assert {"electrode_low", "electrode_high"} <= regions

    def test_the_rf_mesh_is_not_the_optical_one(self, solved):
        """ADR 0002: the RF solve meshes its own Window."""
        study, _ = solved

        assert (study.stage_dir("rf") / "palace.msh").exists()
        assert not (study.stage_dir("optical") / "palace.msh").exists()

    def test_the_window_spans_the_electrodes_and_their_surroundings(self, solved):
        import meshio

        study, _ = solved
        points = np.asarray(
            meshio.read(str(study.stage_dir("rf") / "palace.msh")).points
        )
        low, high = study.rf.staircase().electrode_spans

        assert points[:, 0].min() < low[0]
        assert points[:, 0].max() > high[1]


class TestSignalConductor:
    def test_the_impedance_integrates_over_the_signal_electrode(
        self, tmp_path, monkeypatch
    ):
        """The elements come from the device description, not the caller."""
        import meshio

        from gsim.femwell import adapter

        captured: dict[str, np.ndarray] = {}
        extract = adapter.z0_power_current

        def spy(mode, *, frequency_hz, current_elements=None, **kwargs):
            captured["elements"] = np.asarray(current_elements)
            return extract(
                mode,
                frequency_hz=frequency_hz,
                current_elements=current_elements,
                **kwargs,
            )

        monkeypatch.setattr(adapter, "z0_power_current", spy)
        study = build_study(tmp_path, biases=[0.0])
        study.rf(frequencies_hz=[10e9], n_strips=2)
        study.rf.run()

        mesh = meshio.read(str(study.stage_dir("rf") / "palace.msh"))
        signal = adapter.region_elements(mesh, "electrode_low")

        assert signal.size > 0
        assert captured["elements"].tolist() == signal.tolist()


@pytest.fixture(scope="module")
def pec_solved(tmp_path_factory):
    """The same RF Stage run with perfect electrodes instead of lossy ones."""
    study = build_study(tmp_path_factory.mktemp("modulator-rf-pec"))
    study.rf(
        frequencies_hz=FREQS_HZ,
        n_strips=3,
        conductor_model="pec",
        # The contour integral reads h on the domain boundary, where
        # femwell's first-order h is piecewise constant.
        order=2,
    )
    return study, study.rf.run()


class TestPerfectConductorElectrodes:
    """The ``"pec"`` conductor model on the femwell Route (ADR 0003).

    A perfect electrode is left out of the meshed domain, so the Stage
    has no conduction current to integrate and reads Ampere's contour
    integral around the hole instead. What that must not do is change
    the answer: the same line, modelled with lossless metal instead of
    lossy metal, has to come out with much the same impedance and much
    less loss.
    """

    def test_the_electrodes_are_not_regions_of_the_mesh(self, pec_solved):
        import meshio

        study, _ = pec_solved
        mesh = meshio.read(str(study.stage_dir("rf") / "palace.msh"))
        regions = {
            str(name)
            for name, data in mesh.field_data.items()
            if int(np.asarray(data)[1]) == 2
        }

        assert {"strip_0", "strip_1", "strip_2"} <= regions
        assert not {"electrode_low", "electrode_high"} & regions

    def test_the_contour_current_reaches_a_physical_impedance(self, pec_solved):
        _, line = pec_solved

        assert np.all(np.isfinite(line.z0_ohm))
        assert np.all(line.z0_ohm.real > 10.0)
        assert np.all(line.z0_ohm.real < 1e3)

    def test_it_lands_where_the_lossy_metal_model_does(self, pec_solved, solved):
        """Two models of the same electrode, one line: same impedance."""
        _, pec = pec_solved
        _, volume = solved

        assert pec.z0_ohm[0].real == pytest.approx(volume.z0_ohm[0].real, rel=0.1)

    def test_the_line_is_far_less_lossy_without_the_metal(self, pec_solved, solved):
        """The metal's own loss is what the pec model drops."""
        _, pec = pec_solved
        _, volume = solved

        assert np.all(pec.alpha_rf_np_m > 0.0)
        assert np.all(pec.alpha_rf_np_m < 0.01 * volume.alpha_rf_np_m)

    def test_a_first_order_solve_says_the_impedance_is_biased(self, tmp_path):
        study = build_study(tmp_path, biases=[0.0])
        study.rf(frequencies_hz=[10e9], n_strips=2, conductor_model="pec", order=1)
        with pytest.warns(UserWarning, match="biased high by tens of percent"):
            study.rf.run()


class TestWindowTooSmall:
    def test_the_default_window_does_not_warn(self, tmp_path):
        """The full extent shields the line rather than squeezing it."""
        import warnings

        study = build_study(tmp_path, biases=[0.0])
        study.rf(frequencies_hz=[10e9], n_strips=2)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            study.rf.run()

        assert not [w for w in caught if "window boundary" in str(w.message)]

    def test_a_squeezed_mode_warns_naming_the_stage(self, tmp_path):
        study = build_study(tmp_path, biases=[0.0])
        # A window barely clearing the electrodes leaves the line mode
        # pressed against the boundary (ADR 0002).
        study.rf(
            frequencies_hz=[10e9],
            n_strips=2,
            window=(CENTER_Y - 4.0, CENTER_Y + 4.0),
            window_z=(-2.0, 2.0),
        )

        with pytest.warns(UserWarning, match="rf stage"):
            study.rf.run()


class TestModeSelection:
    def test_an_override_rule_selects_the_mode(self, tmp_path):
        study = build_study(tmp_path, biases=[0.0])
        study.rf(frequencies_hz=[10e9], n_strips=2, num_modes=2, rule=lambda modes: [])

        with pytest.raises(NoLineModeError):
            study.rf.run()

    def test_ambiguous_candidates_warn_through_the_stage(self, tmp_path):
        """The shared rule's degeneracy warning reaches the user."""
        study = build_study(tmp_path, biases=[0.0])
        study.rf(
            frequencies_hz=[10e9],
            n_strips=2,
            num_modes=2,
            rule=list,
            degeneracy_rtol=1e3,
        )

        with pytest.warns(UserWarning, match="degenerate"):
            study.rf.run()


@pytest.mark.tcad_local
class TestEndToEnd:
    def test_a_real_charge_solve_reaches_the_line_parameters(self, tmp_path):
        pytest.importorskip("devsim")
        demo = build_demo()
        component, stack = demo.component, demo.stack
        study = Study(
            component=component,
            stack=stack,
            device=Device(p_regions=["p_rib", "p_pad"], n_regions=["n_rib", "n_pad"]),
            output_dir=tmp_path,
        )
        study.charge(biases=[0.0, 2.0])
        study.rf(frequencies_hz=FREQS_HZ, n_strips=3)

        line = study.rf.run()

        assert study.charge.has_run is True
        assert line.freq_hz == pytest.approx(FREQS_HZ)
        assert np.all(line.n_rf > 1.0)
        assert study.rf.solved_bias_v == 2.0


class TestTwoPortExport:
    """The Study hands the Traveling-wave electrode over, no hand-carried arrays."""

    def test_the_touchstone_export_reads_back_as_the_solved_line(self, solved):
        skrf = pytest.importorskip("skrf")
        study, line = solved

        path = study.line.export_touchstone()
        network = skrf.Network(str(path))

        assert path == study.stage_dir("line") / "electrode.s2p"
        np.testing.assert_allclose(network.f, line.freq_hz)
        # A passive line referenced to a real impedance transmits at most
        # what it is fed.
        assert np.all(np.isfinite(network.s))
        assert np.all(np.abs(network.s[:, 1, 0]) <= 1.0 + 1e-9)

    def test_the_sax_model_is_evaluated_off_the_study(self, solved):
        study, _ = solved

        sdict = study.line.sax_model()(f=np.linspace(10e9, 40e9, 7))

        s21 = sdict[("o2", "o1")]
        assert s21.shape == (7,)
        assert np.all(np.isfinite(s21))
        assert np.all(np.abs(s21) <= 1.0 + 1e-9)

    def test_the_exports_round_trip_on_the_real_solve(self, solved):
        """Ticket: the handoff reassembles to the Study's own answers."""
        study, _ = solved

        comparison = study.line.verify_exports(quiet=True)

        assert comparison.check() is comparison
        assert comparison.freq_hz == pytest.approx(FREQS_HZ)


@pytest.fixture(scope="module")
def unloaded(tmp_path_factory):
    """The unloaded solve on the same Study, carriers switched off."""
    study = build_study(tmp_path_factory.mktemp("modulator-rf-unloaded"))
    study.rf(frequencies_hz=FREQS_HZ, n_strips=3)
    return study, study.rf.run_unloaded()


class TestUnloadedSolve:
    def test_the_result_is_flagged_unloaded(self, unloaded):
        _, line = unloaded

        assert isinstance(line, RFLineParams)
        assert line.unloaded is True
        assert line.freq_hz == pytest.approx(FREQS_HZ)

    def test_the_bare_line_sits_between_oxide_and_the_strip_dielectric(self, unloaded):
        study, line = unloaded

        # No carriers: the mode's index sits between the light lines of
        # the materials it spreads over — above the oxide's sqrt(3.9),
        # below the strips' sqrt(11.7) — nowhere near the slow-wave
        # index a loaded junction produces. Measured ~2.2 on the demo
        # device.
        assert np.all(line.n_rf > 1.9)
        assert np.all(line.n_rf < np.sqrt(study.rf.strip_permittivity))

    def test_the_impedance_is_a_plausible_bare_lines(self, unloaded):
        _, line = unloaded

        # The demo device measures ~180 ohm: the electrodes are two
        # narrow (2 um) conductors on a thin stack, a high-impedance
        # pair rather than a 50-ohm CPW. The bracket rules out a
        # metal-shorted (~0) or non-line (>> 250 ohm) answer.
        assert np.all(line.z0_ohm.real > 30.0)
        assert np.all(line.z0_ohm.real < 250.0)

    def test_the_unloaded_solve_is_cached_and_invalidated(self, unloaded):
        study, line = unloaded

        assert study.rf.run_unloaded() is line
        study.rf(n_strips=4)
        assert study.rf._unloaded_result is None

    def test_the_loaded_result_is_untouched(self, unloaded):
        study, _ = unloaded

        assert study.rf.has_run is False


@pytest.mark.tcad_local
class TestCrosscheck:
    """Ticket: the two loaded-line routes agree on the demo device.

    The comparison needs the direct solve to actually resolve the
    junction, which the defaults do not attempt: the Strips must tile
    the whole doped slab (so the pads' series resistance is in), be
    narrower than the depletion region (61 across 1.2 um), and the mode
    selection must admit the slow-wave Mode, whose loss ratio sits above
    the line-tuned default bound. 20-30 GHz is the band where both
    solves stay on the quasi-TEM branch: lower, the RC slow wave loses
    about as much as it advances and the selection rightly refuses it;
    higher, the unloaded solve wanders onto a substrate branch.
    """

    def test_the_routes_agree_within_the_gates_tolerance(self, tmp_path):
        pytest.importorskip("devsim")
        demo = build_demo()
        component, stack = demo.component, demo.stack
        study = Study(
            component=component,
            stack=stack,
            device=Device(p_regions=["p_rib", "p_pad"], n_regions=["n_rib", "n_pad"]),
            output_dir=tmp_path,
        )
        study.charge(biases=[0.0, 2.0])
        study.rf(
            frequencies_hz=[20e9, 30e9],
            n_strips=61,
            strip_span=SLAB,
            num_modes=8,
            n_guess=6.0,
            max_loss_ratio=0.6,
        )

        comparison = study.rf.crosscheck()

        assert comparison.direct.unloaded is False
        assert comparison.assembled.unloaded is False
        assert comparison.freq_hz == pytest.approx([20e9, 30e9])
        # The junction loads the line: the direct solve is slower and
        # lossier than the bare electrode by far more than the routes'
        # residual disagreement.
        unloaded = study.rf.run_unloaded()
        assert np.all(comparison.direct.n_rf > 1.5 * unloaded.n_rf)
        # The gate: both routes' n_RF, loss and Z0 within the stated
        # tolerances, or check() names the diverging quantity.
        comparison.check()


class TestWallMode:
    """The femwell Route says when it selected the wall Mode (ticket 23).

    At 0 V the undepleted 1e18 Strips make the loaded line an RC slow
    wave losing as fast as it advances, the default loss bound drops it,
    and what is left inside the bound is the Mode running between both
    electrodes together and the metallic Window wall. Depleting the
    Junction puts the line Mode back inside the bound, where the default
    rule finds it.
    """

    def test_an_undepleted_bias_lands_on_the_wall_mode_and_says_so(self, tmp_path):
        study = build_study(tmp_path, biases=(0.0, 4.0))
        study.rf(
            frequencies_hz=[10e9],
            n_strips=3,
            conductor_model="pec",
            order=2,
            bias_v=0.0,
            # Aimed straight at the wall Mode's index; the Stage's own
            # default of 3.0 lands there too, one candidate later.
            n_guess=2.0,
        )
        with pytest.warns(UserWarning, match="window wall"):
            study.rf.run()

    def test_a_depleted_bias_lands_on_the_line_mode(self, tmp_path):
        study = build_study(tmp_path, biases=(0.0, 4.0))
        study.rf(
            frequencies_hz=[10e9],
            n_strips=3,
            conductor_model="pec",
            order=2,
            bias_v=4.0,
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            line = study.rf.run()
        assert not [w for w in record if "window wall" in str(w.message)]
        # The loaded line Mode, not the wall Mode's ~180 ohm.
        assert 20.0 < line.z0_ohm[0].real < 80.0
        assert line.n_rf[0] > 2.5
