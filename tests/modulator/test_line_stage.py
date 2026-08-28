"""The line Stage: the electrode, the terminations, and the device report.

Nothing here solves anything. Both EM Stages are stubbed with canned
results, so what is under test is the assembly the line Stage does — the
optical sweep and the RF line parameters turned into the existing
:class:`~gsim.common.twmzm_report.TWMZMReport` — and the lifecycle around
it: the report runs whatever upstream Stage has not run, and re-configuring
the line throws the report away without touching either solve.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from gsim.common.twmzm_report import (
    OpticalPhaseSweep,
    line_params_from_neff,
    twmzm_figures_of_merit,
)
from gsim.modulator.optical import OpticalMode, OpticalStage, OpticalSweep
from gsim.modulator.rf import RFStage

WAVELENGTH_UM = 1.55
N_GROUP = 3.8
RF_FREQS = [10e9, 40e9, 100e9]
RF_N_EFF = [3.20 - 0.004j, 3.15 - 0.010j, 3.10 - 0.020j]
RF_Z0 = [46.0 + 1.0j, 44.0 + 0.5j, 42.0 + 0.2j]
BIASES = [0.0, 1.0, 2.0]
#: Phase index per bias: real part falls as the junction depletes.
OPTICAL_N_EFF = [2.400000 - 1e-5j, 2.399950 - 9e-6j, 2.399870 - 8e-6j]


def optical_sweep(
    biases: list[float] | None = None, n_eff: list[complex] | None = None
) -> OpticalSweep:
    """A canned optical sweep, as the optical Stage would return it."""
    biases = BIASES if biases is None else biases
    n_eff = OPTICAL_N_EFF if n_eff is None else n_eff
    reference = n_eff[0].real
    return OpticalSweep(
        contact="cathode",
        wavelength_um=WAVELENGTH_UM,
        reference_bias_v=biases[0],
        points=[
            OpticalMode(
                bias_v=bias,
                n_eff=value,
                index_shift=value.real - reference,
                loss_db_cm=abs(value.imag) * 1e3,
                boundary_field_ratio=1e-4,
            )
            for bias, value in zip(biases, n_eff, strict=True)
        ],
    )


def rf_params():
    """Canned RF line parameters, as the RF Stage would return them."""
    return line_params_from_neff(
        np.asarray(RF_FREQS, dtype=np.float64), RF_N_EFF, z0_ohm=RF_Z0
    )


@pytest.fixture
def solved(study, monkeypatch):
    """A Study whose EM Stages answer from canned results, counting solves."""
    solves = {"optical": 0, "rf": 0}

    def solve_optical(_stage):
        solves["optical"] += 1
        return optical_sweep()

    def solve_rf(_stage):
        solves["rf"] += 1
        return rf_params()

    monkeypatch.setattr(OpticalStage, "_solve", solve_optical)
    monkeypatch.setattr(RFStage, "_solve", solve_rf)
    study.solves = solves
    return study


def hand_assembled(
    *,
    length_m: float,
    z_load_ohm: complex = 50.0,
    z_gen_ohm: complex = 50.0,
    n_group: float = N_GROUP,
):
    """The report the notebook assembles by hand from the same inputs."""
    sweep = optical_sweep()
    return twmzm_figures_of_merit(
        rf_params(),
        OpticalPhaseSweep(
            voltages_v=sweep.voltages,
            dn_eff=sweep.index_shift,
            alpha_opt_db_cm=sweep.loss_db_cm,
            wavelength_um=WAVELENGTH_UM,
            n_group=n_group,
        ),
        length_m=length_m,
        z_load_ohm=z_load_ohm,
        z_gen_ohm=z_gen_ohm,
    )


class TestConfiguration:
    def test_defaults_are_readable(self, study):
        assert study.line.length_um == 3000.0
        assert study.line.z_load_ohm == 50.0
        assert study.line.z_gen_ohm == 50.0
        assert study.line.n_group is None
        assert study.line.response_frequencies_hz is None
        assert study.line.has_run is False

    def test_the_section_is_callable(self, study):
        assert study.line(length_um=5000.0, z_load_ohm=75.0) is study.line
        assert study.line.length_um == 5000.0
        assert study.line.z_load_ohm == 75.0

    def test_unknown_setting_is_rejected(self, study):
        with pytest.raises(ValueError, match="nope"):
            study.line(nope=1)

    def test_a_nonpositive_length_is_rejected(self, study):
        with pytest.raises(ValueError):
            study.line(length_um=0.0)

    def test_a_nonpositive_group_index_is_rejected(self, study):
        with pytest.raises(ValueError):
            study.line(n_group=0.0)

    def test_a_response_grid_that_is_empty_is_rejected(self, study):
        with pytest.raises(ValueError):
            study.line(response_frequencies_hz=[])

    def test_a_nonpositive_response_frequency_is_rejected(self, study):
        with pytest.raises(ValueError, match="positive"):
            study.line(response_frequencies_hz=[0.0])

    def test_the_response_grid_is_kept_ascending(self, study):
        study.line(response_frequencies_hz=[40e9, 10e9])
        assert study.line.response_frequencies_hz == [10e9, 40e9]


class TestReport:
    def test_the_report_matches_the_hand_assembled_call(self, solved):
        solved.line(length_um=3000.0, n_group=N_GROUP)

        report = solved.line.run()
        expected = hand_assembled(length_m=3e-3)

        assert report.length_m == pytest.approx(3e-3)
        assert report.freq_hz == pytest.approx(expected.freq_hz)
        assert report.response == pytest.approx(expected.response)
        assert report.bandwidth_3db_hz == pytest.approx(expected.bandwidth_3db_hz)
        assert report.walkoff_bandwidth_hz == pytest.approx(
            expected.walkoff_bandwidth_hz
        )
        assert report.velocity_mismatch == pytest.approx(expected.velocity_mismatch)
        assert report.vpi_l_vcm == pytest.approx(expected.vpi_l_vcm)
        assert report.voltages_v == pytest.approx(expected.voltages_v)
        for name, values in expected.rlgc.items():
            assert report.rlgc[name] == pytest.approx(values)

    def test_the_terminations_reach_the_response(self, solved):
        solved.line(length_um=3000.0, n_group=N_GROUP, z_load_ohm=75.0, z_gen_ohm=25.0)

        report = solved.line.run()
        expected = hand_assembled(length_m=3e-3, z_load_ohm=75.0, z_gen_ohm=25.0)

        assert report.z_load_ohm == 75.0
        assert report.z_gen_ohm == 25.0
        assert report.response == pytest.approx(expected.response)

    def test_the_length_reaches_the_figures_of_merit(self, solved):
        solved.line(length_um=1000.0, n_group=N_GROUP)
        short = solved.line.run()
        solved.line(length_um=6000.0)
        long = solved.line.run()

        # Walk-off scales as 1/L, so six times the electrode is a sixth
        # of the walk-off-limited bandwidth.
        assert short.walkoff_bandwidth_hz == pytest.approx(
            6.0 * long.walkoff_bandwidth_hz
        )

    def test_the_study_exposes_the_report(self, solved):
        solved.line(n_group=N_GROUP)
        assert solved.report() is solved.line.run()

    def test_an_optical_sweep_of_one_bias_point_is_an_actionable_error(
        self, study, monkeypatch
    ):
        monkeypatch.setattr(
            OpticalStage,
            "_solve",
            lambda _stage: optical_sweep(biases=[0.0], n_eff=[OPTICAL_N_EFF[0]]),
        )
        monkeypatch.setattr(RFStage, "_solve", lambda _stage: rf_params())
        study.line(n_group=N_GROUP)

        with pytest.raises(ValueError, match="charge"):
            study.line.run()


class TestBiasOrder:
    def test_an_unordered_bias_sweep_is_differentiated_in_order(
        self, study, monkeypatch
    ):
        """V_pi L is a slope, so the sweep is sorted before differentiating."""
        shuffled = [0.0, 2.0, 1.0]
        monkeypatch.setattr(
            OpticalStage,
            "_solve",
            lambda _stage: optical_sweep(
                biases=shuffled,
                n_eff=[OPTICAL_N_EFF[0], OPTICAL_N_EFF[2], OPTICAL_N_EFF[1]],
            ),
        )
        monkeypatch.setattr(RFStage, "_solve", lambda _stage: rf_params())
        study.line(length_um=3000.0, n_group=N_GROUP)

        report = study.line.run()
        expected = hand_assembled(length_m=3e-3)

        assert report.voltages_v == pytest.approx(expected.voltages_v)
        assert report.vpi_l_vcm == pytest.approx(expected.vpi_l_vcm)

    def test_a_bias_visited_twice_is_an_actionable_error(self, study, monkeypatch):
        monkeypatch.setattr(
            OpticalStage,
            "_solve",
            lambda _stage: optical_sweep(biases=[0.0, 0.0], n_eff=OPTICAL_N_EFF[:2]),
        )
        monkeypatch.setattr(RFStage, "_solve", lambda _stage: rf_params())
        study.line(n_group=N_GROUP)

        with pytest.raises(ValueError, match=r"twice|repeat"):
            study.line.run()


class TestUnmeasuredImpedance:
    def test_a_nan_impedance_is_named_rather_than_reported_as_a_result(
        self, study, monkeypatch
    ):
        """The palace RF route leaves Z0 NaN; the report says so."""
        nan_z0 = [complex(float("nan"), float("nan"))] * len(RF_FREQS)
        monkeypatch.setattr(OpticalStage, "_solve", lambda _stage: optical_sweep())
        monkeypatch.setattr(
            RFStage,
            "_solve",
            lambda _stage: line_params_from_neff(
                np.asarray(RF_FREQS, dtype=np.float64), RF_N_EFF, z0_ohm=nan_z0
            ),
        )
        study.line(n_group=N_GROUP)

        with pytest.warns(UserWarning, match="impedance"):
            report = study.line.run()

        # The figures that do not need Z0 stay usable.
        assert report.walkoff_bandwidth_hz > 0.0
        assert np.all(np.isfinite(report.vpi_l_vcm))


class TestGroupIndex:
    def test_the_configured_group_index_sets_the_velocity_mismatch(self, solved):
        solved.line(n_group=2.5)

        report = solved.line.run()

        assert report.velocity_mismatch == pytest.approx(
            np.asarray(RF_N_EFF, dtype=complex).real - 2.5
        )

    def test_without_one_the_phase_index_stands_in_and_says_so(self, solved):
        with pytest.warns(UserWarning, match="group index"):
            report = solved.line.run()

        assert report.velocity_mismatch == pytest.approx(
            np.asarray(RF_N_EFF, dtype=complex).real - OPTICAL_N_EFF[0].real
        )

    def test_a_configured_group_index_warns_about_nothing(self, solved):
        solved.line(n_group=N_GROUP)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            solved.line.run()


class TestResponseGrid:
    def test_the_solved_frequencies_are_the_default_grid(self, solved):
        solved.line(n_group=N_GROUP)
        assert solved.line.run().freq_hz == pytest.approx(RF_FREQS)

    def test_a_dense_grid_interpolates_the_line_parameters(self, solved):
        grid = np.linspace(10e9, 100e9, 91)
        solved.line(n_group=N_GROUP, response_frequencies_hz=list(grid))

        report = solved.line.run()

        assert report.freq_hz == pytest.approx(grid)
        n_rf = np.asarray(RF_N_EFF, dtype=complex).real
        assert report.velocity_mismatch + N_GROUP == pytest.approx(
            np.interp(grid, RF_FREQS, n_rf)
        )
        assert report.z0_ohm[0] == pytest.approx(RF_Z0[0])

    def test_a_grid_past_the_solved_range_warns_about_the_clamp(self, solved):
        solved.line(n_group=N_GROUP, response_frequencies_hz=[10e9, 200e9])

        with pytest.warns(UserWarning, match="200|solved"):
            report = solved.line.run()

        # numpy clamps rather than extrapolating: the last solved value holds.
        assert report.velocity_mismatch[-1] + N_GROUP == pytest.approx(
            RF_N_EFF[-1].real
        )


class TestLifecycle:
    def test_the_report_runs_the_stages_that_have_not_run(self, solved):
        solved.line(n_group=N_GROUP)

        solved.line.run()

        assert solved.solves == {"optical": 1, "rf": 1}
        assert solved.optical.has_run is True
        assert solved.rf.has_run is True

    def test_asking_twice_solves_once(self, solved):
        solved.line(n_group=N_GROUP)

        first = solved.line.run()

        assert solved.line.run() is first
        assert solved.solves == {"optical": 1, "rf": 1}

    def test_re_configuring_the_line_drops_the_report_and_no_solve(self, solved):
        solved.line(n_group=N_GROUP)
        solved.line.run()

        solved.line(length_um=5000.0)

        assert solved.line.has_run is False
        assert solved.optical.has_run is True
        assert solved.rf.has_run is True

        solved.line.run()
        assert solved.solves == {"optical": 1, "rf": 1}

    def test_re_configuring_a_stage_upstream_drops_the_report(self, solved):
        solved.line(n_group=N_GROUP)
        solved.line.run()

        solved.optical(wavelength_um=1.31)

        assert solved.optical.has_run is False
        assert solved.line.has_run is False
        assert solved.rf.has_run is True

    def test_the_line_is_the_last_stage_of_the_study(self, study):
        assert list(study.stages) == ["charge", "carriers", "optical", "rf", "line"]
        assert study.stages["line"] is study.line
