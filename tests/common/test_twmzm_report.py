"""Hermetic tests: solver outputs wired into TW-MZM figures of merit.

Synthetic mode results drive the wiring; results are checked against the
analytic limits of the ticket-02 assembly functions.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.common.twmzm import SINC_3DB_ARGUMENT, walkoff_bandwidth
from gsim.common.twmzm_report import (
    LoadedLineComparison,
    OpticalPhaseSweep,
    line_params_from_gamma,
    line_params_from_neff,
    twmzm_figures_of_merit,
)

FREQ = np.linspace(1e9, 100e9, 200)
WL_UM = 1.55


def _optical(n_group=3.8, slope=-1e-4):
    voltages = np.linspace(0.0, 4.0, 9)
    return OpticalPhaseSweep(
        voltages_v=voltages,
        dn_eff=slope * voltages,
        wavelength_um=WL_UM,
        n_group=n_group,
    )


class TestLineParamsExtraction:
    def test_complex_neff_hand_values(self):
        freq = np.array([10e9, 50e9])
        n_eff = np.array([2.0 - 0.1j, 2.5 - 0.2j])
        rf = line_params_from_neff(freq, n_eff, z0_ohm=40.0)
        np.testing.assert_allclose(rf.n_rf, [2.0, 2.5])
        np.testing.assert_allclose(
            rf.alpha_rf_np_m, 2.0 * np.pi * freq * [0.1, 0.2] / C0
        )
        np.testing.assert_allclose(rf.z0_ohm, 40.0)

    def test_either_imag_sign_gives_loss(self):
        freq = np.array([10e9])
        plus = line_params_from_neff(freq, [2.0 + 0.1j], z0_ohm=50.0)
        minus = line_params_from_neff(freq, [2.0 - 0.1j], z0_ohm=50.0)
        np.testing.assert_allclose(plus.alpha_rf_np_m, minus.alpha_rf_np_m)
        assert plus.alpha_rf_np_m[0] > 0

    def test_gamma_round_trips_through_rlgc(self):
        # Lossless 50-ohm line: R = G = 0, L/C give back n_rf and Z0.
        rf = line_params_from_neff(FREQ, 2.5 + 0j, z0_ohm=50.0)
        rlgc = rf.rlgc
        np.testing.assert_allclose(rlgc["R"], 0.0, atol=1e-9)
        np.testing.assert_allclose(rlgc["G"], 0.0, atol=1e-12)
        np.testing.assert_allclose(np.sqrt(rlgc["L"] / rlgc["C"]), 50.0, rtol=1e-9)
        np.testing.assert_allclose(C0 * np.sqrt(rlgc["L"] * rlgc["C"]), 2.5, rtol=1e-9)

    def test_scalar_broadcast(self):
        rf = line_params_from_neff(FREQ, 2.0 - 0.05j, z0_ohm=45.0)
        assert rf.n_rf.shape == FREQ.shape
        assert rf.z0_ohm.shape == FREQ.shape


class TestFiguresOfMerit:
    def test_matched_lossless_velocity_matched_is_flat(self):
        optical = _optical(n_group=2.5)
        rf = line_params_from_neff(FREQ, 2.5 + 0j, z0_ohm=50.0)
        report = twmzm_figures_of_merit(
            rf, optical, length_m=5e-3, z_load_ohm=50.0, z_gen_ohm=50.0
        )
        np.testing.assert_allclose(np.abs(report.response), 1.0, atol=1e-9)
        assert report.bandwidth_3db_hz is None
        assert report.walkoff_bandwidth_hz is None
        np.testing.assert_allclose(report.velocity_mismatch, 0.0)

    def test_velocity_mismatch_reproduces_walkoff_limit(self):
        n_rf, n_opt, length = 6.0, 3.8, 10e-3
        optical = _optical(n_group=n_opt)
        rf = line_params_from_neff(FREQ, n_rf + 0j, z0_ohm=50.0)
        report = twmzm_figures_of_merit(rf, optical, length_m=length)
        expected = walkoff_bandwidth(length_m=length, n_rf=n_rf, n_opt=n_opt)
        assert report.walkoff_bandwidth_hz == pytest.approx(expected)
        # Lossless matched line: the full response's 3 dB point is the
        # analytic sinc walk-off frequency.
        assert report.bandwidth_3db_hz == pytest.approx(expected, rel=1e-2)
        # And the sinc shape itself is reproduced.
        u = np.pi * FREQ * length * abs(n_rf - n_opt) / C0
        np.testing.assert_allclose(
            np.abs(report.response), np.abs(np.sinc(u / np.pi)), atol=1e-6
        )

    def test_vpi_l_matches_closed_form_for_linear_sweep(self):
        slope = -2e-4
        optical = _optical(slope=slope)
        rf = line_params_from_neff(FREQ, 3.8 + 0j, z0_ohm=50.0)
        report = twmzm_figures_of_merit(rf, optical, length_m=5e-3)
        expected_vcm = WL_UM / (2.0 * abs(slope)) / 1e4
        np.testing.assert_allclose(report.vpi_l_vcm, expected_vcm, rtol=1e-9)

    def test_sinc_3db_argument_constant(self):
        u = SINC_3DB_ARGUMENT
        assert abs(np.sin(u) / u) == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-12)

    def test_rejects_nonpositive_length(self):
        rf = line_params_from_neff(FREQ, 2.5 + 0j, z0_ohm=50.0)
        with pytest.raises(ValueError, match="length_m"):
            twmzm_figures_of_merit(rf, _optical(), length_m=0.0)

    def test_report_carries_line_and_bias_axes(self):
        optical = _optical()
        rf = line_params_from_neff(FREQ, 4.0 - 0.02j, z0_ohm=42.0 + 3.0j)
        report = twmzm_figures_of_merit(rf, optical, length_m=3e-3)
        assert report.freq_hz.shape == FREQ.shape
        assert report.vpi_l_vcm.shape == optical.voltages_v.shape
        assert set(report.rlgc) == {"R", "L", "G", "C"}
        assert report.z_load_ohm == 50.0 + 0j

    def test_walkoff_limit_is_not_read_off_the_mean_index(self):
        # n_RF crossing n_g inside the band has a mean mismatch near zero;
        # the limit must follow the index the line actually has.
        freq = np.linspace(10e9, 100e9, 50)
        n_rf = np.linspace(4.2, 3.4, 50)
        rf = line_params_from_neff(freq, n_rf + 0j, z0_ohm=50.0)
        report = twmzm_figures_of_merit(rf, _optical(n_group=3.8), length_m=3e-3)
        assert report.walkoff_bandwidth_hz == pytest.approx(
            walkoff_bandwidth(length_m=3e-3, n_rf=3.4, n_opt=3.8)
        )


class TestMachZehnder:
    """The Phase shifter put in the arms of a Mach-Zehnder, in the report."""

    RF = line_params_from_neff(FREQ, 3.8 + 0j, z0_ohm=50.0)
    SLOPE = -2e-4
    LENGTH_M = 5e-3
    #: lambda / (2 |slope| L) for the linear sweep of ``_optical``: 0.775 V.
    V_PI = WL_UM * 1e-6 / (2.0 * abs(SLOPE) * LENGTH_M)

    def _report(self, optical=None, **settings):
        optical = _optical(slope=self.SLOPE) if optical is None else optical
        return twmzm_figures_of_merit(
            self.RF, optical, length_m=self.LENGTH_M, **settings
        )

    def test_push_pull_at_quadrature_is_the_default(self):
        report = self._report()
        assert report.drive == "push-pull"
        # The arms rest on the middle of the 0-4 V sweep, and the voltage
        # between them can then reach +/- 4 V.
        assert report.arm_bias_v == pytest.approx(2.0)
        assert report.drive_v[[0, -1]] == pytest.approx([-4.0, 4.0])
        assert report.transfer.shape == report.drive_v.shape
        assert np.interp(0.0, report.drive_v, report.transfer) == pytest.approx(0.5)

    def test_v_pi_is_the_modulation_efficiency_over_the_length(self):
        report = self._report()
        assert report.v_pi_v == pytest.approx(self.V_PI)
        assert report.v_pi_v == pytest.approx(
            report.vpi_l_vcm[0] / (self.LENGTH_M * 1e2)
        )
        assert report.transfer_message is None

    def test_a_lossless_balanced_modulator_has_no_loss_and_full_extinction(self):
        report = self._report()
        assert report.insertion_loss_db == pytest.approx(0.0, abs=1e-12)
        assert report.extinction_ratio_db == np.inf
        assert report.transfer.max() == pytest.approx(1.0, abs=1e-4)
        assert report.transfer.min() == pytest.approx(0.0, abs=1e-4)

    def test_the_loss_sweep_reaches_the_insertion_loss(self):
        optical = _optical(slope=self.SLOPE)
        optical.alpha_opt_db_cm = np.full_like(optical.voltages_v, 8.0)
        report = self._report(optical)
        # 8 dB/cm over 5 mm, in both arms alike.
        assert report.insertion_loss_db == pytest.approx(4.0)
        assert report.extinction_ratio_db == np.inf

    def test_the_arm_imbalance_reaches_the_extinction_ratio(self):
        report = self._report(arm_imbalance_db=0.5)
        q = 10.0 ** (-0.5 / 20.0)
        assert report.extinction_ratio_db == pytest.approx(
            20.0 * np.log10((1.0 + q) / (1.0 - q))
        )

    def test_the_phase_offset_moves_the_rest_point(self):
        report = self._report(phase_offset_rad=0.0)
        assert np.interp(0.0, report.drive_v, report.transfer) == pytest.approx(1.0)

    def test_the_chirp_follows_the_drive_configuration(self):
        push_pull = self._report()
        single = self._report(drive="single-drive")
        assert push_pull.chirp.shape == push_pull.voltages_v.shape
        assert np.all(push_pull.chirp == 0.0)
        assert single.chirp == pytest.approx(1.0)
        assert self._report(
            drive="single-drive", phase_offset_rad=-np.pi / 2.0
        ).chirp == pytest.approx(-1.0)

    def test_a_single_drive_rests_where_it_is_told_to(self):
        report = self._report(drive="single-drive", arm_bias_v=0.0)
        assert report.arm_bias_v == 0.0
        assert report.drive_v[[0, -1]] == pytest.approx([0.0, 4.0])

    def test_a_sweep_too_short_to_reach_v_pi_says_so(self):
        # A tenth of the index shift: V_pi is 7.75 V and the sweep spans 4.
        report = self._report(_optical(slope=self.SLOPE / 10.0), drive="single-drive")
        assert report.v_pi_v is None
        assert report.insertion_loss_db is None
        assert report.extinction_ratio_db is None
        assert "too short to reach V_pi" in report.transfer_message
        # The transfer it did cover is still reported.
        assert np.all(np.isfinite(report.transfer))

    def test_a_sweep_out_of_bias_order_is_read_in_order(self):
        ordered = _optical(slope=self.SLOPE)
        ordered.alpha_opt_db_cm = 8.0 - ordered.voltages_v + 0.1 * ordered.voltages_v**2
        shuffle = np.array([3, 0, 8, 1, 5, 2, 7, 4, 6])
        shuffled = OpticalPhaseSweep(
            voltages_v=ordered.voltages_v[shuffle],
            dn_eff=ordered.dn_eff[shuffle],
            alpha_opt_db_cm=ordered.alpha_opt_db_cm[shuffle],
            wavelength_um=WL_UM,
            n_group=3.8,
        )
        expected = self._report(ordered, drive="single-drive")
        report = self._report(shuffled, drive="single-drive")
        assert report.transfer == pytest.approx(expected.transfer)
        assert report.v_pi_v == pytest.approx(expected.v_pi_v)
        # The chirp is per bias point, so it stays in the sweep's own order.
        assert report.chirp == pytest.approx(expected.chirp[shuffle])


class TestUnloadedFlag:
    def test_line_params_are_loaded_unless_said_otherwise(self):
        rf = line_params_from_neff(FREQ, 2.5 + 0j, z0_ohm=50.0)
        assert rf.unloaded is False

    def test_an_unloaded_solve_is_flagged(self):
        rf = line_params_from_neff(FREQ, 2.5 + 0j, z0_ohm=50.0, unloaded=True)
        assert rf.unloaded is True


class TestLineParamsFromGamma:
    def test_roundtrips_the_gamma_property(self):
        rf = line_params_from_neff(FREQ, 2.5 - 0.05j, z0_ohm=45.0 + 2.0j)

        back = line_params_from_gamma(FREQ, rf.gamma_per_m, z0_ohm=rf.z0_ohm)

        assert back.n_rf == pytest.approx(rf.n_rf)
        assert back.alpha_rf_np_m == pytest.approx(rf.alpha_rf_np_m)
        assert back.z0_ohm == pytest.approx(rf.z0_ohm)
        assert back.unloaded is False

    def test_hand_values(self):
        freq = np.array([10e9])
        omega = 2 * np.pi * freq
        gamma = 30.0 + 1j * omega * 2.5 / C0

        rf = line_params_from_gamma(freq, gamma, z0_ohm=40.0)

        assert rf.n_rf == pytest.approx([2.5])
        assert rf.alpha_rf_np_m == pytest.approx([30.0])

    def test_either_sign_convention_is_loss(self):
        freq = np.array([10e9])
        omega = 2 * np.pi * freq
        gamma = -30.0 - 1j * omega * 2.5 / C0

        rf = line_params_from_gamma(freq, gamma, z0_ohm=40.0)

        assert rf.n_rf == pytest.approx([2.5])
        assert rf.alpha_rf_np_m == pytest.approx([30.0])


def _line(n_rf=2.5, alpha=40.0, z0=45.0, unloaded=False):
    freq = np.array([10e9, 40e9])
    return line_params_from_neff(
        freq,
        n_rf - 1j * alpha * C0 / (2 * np.pi * freq),
        z0_ohm=z0,
        unloaded=unloaded,
    )


class TestLoadedLineComparison:
    def test_identical_routes_have_zero_deltas_and_pass(self):
        cmp = LoadedLineComparison(direct=_line(), assembled=_line())

        assert cmp.delta_n_rf == pytest.approx([0.0, 0.0], abs=1e-15)
        assert cmp.delta_alpha == pytest.approx([0.0, 0.0], abs=1e-15)
        assert cmp.delta_z0 == pytest.approx([0.0, 0.0], abs=1e-15)
        cmp.check()

    def test_the_deltas_are_relative_to_the_direct_route(self):
        cmp = LoadedLineComparison(direct=_line(n_rf=2.0), assembled=_line(n_rf=2.2))

        assert cmp.delta_n_rf == pytest.approx([0.1, 0.1])

    def test_a_diverged_index_fails_naming_quantity_and_frequency(self):
        cmp = LoadedLineComparison(direct=_line(n_rf=2.0), assembled=_line(n_rf=3.0))

        with pytest.raises(ValueError, match=r"n_RF.*10 GHz"):
            cmp.check()

    def test_a_diverged_impedance_fails_naming_the_impedance(self):
        cmp = LoadedLineComparison(direct=_line(z0=40.0), assembled=_line(z0=80.0))

        with pytest.raises(ValueError, match=r"Z0"):
            cmp.check()

    def test_the_tolerances_are_adjustable(self):
        cmp = LoadedLineComparison(direct=_line(n_rf=2.0), assembled=_line(n_rf=2.2))

        cmp.check(rtol_n_rf=0.2)
        with pytest.raises(ValueError, match="n_RF"):
            cmp.check(rtol_n_rf=0.05)

    def test_mismatched_frequency_axes_are_rejected(self):
        long = line_params_from_neff(np.array([1e9, 2e9, 3e9]), 2.5, z0_ohm=45.0)

        with pytest.raises(ValueError, match="freq"):
            LoadedLineComparison(direct=_line(), assembled=long)


class TestProvenance:
    def test_line_params_carry_no_bias_or_contact_unless_given(self):
        line = line_params_from_neff([10e9], [2.0 - 0.01j], z0_ohm=[50.0])
        assert line.bias_v is None
        assert line.signal_contact is None

    def test_the_bias_and_the_contact_travel_with_the_record(self):
        line = line_params_from_neff(
            [10e9], [2.0 - 0.01j], z0_ohm=[50.0], bias_v=2.0, signal_contact="cathode"
        )
        assert line.bias_v == 2.0
        assert line.signal_contact == "cathode"
        from_gamma = line_params_from_gamma(
            line.freq_hz, line.gamma_per_m, z0_ohm=line.z0_ohm, bias_v=2.0
        )
        assert from_gamma.bias_v == 2.0


class TestResampling:
    def _line(self):
        return line_params_from_neff(
            [10e9, 20e9, 40e9],
            [3.2 - 0.004j, 3.1 - 0.010j, 3.0 - 0.020j],
            z0_ohm=[46.0 + 1.0j, 44.0 + 0.5j, 42.0 + 0.2j],
            bias_v=1.5,
            signal_contact="cathode",
        )

    def test_the_solved_points_are_reproduced(self):
        line = self._line()
        same = line.resampled(line.freq_hz)
        np.testing.assert_allclose(same.n_rf, line.n_rf)
        np.testing.assert_allclose(same.alpha_rf_np_m, line.alpha_rf_np_m)
        np.testing.assert_allclose(same.z0_ohm, line.z0_ohm)

    def test_between_points_every_quantity_interpolates_linearly(self):
        line = self._line()
        mid = line.resampled([15e9, 30e9])
        np.testing.assert_allclose(mid.n_rf, [3.15, 3.05])
        np.testing.assert_allclose(
            mid.alpha_rf_np_m,
            np.interp([15e9, 30e9], line.freq_hz, line.alpha_rf_np_m),
        )
        # Real and imaginary parts of the impedance interpolate separately.
        np.testing.assert_allclose(mid.z0_ohm, [45.0 + 0.75j, 43.0 + 0.35j])

    def test_outside_the_solved_range_the_end_values_hold(self):
        line = self._line()
        clamped = line.resampled([1e9, 100e9])
        np.testing.assert_allclose(clamped.n_rf, [3.2, 3.0])
        np.testing.assert_allclose(clamped.z0_ohm, [46.0 + 1.0j, 42.0 + 0.2j])

    def test_the_provenance_and_the_flag_travel_with_it(self):
        line = self._line().model_copy(update={"unloaded": True})
        resampled = line.resampled([12e9])
        assert resampled.bias_v == 1.5
        assert resampled.signal_contact == "cathode"
        assert resampled.unloaded is True

    def test_a_descending_grid_is_refused(self):
        with pytest.raises(ValueError, match="ascending"):
            self._line().resampled([20e9, 10e9])
