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
    OpticalPhaseSweep,
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
