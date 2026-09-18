"""Tests for the traveling-wave MZM assembly layer (gsim.common.twmzm).

Analytic checks use the standard traveling-wave modulator limits:

- perfect velocity match, no loss, matched impedances -> flat response;
- velocity mismatch only -> |sin(u)/u| walk-off roll-off with
  u = pi f L (n_rf - n_opt) / c;
- RF loss only -> |(1 - exp(-alpha L)) / (alpha L)|;
- V_pi L = lambda / (2 |d(dn_eff)/dV|).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.common.twmzm import (
    JunctionBranch,
    eo_bandwidth,
    eo_response,
    loaded_line_params,
    rlgc_from_line_params,
    series_rc_from_admittance,
    vpi_length_vcm,
    walkoff_bandwidth,
    walkoff_bandwidth_dispersive,
)
from tests._helpers import series_rc_admittance


class TestEOResponse:
    def test_matched_lossless_velocity_matched_is_flat(self):
        freq = np.linspace(1e6, 100e9, 50)
        m = eo_response(
            freq,
            length_m=5e-3,
            n_rf=4.0,
            n_opt=4.0,
            alpha_rf_np_m=0.0,
            z0_ohm=50.0,
            z_load_ohm=50.0,
            z_gen_ohm=50.0,
        )
        assert np.allclose(np.abs(m), 1.0, atol=1e-9)

    def test_velocity_mismatch_sinc_rolloff(self):
        length = 10e-3
        n_rf, n_opt = 4.0, 3.6
        freq = np.array([1e9, 20e9, 50e9])
        m = eo_response(
            freq,
            length_m=length,
            n_rf=n_rf,
            n_opt=n_opt,
            alpha_rf_np_m=0.0,
            z0_ohm=50.0,
            z_load_ohm=50.0,
            z_gen_ohm=50.0,
        )
        u = np.pi * freq * length * (n_rf - n_opt) / C0
        expected = np.abs(np.sin(u) / u)
        assert np.abs(m) == pytest.approx(expected, rel=1e-9)

    def test_loss_only_rolloff(self):
        length = 8e-3
        alpha = 200.0  # Np/m
        freq = np.array([10e9])
        m = eo_response(
            freq,
            length_m=length,
            n_rf=3.8,
            n_opt=3.8,
            alpha_rf_np_m=alpha,
            z0_ohm=50.0,
            z_load_ohm=50.0,
            z_gen_ohm=50.0,
            normalize=False,
        )
        # Matched drive halves the generator voltage; loss averages to
        # (1 - exp(-alpha L)) / (alpha L).
        expected = 0.5 * (1.0 - np.exp(-alpha * length)) / (alpha * length)
        assert np.abs(m[0]) == pytest.approx(expected, rel=1e-9)

    def test_normalized_to_unity_at_dc(self):
        freq = np.linspace(1e5, 60e9, 30)
        m = eo_response(
            freq,
            length_m=6e-3,
            n_rf=3.9,
            n_opt=3.6,
            alpha_rf_np_m=150.0,
            z0_ohm=42.0,
            z_load_ohm=50.0,
            z_gen_ohm=50.0,
        )
        # DC-normalized: response tends to 1 at low frequency.
        assert np.abs(m[0]) == pytest.approx(1.0, abs=1e-3)

    def test_frequency_dependent_line_params(self):
        freq = np.linspace(1e8, 50e9, 20)
        n_rf = np.linspace(4.1, 3.9, 20)
        alpha = 30.0 * np.sqrt(freq / 1e9)
        z0 = np.linspace(48.0, 44.0, 20)
        m = eo_response(
            freq,
            length_m=4e-3,
            n_rf=n_rf,
            n_opt=3.7,
            alpha_rf_np_m=alpha,
            z0_ohm=z0,
            z_load_ohm=50.0,
            z_gen_ohm=50.0,
        )
        assert m.shape == freq.shape
        assert np.all(np.isfinite(np.abs(m)))
        assert np.abs(m[-1]) < np.abs(m[0])

    def test_mismatched_load_differs_from_matched(self):
        freq = np.array([25e9])
        kwargs = dict(
            length_m=5e-3,
            n_rf=3.9,
            n_opt=3.6,
            alpha_rf_np_m=100.0,
            z0_ohm=40.0,
            z_gen_ohm=50.0,
        )
        matched = eo_response(freq, z_load_ohm=40.0, **kwargs)
        mismatched = eo_response(freq, z_load_ohm=50.0, **kwargs)
        assert np.abs(matched[0]) != pytest.approx(np.abs(mismatched[0]), rel=1e-6)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            eo_response(
                np.array([1e9]),
                length_m=0.0,
                n_rf=3.9,
                n_opt=3.6,
                alpha_rf_np_m=0.0,
                z0_ohm=50.0,
                z_load_ohm=50.0,
                z_gen_ohm=50.0,
            )


class TestEOBandwidth:
    def test_walkoff_bandwidth_analytic(self):
        # Lossless matched line: |sin u / u| = 1/sqrt(2) at u ~ 1.3916.
        length = 10e-3
        n_rf, n_opt = 4.0, 3.6
        expected = 1.3915573 * C0 / (np.pi * length * (n_rf - n_opt))
        assert walkoff_bandwidth(
            length_m=length, n_rf=n_rf, n_opt=n_opt
        ) == pytest.approx(expected, rel=1e-5)

    def test_bandwidth_from_response_matches_walkoff(self):
        length = 10e-3
        n_rf, n_opt = 4.0, 3.6
        freq = np.linspace(1e6, 40e9, 4000)
        m = eo_response(
            freq,
            length_m=length,
            n_rf=n_rf,
            n_opt=n_opt,
            alpha_rf_np_m=0.0,
            z0_ohm=50.0,
            z_load_ohm=50.0,
            z_gen_ohm=50.0,
        )
        f3db = eo_bandwidth(freq, m)
        assert f3db == pytest.approx(
            walkoff_bandwidth(length_m=length, n_rf=n_rf, n_opt=n_opt), rel=1e-3
        )

    def test_no_crossing_returns_none(self):
        freq = np.linspace(1e6, 10e9, 10)
        m = np.ones(10, dtype=complex)
        assert eo_bandwidth(freq, m) is None


class TestVpiLength:
    def test_linear_dneff(self):
        # dn_eff = s * V with s = 5e-5 / V at 1.55 um:
        # V_pi L = lambda / (2 s) = 1.55 / 1e-4 um V = 1.55 V cm.
        v = np.linspace(0.0, 4.0, 9)
        dneff = 5e-5 * v
        vpil = vpi_length_vcm(v, dneff, wavelength_um=1.55)
        assert vpil == pytest.approx(np.full(9, 1.55), rel=1e-9)

    def test_sublinear_dneff_increases_with_bias(self):
        # Depletion-type saturation: slope falls, V_pi L grows with bias.
        v = np.linspace(0.0, 4.0, 41)
        dneff = 1e-4 * np.sqrt(v + 0.5)
        vpil = vpi_length_vcm(v, dneff, wavelength_um=1.55)
        assert np.all(np.diff(vpil) > 0)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            vpi_length_vcm(np.array([0.0, 1.0]), np.array([0.0]), wavelength_um=1.55)


class TestRLGC:
    def test_lossless_line_roundtrip(self):
        l_per_m = 2.5e-7
        c_per_m = 1e-10
        z0 = np.sqrt(l_per_m / c_per_m)  # 50 ohm
        freq = np.array([1e9, 10e9])
        omega = 2 * np.pi * freq
        gamma = 1j * omega * np.sqrt(l_per_m * c_per_m)
        rlgc = rlgc_from_line_params(freq, gamma_per_m=gamma, z0_ohm=z0)
        assert rlgc["R"] == pytest.approx(np.zeros(2), abs=1e-9)
        assert rlgc["L"] == pytest.approx(np.full(2, l_per_m), rel=1e-12)
        assert rlgc["G"] == pytest.approx(np.zeros(2), abs=1e-12)
        assert rlgc["C"] == pytest.approx(np.full(2, c_per_m), rel=1e-12)

    def test_lossy_line_has_positive_r(self):
        freq = np.array([5e9])
        omega = 2 * np.pi * freq
        gamma = 40.0 + 1j * omega * 4.0 / C0  # alpha = 40 Np/m, n_rf = 4
        rlgc = rlgc_from_line_params(freq, gamma_per_m=gamma, z0_ohm=45.0 + 2.0j)
        assert rlgc["R"][0] > 0
        assert rlgc["C"][0] > 0


class TestSeriesRCFromAdmittance:
    def test_recovers_the_branch_it_came_from(self):
        r_s, c_j = 8e-4, 2.4e-10  # 0.8 ohm mm, 0.24 fF/um
        y = series_rc_admittance(1e6, r_s, c_j)
        r_fit, c_fit = series_rc_from_admittance(y, freq_hz=1e6)
        assert r_fit == pytest.approx(r_s, rel=1e-12)
        assert c_fit == pytest.approx(c_j, rel=1e-12)

    def test_pure_capacitor_has_zero_resistance(self):
        c_j = 1e-10
        omega = 2 * np.pi * 1e6
        r_fit, c_fit = series_rc_from_admittance(1j * omega * c_j, freq_hz=1e6)
        assert r_fit == pytest.approx(0.0, abs=1e-15)
        assert c_fit == pytest.approx(c_j, rel=1e-12)

    def test_a_sweep_of_admittances_fits_pointwise(self):
        r_s = np.array([1e-3, 2e-3, 3e-3])
        c_j = np.array([3e-10, 2e-10, 1e-10])
        y = series_rc_admittance(1e6, r_s, c_j)
        r_fit, c_fit = series_rc_from_admittance(y, freq_hz=1e6)
        assert r_fit == pytest.approx(r_s, rel=1e-12)
        assert c_fit == pytest.approx(c_j, rel=1e-12)

    def test_rejects_an_inductive_admittance(self):
        with pytest.raises(ValueError, match="capacitive"):
            series_rc_from_admittance(1e-3 - 1e-4j, freq_hz=1e6)

    def test_rejects_a_negative_conductance(self):
        y = series_rc_admittance(1e6, 1e-3, 2e-10)
        with pytest.raises(ValueError, match="series RC"):
            series_rc_from_admittance(-y.real + 1j * y.imag, freq_hz=1e6)

    def test_rejects_a_nonpositive_frequency(self):
        with pytest.raises(ValueError, match="frequency"):
            series_rc_from_admittance(1e-3 + 1e-4j, freq_hz=0.0)


class TestLoadedLineParams:
    UNLOADED: ClassVar = {
        "R": np.zeros(2),
        "L": np.full(2, 4e-7),
        "G": np.zeros(2),
        "C": np.full(2, 8e-11),
    }
    FREQ = np.array([1e9, 10e9])

    def test_zero_junction_branch_recovers_the_unloaded_line(self):
        gamma, z0 = loaded_line_params(
            self.FREQ, rlgc=self.UNLOADED, junction=(0.0, 0.0)
        )
        omega = 2 * np.pi * self.FREQ
        assert gamma == pytest.approx(1j * omega * np.sqrt(4e-7 * 8e-11), rel=1e-12)
        assert z0 == pytest.approx(np.full(2, np.sqrt(4e-7 / 8e-11)), rel=1e-12)

    def test_a_lossless_junction_adds_its_capacitance(self):
        c_j = 2e-10
        gamma, z0 = loaded_line_params(
            self.FREQ, rlgc=self.UNLOADED, junction=(0.0, c_j)
        )
        omega = 2 * np.pi * self.FREQ
        c_total = 8e-11 + c_j
        assert gamma == pytest.approx(1j * omega * np.sqrt(4e-7 * c_total), rel=1e-12)
        assert z0 == pytest.approx(np.sqrt(4e-7 / c_total) * np.ones(2), rel=1e-12)

    def test_the_series_resistance_makes_the_line_lossy(self):
        gamma, z0 = loaded_line_params(
            self.FREQ, rlgc=self.UNLOADED, junction=JunctionBranch(2e-3, 2e-10)
        )
        # Hand-computed: Z = jwL', Y = jwC' + jwC_j / (1 + jwR_sC_j).
        omega = 2 * np.pi * self.FREQ
        y_j = 1j * omega * 2e-10 / (1 + 1j * omega * 2e-3 * 2e-10)
        z_series = 1j * omega * 4e-7
        y_shunt = 1j * omega * 8e-11 + y_j
        assert gamma == pytest.approx(np.sqrt(z_series * y_shunt), rel=1e-12)
        assert z0 == pytest.approx(np.sqrt(z_series / y_shunt), rel=1e-12)
        assert np.all(gamma.real > 0)

    def test_rejects_mismatched_rlgc_shapes(self):
        bad = dict(self.UNLOADED, L=np.full(3, 4e-7))
        with pytest.raises(ValueError, match="shape"):
            loaded_line_params(self.FREQ, rlgc=bad, junction=(0.0, 0.0))


class TestDispersiveWalkoff:
    """The walk-off limit of a line whose RF index moves with frequency."""

    FREQ = np.linspace(10e9, 100e9, 10)
    LENGTH = 10e-3
    N_OPT = 3.8

    def test_flat_index_recovers_the_closed_form(self):
        expected = walkoff_bandwidth(length_m=self.LENGTH, n_rf=6.0, n_opt=self.N_OPT)
        assert walkoff_bandwidth_dispersive(
            self.FREQ, np.full(10, 6.0), length_m=self.LENGTH, n_opt=self.N_OPT
        ) == pytest.approx(expected)

    def test_velocity_matched_everywhere_is_unbounded(self):
        assert (
            walkoff_bandwidth_dispersive(
                self.FREQ, np.full(10, self.N_OPT), length_m=1e-3, n_opt=self.N_OPT
            )
            is None
        )

    def test_limit_is_self_consistent_inside_the_solved_range(self):
        n_rf = np.linspace(4.6, 4.0, 10)
        limit = walkoff_bandwidth_dispersive(
            self.FREQ, n_rf, length_m=self.LENGTH, n_opt=self.N_OPT
        )
        assert limit is not None
        assert self.FREQ[0] < limit < self.FREQ[-1]
        mismatch = abs(np.interp(limit, self.FREQ, n_rf) - self.N_OPT)
        assert limit == pytest.approx(
            walkoff_bandwidth(
                length_m=self.LENGTH, n_rf=self.N_OPT + mismatch, n_opt=self.N_OPT
            ),
            rel=1e-6,
        )

    def test_index_crossing_the_group_index_does_not_average_away(self):
        # n_RF crosses n_g mid-band, so its mean mismatch is ~0 and a limit
        # read off the mean runs away. Past the solved range the last
        # solved index is held, and the limit follows from that one.
        n_rf = np.linspace(4.2, 3.4, 10)
        limit = walkoff_bandwidth_dispersive(
            self.FREQ, n_rf, length_m=3e-3, n_opt=self.N_OPT
        )
        assert limit == pytest.approx(
            walkoff_bandwidth(length_m=3e-3, n_rf=3.4, n_opt=self.N_OPT)
        )

    def test_limit_below_the_solved_range_holds_the_first_index(self):
        limit = walkoff_bandwidth_dispersive(
            self.FREQ, np.linspace(9.0, 8.0, 10), length_m=50e-3, n_opt=self.N_OPT
        )
        assert limit < self.FREQ[0]
        assert limit == pytest.approx(
            walkoff_bandwidth(length_m=50e-3, n_rf=9.0, n_opt=self.N_OPT)
        )
