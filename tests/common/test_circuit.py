"""The line two-port export: S-matrix, Touchstone writer, SAX callable.

Everything here is pure array math against the analytic lossy-line
S-parameters, so nothing solves anything. The independent reference the
S-matrix is checked against is the reflection-coefficient form of the
same network — a different derivation of the same physics, coded from
scratch in this file — plus the closed-form special cases (matched,
half-wave, lossless-unitary) where the answer is a number.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.common.circuit import (
    line_smatrix,
    sax_line_model,
    write_touchstone,
)

FREQ_HZ = np.asarray([1e9, 10e9, 40e9], dtype=np.float64)
LENGTH_M = 3e-3
Z_REF = 50.0


def lossy_line() -> tuple[np.ndarray, np.ndarray]:
    """A mismatched lossy line's gamma(f) and Z0(f) over FREQ_HZ."""
    n_rf = np.asarray([3.4, 3.2, 3.1])
    alpha = np.asarray([20.0, 80.0, 220.0])  # Np/m
    gamma = alpha + 1j * 2.0 * np.pi * FREQ_HZ * n_rf / 299792458.0
    z0 = np.asarray([42.0 + 4.0j, 44.0 + 2.0j, 46.0 + 1.0j])
    return gamma, z0


def reference_smatrix(gamma, z0, length_m, z_ref):
    """The same two-port from the reflection-coefficient derivation."""
    reflection = (z0 - z_ref) / (z0 + z_ref)
    phase = np.exp(-gamma * length_m)
    denom = 1.0 - reflection**2 * phase**2
    s11 = reflection * (1.0 - phase**2) / denom
    s21 = (1.0 - reflection**2) * phase / denom
    return s11, s21


class TestLineSmatrix:
    def test_matched_line_is_reflectionless_and_delays(self):
        gamma, _ = lossy_line()

        s = line_smatrix(gamma, Z_REF, length_m=LENGTH_M, z_ref_ohm=Z_REF)

        np.testing.assert_allclose(s[:, 0, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(s[:, 1, 0], np.exp(-gamma * LENGTH_M), rtol=1e-12)

    def test_matches_the_reflection_coefficient_derivation(self):
        gamma, z0 = lossy_line()

        s = line_smatrix(gamma, z0, length_m=LENGTH_M, z_ref_ohm=Z_REF)
        s11, s21 = reference_smatrix(gamma, z0, LENGTH_M, Z_REF)

        np.testing.assert_allclose(s[:, 0, 0], s11, rtol=1e-10)
        np.testing.assert_allclose(s[:, 1, 0], s21, rtol=1e-10)

    def test_the_network_is_reciprocal_and_symmetric(self):
        gamma, z0 = lossy_line()

        s = line_smatrix(gamma, z0, length_m=LENGTH_M, z_ref_ohm=Z_REF)

        np.testing.assert_allclose(s[:, 0, 1], s[:, 1, 0], rtol=1e-12)
        np.testing.assert_allclose(s[:, 0, 0], s[:, 1, 1], rtol=1e-12)

    def test_a_lossless_line_is_unitary(self):
        beta = 2.0 * np.pi * FREQ_HZ * 3.2 / 299792458.0
        gamma = 1j * beta

        s = line_smatrix(gamma, 42.0, length_m=LENGTH_M, z_ref_ohm=Z_REF)

        power = np.abs(s[:, 0, 0]) ** 2 + np.abs(s[:, 1, 0]) ** 2
        np.testing.assert_allclose(power, 1.0, rtol=1e-12)

    def test_a_lossless_half_wave_line_disappears(self):
        # beta L = pi: any lossless line is reflectionless and inverts.
        freq = 10e9
        n_rf = 299792458.0 / (2.0 * freq * LENGTH_M)
        gamma = 1j * 2.0 * np.pi * freq * n_rf / 299792458.0

        s = line_smatrix(gamma, 137.0, length_m=LENGTH_M, z_ref_ohm=Z_REF)

        np.testing.assert_allclose(s[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(s[1, 0], -1.0, rtol=1e-10)

    def test_a_nonpositive_length_is_refused(self):
        gamma, z0 = lossy_line()

        with pytest.raises(ValueError, match="length"):
            line_smatrix(gamma, z0, length_m=0.0, z_ref_ohm=Z_REF)

    def test_mismatched_shapes_are_refused(self):
        gamma, _ = lossy_line()

        with pytest.raises(ValueError, match="shape"):
            line_smatrix(gamma, np.asarray([50.0, 50.0]), length_m=LENGTH_M)


class TestTouchstone:
    def test_scikit_rf_reads_back_the_same_network(self, tmp_path):
        skrf = pytest.importorskip("skrf")
        gamma, z0 = lossy_line()
        s = line_smatrix(gamma, z0, length_m=LENGTH_M, z_ref_ohm=Z_REF)

        path = write_touchstone(
            tmp_path / "line.s2p", freq_hz=FREQ_HZ, s=s, z_ref_ohm=Z_REF
        )
        network = skrf.Network(str(path))

        np.testing.assert_allclose(network.f, FREQ_HZ)
        np.testing.assert_allclose(network.s, s, atol=1e-9)
        np.testing.assert_allclose(network.z0, Z_REF)

    def test_the_suffix_is_supplied_when_missing(self, tmp_path):
        gamma, z0 = lossy_line()
        s = line_smatrix(gamma, z0, length_m=LENGTH_M)

        path = write_touchstone(tmp_path / "line", freq_hz=FREQ_HZ, s=s)

        assert path.suffix == ".s2p"
        assert path.exists()

    def test_a_complex_reference_is_refused(self, tmp_path):
        gamma, z0 = lossy_line()
        s = line_smatrix(gamma, z0, length_m=LENGTH_M)

        with pytest.raises(ValueError, match="real"):
            write_touchstone(
                tmp_path / "line.s2p", freq_hz=FREQ_HZ, s=s, z_ref_ohm=50.0 + 1j
            )

    def test_comment_lines_carry_the_provenance(self, tmp_path):
        gamma, z0 = lossy_line()
        s = line_smatrix(gamma, z0, length_m=LENGTH_M)

        path = write_touchstone(
            tmp_path / "line.s2p",
            freq_hz=FREQ_HZ,
            s=s,
            comments=["length_m = 0.003"],
        )

        header = path.read_text().splitlines()
        assert "! length_m = 0.003" in header


class TestSaxLineModel:
    def test_the_default_frequencies_reproduce_the_solved_matrix(self):
        gamma, z0 = lossy_line()
        s = line_smatrix(gamma, z0, length_m=LENGTH_M, z_ref_ohm=Z_REF)

        model = sax_line_model(FREQ_HZ, gamma, z0, length_m=LENGTH_M, z_ref_ohm=Z_REF)
        sdict = model()

        np.testing.assert_allclose(sdict[("o1", "o1")], s[:, 0, 0], rtol=1e-12)
        np.testing.assert_allclose(sdict[("o2", "o1")], s[:, 1, 0], rtol=1e-12)
        np.testing.assert_allclose(sdict[("o1", "o2")], s[:, 0, 1], rtol=1e-12)
        np.testing.assert_allclose(sdict[("o2", "o2")], s[:, 1, 1], rtol=1e-12)

    def test_between_solved_points_the_line_parameters_interpolate(self):
        gamma, z0 = lossy_line()
        f_mid = 5.5e9

        model = sax_line_model(FREQ_HZ, gamma, z0, length_m=LENGTH_M, z_ref_ohm=Z_REF)
        sdict = model(f=f_mid)

        gamma_mid = np.interp(f_mid, FREQ_HZ, gamma.real) + 1j * np.interp(
            f_mid, FREQ_HZ, gamma.imag
        )
        z0_mid = np.interp(f_mid, FREQ_HZ, z0.real) + 1j * np.interp(
            f_mid, FREQ_HZ, z0.imag
        )
        expected = line_smatrix(gamma_mid, z0_mid, length_m=LENGTH_M, z_ref_ohm=Z_REF)
        np.testing.assert_allclose(sdict[("o2", "o1")], expected[..., 1, 0])

    def test_a_scalar_frequency_gives_scalar_entries(self):
        gamma, z0 = lossy_line()

        model = sax_line_model(FREQ_HZ, gamma, z0, length_m=LENGTH_M)
        sdict = model(f=10e9)

        assert np.shape(sdict[("o2", "o1")]) == ()

    def test_the_callable_carries_no_solver_state(self):
        # The SAX convention is a plain function over numpy arrays: the
        # closure holds copies, so mutating the inputs cannot move it.
        gamma, z0 = lossy_line()
        model = sax_line_model(FREQ_HZ, gamma, z0, length_m=LENGTH_M)
        before = model()[("o2", "o1")].copy()

        gamma += 1e3
        z0 += 10.0

        np.testing.assert_allclose(model()[("o2", "o1")], before, rtol=1e-12)
