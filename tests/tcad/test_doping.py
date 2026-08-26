"""Pure-function tests for the analytic doping profiles (hand values)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from gsim.tcad.doping import (
    GaussianDoping,
    ImplantDoping,
    StepDoping,
    acceptor_donor_concentrations,
    net_doping_cm3,
)


class TestStepDoping:
    def test_uniform_without_windows(self):
        step = StepDoping(region="slab", dopant_type="donor", concentration_cm3=1e18)
        values = step.concentration([-5.0, 0.0, 5.0], [0.0, 0.1, 0.2])
        np.testing.assert_allclose(values, 1e18)

    def test_box_windows(self):
        step = StepDoping(
            region="slab",
            dopant_type="acceptor",
            concentration_cm3=2e17,
            x_range=(-1.0, 1.0),
            y_range=(0.0, 0.2),
        )
        x = np.array([0.0, 0.0, 2.0, -1.0])
        y = np.array([0.1, 0.5, 0.1, 0.0])
        # Inside; y outside; x outside; on the closed boundary.
        np.testing.assert_allclose(step.concentration(x, y), [2e17, 0.0, 0.0, 2e17])

    def test_scalar_input(self):
        step = StepDoping(region="slab", dopant_type="donor", concentration_cm3=1e18)
        assert float(step.concentration(0.0, 0.0)) == 1e18

    def test_rejects_descending_range(self):
        with pytest.raises(ValidationError):
            StepDoping(
                region="slab",
                dopant_type="donor",
                concentration_cm3=1e18,
                x_range=(1.0, -1.0),
            )

    def test_rejects_nonpositive_concentration(self):
        with pytest.raises(ValidationError):
            StepDoping(region="slab", dopant_type="donor", concentration_cm3=0.0)


class TestGaussianDoping:
    def test_peak_and_sigma_hand_values(self):
        gauss = GaussianDoping(
            region="slab",
            dopant_type="acceptor",
            peak_cm3=1e19,
            center=(0.0, 0.0),
            sigma_x=0.5,
        )
        assert float(gauss.concentration(0.0, 0.0)) == pytest.approx(1e19)
        # One sigma off-center: peak * exp(-1/2).
        assert float(gauss.concentration(0.5, 0.0)) == pytest.approx(
            1e19 * math.exp(-0.5)
        )
        # Uniform along y when sigma_y omitted.
        assert float(gauss.concentration(0.0, 3.0)) == pytest.approx(1e19)

    def test_separable_both_axes(self):
        gauss = GaussianDoping(
            region="slab",
            dopant_type="donor",
            peak_cm3=1e18,
            center=(1.0, 0.1),
            sigma_x=0.2,
            sigma_y=0.05,
        )
        expected = 1e18 * math.exp(-0.5) * math.exp(-0.5)
        assert float(gauss.concentration(1.2, 0.15)) == pytest.approx(expected)

    def test_hard_window_clips_tail(self):
        gauss = GaussianDoping(
            region="slab",
            dopant_type="donor",
            peak_cm3=1e18,
            center=(0.0, 0.0),
            sigma_x=1.0,
            x_range=(-1.0, 1.0),
        )
        assert float(gauss.concentration(2.0, 0.0)) == 0.0

    def test_requires_a_sigma(self):
        with pytest.raises(ValidationError):
            GaussianDoping(
                region="slab",
                dopant_type="donor",
                peak_cm3=1e18,
                center=(0.0, 0.0),
            )


class TestImplantDoping:
    def test_peak_at_projected_range(self):
        implant = ImplantDoping(
            region="slab",
            dopant_type="acceptor",
            peak_cm3=5e18,
            surface_y=0.22,
            range_um=0.1,
            straggle_um=0.03,
        )
        # Peak at depth Rp below the surface.
        assert float(implant.concentration(0.0, 0.12)) == pytest.approx(5e18)
        # One straggle deeper: peak * exp(-1/2).
        assert float(implant.concentration(0.0, 0.09)) == pytest.approx(
            5e18 * math.exp(-0.5)
        )

    def test_zero_above_surface(self):
        implant = ImplantDoping(
            region="slab",
            dopant_type="acceptor",
            peak_cm3=5e18,
            surface_y=0.22,
            range_um=0.1,
            straggle_um=0.03,
        )
        assert float(implant.concentration(0.0, 0.3)) == 0.0

    def test_lateral_window(self):
        implant = ImplantDoping(
            region="slab",
            dopant_type="donor",
            peak_cm3=1e20,
            surface_y=0.0,
            range_um=0.05,
            straggle_um=0.02,
            x_range=(2.0, 4.0),
        )
        assert float(implant.concentration(3.0, -0.05)) == pytest.approx(1e20)
        assert float(implant.concentration(0.0, -0.05)) == 0.0


class TestSuperposition:
    def test_acceptors_and_donors_summed_separately(self):
        profiles = [
            StepDoping(
                region="slab",
                dopant_type="acceptor",
                concentration_cm3=1e18,
                x_range=(-2.0, 0.0),
            ),
            StepDoping(
                region="slab",
                dopant_type="donor",
                concentration_cm3=4e17,
                x_range=(0.0, 2.0),
            ),
            StepDoping(region="slab", dopant_type="donor", concentration_cm3=1e15),
        ]
        x = np.array([-1.0, 1.0])
        y = np.zeros(2)
        acceptors, donors = acceptor_donor_concentrations(profiles, x, y)
        np.testing.assert_allclose(acceptors, [1e18, 0.0])
        np.testing.assert_allclose(donors, [1e15, 4e17 + 1e15])

    def test_net_doping_sign_convention(self):
        profiles = [
            StepDoping(region="slab", dopant_type="acceptor", concentration_cm3=1e18),
            StepDoping(region="slab", dopant_type="donor", concentration_cm3=4e17),
        ]
        net = net_doping_cm3(profiles, 0.0, 0.0)
        # donors - acceptors: p-type net doping is negative.
        assert float(net) == pytest.approx(4e17 - 1e18)
