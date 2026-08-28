"""The optical Stage: its Window, its configuration, and the missing extra.

Everything here runs without femwell, skfem or DEVSIM: the Window
derivation and the Stage's configuration are pure derivation, and the
missing-extra path is exactly the one a user without the extra hits.
The real solve lives in ``test_optical_stage_runtime.py``.
"""

from __future__ import annotations

import sys

import pytest

from .conftest import CENTER_Y, HALF_WIDTH, PAD_WIDTH, RIB_HEIGHT

JUNCTION_Y = CENTER_Y


class TestConfiguration:
    def test_defaults_are_readable(self, study):
        assert study.optical.wavelength_um == 1.55
        assert study.optical.num_modes == 1
        assert study.optical.window is None
        assert study.optical.has_run is False

    def test_the_section_is_callable(self, study):
        assert study.optical(wavelength_um=1.31, num_modes=2) is study.optical
        assert study.optical.wavelength_um == 1.31
        assert study.optical.num_modes == 2

    def test_unknown_setting_is_rejected(self, study):
        with pytest.raises(ValueError, match="nope"):
            study.optical(nope=1)

    def test_a_nonsense_wavelength_is_rejected(self, study):
        with pytest.raises(ValueError):
            study.optical(wavelength_um=0.0)

    def test_at_least_one_mode_must_be_asked_for(self, study):
        with pytest.raises(ValueError):
            study.optical(num_modes=0)


class TestDerivedWindow:
    def test_the_window_is_a_box_around_the_rib_not_the_doped_slab(self, study):
        window = study.optical.simulation().cross_section.window

        # Centred on the junction, and wider than the charge window, which
        # is clipped to the doped slab between the contacts (ADR 0002).
        assert window == pytest.approx((JUNCTION_Y - 2.0, JUNCTION_Y + 2.0))
        assert window != pytest.approx(study.layout.window)

    def test_the_vertical_window_clears_the_guiding_layer(self, study):
        window_z = study.optical.simulation().cross_section.window_z

        assert window_z == pytest.approx((-1.0, RIB_HEIGHT + 1.0))

    def test_the_margin_sizes_the_window(self, study):
        study.optical(mode_margin_um=3.0, z_above_um=0.5, z_below_um=0.25)

        cross_section = study.optical.simulation().cross_section

        assert cross_section.window == pytest.approx(
            (JUNCTION_Y - 3.0, JUNCTION_Y + 3.0)
        )
        assert cross_section.window_z == pytest.approx((-0.25, RIB_HEIGHT + 0.5))

    def test_an_explicit_window_overrides_the_derivation(self, study):
        study.optical(window=(-21.0, -19.0), window_z=(-0.5, 0.8))

        cross_section = study.optical.simulation().cross_section

        assert cross_section.window == pytest.approx((-21.0, -19.0))
        assert cross_section.window_z == pytest.approx((-0.5, 0.8))

    def test_the_junction_is_where_the_doped_regions_meet(self, study):
        assert study.layout.junction_position == pytest.approx(JUNCTION_Y)

    def test_the_charge_window_spans_the_whole_doped_slab(self, study):
        """The Window the optical Stage is deliberately not reusing."""
        assert study.layout.window == pytest.approx(
            (
                CENTER_Y - HALF_WIDTH - PAD_WIDTH - 0.5,
                CENTER_Y + HALF_WIDTH + PAD_WIDTH + 0.5,
            )
        )


class TestOwnMesh:
    def test_the_optical_stage_writes_into_its_own_directory(self, study):
        sim = study.optical.simulation()
        assert sim.output_dir == study.stage_dir("optical")
        assert sim.output_dir != study.stage_dir("charge")

    def test_the_perturbed_regions_default_to_the_doped_ones(self, study):
        assert study.optical.perturbed_region_names() == study.device.doped_regions

    def test_the_perturbed_regions_are_overridable(self, study):
        study.optical(perturbed_regions=["p_rib", "n_rib"])
        assert study.optical.perturbed_region_names() == ["p_rib", "n_rib"]


class TestMissingExtra:
    def test_running_without_femwell_names_the_extra(self, study, monkeypatch):
        monkeypatch.setitem(sys.modules, "femwell", None)
        with pytest.raises(ImportError, match=r"gsim\[femwell\]"):
            study.optical.run()

    def test_the_extra_is_checked_before_anything_is_meshed(self, study, monkeypatch):
        """A user without femwell pays for no mesh and no charge solve."""
        monkeypatch.setitem(sys.modules, "femwell", None)

        def fail(*_args, **_kwargs):
            raise AssertionError("the charge stage must not run")

        monkeypatch.setattr("gsim.modulator.charge.ChargeStage._solve", fail)
        with pytest.raises(ImportError):
            study.optical.run()


class TestInvalidation:
    def test_a_carriers_change_drops_the_optical_result(self, study):
        order = list(study.stages)
        assert order.index("carriers") < order.index("optical")
        assert study.optical in study.carriers._downstream
        assert study.optical in study.charge._downstream

    def test_the_optical_stage_feeds_the_line_stage(self, study):
        assert study.optical._downstream == [study.line]


class TestStaircaseWavelength:
    """The Staircase and the continuous profile solve the one problem.

    Both paths turn the same carriers into a complex permittivity, and
    the only wavelength either may use for that is the one the Stage is
    solving at. The plasma-dispersion model's own wavelength is where its
    coefficients were fitted; using it instead inflates every Strip's
    free-carrier absorption whenever the Stage solves somewhere else.
    """

    @staticmethod
    def _strips(study, *, wavelength_um: float):
        """The Strip response of a single-Bias Staircase at a wavelength."""
        import numpy as np

        from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap

        h = np.linspace(JUNCTION_Y - HALF_WIDTH, JUNCTION_Y + HALF_WIDTH, 41)
        z = np.linspace(0.0, RIB_HEIGHT, 5)
        hh, zz = (a.ravel() for a in np.meshgrid(h, z))
        n_side = hh < JUNCTION_Y
        study.charge._result = BiasSweepResult(
            contact="cathode",
            points=[
                BiasPoint(
                    bias_v=0.0,
                    carriers=CarrierMap(
                        x_um=hh,
                        y_um=zz,
                        region=["n_rib" if side else "p_rib" for side in n_side],
                        electrons_cm3=np.where(n_side, 1e18, 1e10),
                        holes_cm3=np.where(n_side, 1e10, 1e18),
                        potential_v=np.zeros(hh.size),
                        net_doping_cm3=np.zeros(hh.size),
                    ),
                )
            ],
        )
        study.charge._has_run = True
        study.optical(route="femwell", n_strips=4, wavelength_um=wavelength_um)
        return study.optical.staircase(study.carriers.run().points[0]).strips

    def test_the_strips_take_the_extinction_of_the_solve_wavelength(self, study):
        """What the Staircase carries is what the continuous path builds."""
        from gsim.common.carriers import permittivity_perturbation

        wavelength_um = 1.31
        assert study.carriers.dispersion.wavelength_um != wavelength_um
        strips = self._strips(study, wavelength_um=wavelength_um)

        for i, eps in enumerate(strips["eps_complex"]):
            expected = permittivity_perturbation(
                n0=study.optical.strip_index,
                dn=float(strips["dn"][i]),
                dalpha_cm=float(strips["dalpha_cm"][i]),
                wavelength_um=wavelength_um,
            )
            assert eps == pytest.approx(expected)

    def test_moving_the_solve_wavelength_moves_the_strip_loss(self, study):
        """And it is the solve wavelength that moves it, not the fit."""
        at_fit = self._strips(study, wavelength_um=1.55)["eps_complex"]
        at_solve = self._strips(study, wavelength_um=1.31)["eps_complex"]

        for fitted, solved in zip(at_fit, at_solve, strict=True):
            assert solved.imag == pytest.approx(fitted.imag * 1.31 / 1.55)
