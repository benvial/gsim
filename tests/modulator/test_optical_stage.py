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

    def test_the_optical_stage_is_last_for_now(self, study):
        assert study.optical._downstream == []
