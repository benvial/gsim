"""The RF Stage: its Staircase, its signal conductor, its Window.

Nothing here needs gmsh or femwell. The Staircase is built from a canned
Bias sweep, so what is under test is the derivation the Stage does — the
Strips tiling the Junction extent, the electrode carrying the RF signal,
and the Window defaulting to the whole Cross-section — plus the error a
user without the femwell extra gets. The real solve lives in
``test_rf_stage_runtime.py``.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

from gsim.common.stack.staircase import ElectrodeSpec

from .conftest import CENTER_Y, HALF_WIDTH, PAD_WIDTH, SLAB


class TestConfiguration:
    def test_defaults_are_readable(self, biased):
        assert biased.rf.frequencies_hz == [10e9, 40e9]
        assert biased.rf.n_strips == 5
        assert biased.rf.bias_v is None
        assert biased.rf.window is None
        assert biased.rf.has_run is False

    def test_the_section_is_callable(self, biased):
        assert biased.rf(frequencies_hz=[20e9], n_strips=3) is biased.rf
        assert biased.rf.frequencies_hz == [20e9]
        assert biased.rf.n_strips == 3

    def test_unknown_setting_is_rejected(self, biased):
        with pytest.raises(ValueError, match="nope"):
            biased.rf(nope=1)

    def test_a_frequency_list_that_is_empty_is_rejected(self, biased):
        with pytest.raises(ValueError):
            biased.rf(frequencies_hz=[])

    def test_a_nonpositive_frequency_is_rejected(self, biased):
        with pytest.raises(ValueError, match="positive"):
            biased.rf(frequencies_hz=[0.0])

    def test_fewer_than_one_strip_is_rejected(self, biased):
        with pytest.raises(ValueError):
            biased.rf(n_strips=0)


class TestBiasPoint:
    def test_the_last_point_of_the_sweep_is_the_default(self, biased):
        assert biased.rf.bias_point().bias_v == 2.0

    def test_a_chosen_bias_selects_its_point(self, biased):
        biased.rf(bias_v=0.0)
        assert biased.rf.bias_point().bias_v == 0.0

    def test_a_bias_the_sweep_never_visited_is_reported(self, biased):
        biased.rf(bias_v=-3.0)
        with pytest.raises(ValueError, match=r"-3|0\.0, 2\.0"):
            biased.rf.bias_point()


class TestStaircase:
    def test_the_strips_tile_the_junction_extent(self, biased):
        biased.rf(n_strips=4)

        staircase = biased.rf.staircase()

        assert len(staircase.strip_names) == 4
        edges = np.asarray(staircase.strips["edges_um"], dtype=float)
        assert edges[0] == pytest.approx(CENTER_Y - HALF_WIDTH)
        assert edges[-1] == pytest.approx(CENTER_Y + HALF_WIDTH)

    def test_the_strip_span_is_overridable(self, biased):
        """A wider span carries the pads, and their resistance, into RF."""
        slab = (CENTER_Y - HALF_WIDTH - PAD_WIDTH, CENTER_Y + HALF_WIDTH + PAD_WIDTH)
        biased.rf(n_strips=4, strip_span=slab)

        edges = np.asarray(biased.rf.staircase().strips["edges_um"], dtype=float)

        assert edges[0] == pytest.approx(slab[0])
        assert edges[-1] == pytest.approx(slab[1])

    def test_strips_too_wide_for_the_rib_are_reported(self, biased):
        """Widening the span without more strips loses the junction."""
        biased.rf(n_strips=2, strip_span=SLAB)
        with pytest.warns(UserWarning, match="not resolved"):
            biased.rf.staircase()

    def test_strips_that_resolve_the_rib_are_not_reported(self, biased):
        biased.rf(n_strips=5, strip_span=SLAB)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            biased.rf.staircase()

    def test_the_strips_carry_the_carrier_derived_conductivity(self, biased):
        staircase = biased.rf.staircase()

        sigma = np.asarray(staircase.strips["sigma_s_per_m"], dtype=float)
        assert np.all(sigma > 0.0)
        # mu_n > mu_p, and the n side is the low-h one on this device.
        assert sigma[0] > sigma[-1]

    def test_the_mobilities_come_from_the_carriers_stage(self, biased):
        biased.carriers(mu_n_cm2=1.0, mu_p_cm2=1.0)
        slow = np.asarray(biased.rf.staircase().strips["sigma_s_per_m"], dtype=float)

        biased.carriers(mu_n_cm2=1000.0, mu_p_cm2=1000.0)
        fast = np.asarray(biased.rf.staircase().strips["sigma_s_per_m"], dtype=float)

        assert np.all(fast > slow)

    def test_the_electrodes_flank_the_junction_extent(self, biased):
        biased.rf(electrodes=ElectrodeSpec(width_um=3.0, gap_um=0.5))

        staircase = biased.rf.staircase()

        low, high = staircase.electrode_spans
        assert low == pytest.approx((CENTER_Y - HALF_WIDTH - 3.5, CENTER_Y - 0.8))
        assert high == pytest.approx((CENTER_Y + 0.8, CENTER_Y + HALF_WIDTH + 3.5))

    def test_the_caller_never_assembles_a_second_component(self, biased):
        staircase = biased.rf.staircase()

        assert staircase.component is not biased.component
        assert set(staircase.strip_names) <= set(staircase.stack("rf").layers)


class TestConductorModel:
    """Which model of the electrode metal a run takes (ADR 0003)."""

    def test_the_femwell_route_meshes_the_metal_as_a_region(self, biased):
        """femwell can carry the metal's own loss, so it does."""
        assert biased.rf.effective_conductor_model() == "volume"
        staircase = biased.rf.staircase()
        assert staircase.conductor_model == "volume"
        stack = staircase.stack("rf")
        assert stack.layers[staircase.electrode_names[0]].layer_type == "dielectric"

    def test_the_palace_route_meshes_the_metal_as_a_perfect_conductor(self, biased):
        """A metal region takes palace's eigenvalue search over; an
        outline does not."""
        biased.rf(route="palace")
        assert biased.rf.effective_conductor_model() == "pec"
        staircase = biased.rf.staircase()
        assert staircase.conductor_model == "pec"
        stack = staircase.stack("rf")
        assert stack.layers[staircase.electrode_names[0]].layer_type == "conductor"

    @pytest.mark.parametrize("route", ["femwell", "palace"])
    @pytest.mark.parametrize("model", ["volume", "pec"])
    def test_an_explicit_model_overrides_the_route_default(self, biased, route, model):
        """What makes the two routes comparable: one cross-section, both."""
        biased.rf(route=route, conductor_model=model)
        assert biased.rf.effective_conductor_model() == model
        assert biased.rf.staircase().conductor_model == model

    def test_changing_the_model_invalidates_the_result(self, biased):
        biased.rf._result = object()
        biased.rf._has_run = True
        biased.rf(conductor_model="pec")
        assert not biased.rf.has_run


class TestPerfectElectrodesNeedTheWall:
    """femwell has one perfect-conductor condition, for the whole boundary."""

    def test_a_pec_electrode_without_the_wall_is_refused(self, biased):
        """Off, the electrode hole would come out as an open slot."""
        biased.rf(conductor_model="pec", metallic_boundaries=False)
        with pytest.raises(ValueError, match="open slots"):
            biased.rf._require_metallic_boundaries()

    def test_a_pec_electrode_with_the_wall_is_fine(self, biased):
        biased.rf(conductor_model="pec", metallic_boundaries=True)
        biased.rf._require_metallic_boundaries()

    def test_a_volume_electrode_does_not_need_the_wall(self, biased):
        """A metal region is a conductor whatever the boundary is."""
        biased.rf(conductor_model="volume", metallic_boundaries=False)
        assert biased.rf.effective_conductor_model() == "volume"

    def test_the_refusal_comes_before_anything_is_meshed(self, study, monkeypatch):
        """Settings the Route cannot honour cost no charge solve and no mesh."""

        def fail(*_args, **_kwargs):
            raise AssertionError("the charge stage must not run")

        monkeypatch.setattr("gsim.modulator.charge.ChargeStage._solve", fail)
        study.rf(route="femwell", conductor_model="pec", metallic_boundaries=False)
        with pytest.raises(ValueError, match="open slots"):
            study.rf.run()


class TestMetallicWall:
    """Both Routes put the same condition on the Window's outer wall."""

    def test_the_simulation_carries_the_wall_by_default(self, biased):
        assert biased.rf.metallic_boundaries is True
        assert biased.rf.simulation().metallic_boundaries is True

    def test_turning_the_wall_off_reaches_the_simulation(self, biased):
        biased.rf(conductor_model="volume", metallic_boundaries=False)
        assert biased.rf.simulation().metallic_boundaries is False


class TestContourOrder:
    """A perfect conductor's current is read off the field around it."""

    def test_a_first_order_solve_of_a_pec_staircase_is_reported(self, biased):
        biased.rf(conductor_model="pec", order=1)
        with pytest.warns(UserWarning, match="biased high by tens of percent"):
            biased.rf._check_contour_order()

    def test_a_second_order_solve_is_not(self, biased):
        biased.rf(conductor_model="pec", order=2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            biased.rf._check_contour_order()


class TestStripMaterials:
    def test_the_strips_are_valid_up_to_the_highest_frequency_solved(self, biased):
        biased.rf(frequencies_hz=[10e9, 90e9])

        materials = biased.rf.staircase().doping("rf")["materials"]
        model = materials["strip_0"].dispersion_models[0]

        assert model.validity.valid_frequency == (0, 90e9)


class TestSignalConductor:
    def test_the_signal_conductor_is_the_swept_contacts_electrode(self, biased):
        # The charge sweep drives the n-side contact, which is the low-h
        # side of this device, so the low electrode carries the signal.
        assert biased.charge.swept_contact() == "cathode"
        assert biased.rf.signal_contact_name() == "cathode"
        assert biased.rf.signal_electrode() == "electrode_low"

    def test_the_other_contact_selects_the_other_electrode(self, biased):
        biased.rf(signal_contact="anode")
        assert biased.rf.signal_electrode() == "electrode_high"

    def test_a_contact_the_device_does_not_have_is_reported(self, biased):
        biased.rf(signal_contact="gate")
        with pytest.raises(ValueError, match="gate"):
            biased.rf.signal_electrode()

    def test_renamed_electrodes_are_followed(self, biased):
        biased.rf(electrodes=ElectrodeSpec(names=("ground", "signal")))
        biased.rf(signal_contact="anode")
        assert biased.rf.signal_electrode() == "signal"


class TestModeTracking:
    def test_the_first_frequency_is_aimed_at_the_configured_guess(self, biased):
        biased.rf(n_guess=2.5)
        assert biased.rf._guess_for([]) == pytest.approx(2.5)

    def test_later_frequencies_follow_the_mode_they_just_solved(self, biased):
        biased.rf(n_guess=2.5)
        assert biased.rf._guess_for([complex(3.9, -0.2)]) == pytest.approx(3.9)

    def test_tracking_is_switchable_off(self, biased):
        biased.rf(n_guess=2.5, track_modes=False)
        assert biased.rf._guess_for([complex(3.9, -0.2)]) == pytest.approx(2.5)

    def test_a_dense_sweep_is_judged_on_its_own_spacing(self, biased):
        """A step that is fine over a doubling is a jump over 1%."""
        biased.rf(frequencies_hz=[40e9, 40.4e9])
        with pytest.warns(UserWarning, match="jumps across the sweep"):
            biased.rf._check_continuity([3.90 + 0j, 3.70 + 0j])

    def test_a_dispersing_index_is_not_reported_as_a_jump(self, biased):
        biased.rf(frequencies_hz=[10e9, 20e9, 40e9])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            biased.rf._check_continuity([3.90 + 0j, 3.85 + 0j, 3.80 + 0j])

    def test_an_index_that_steps_between_frequencies_is_reported(self, biased):
        biased.rf(frequencies_hz=[10e9, 20e9, 40e9])
        with pytest.warns(UserWarning, match=r"20 -> 40 GHz"):
            biased.rf._check_continuity([3.90 + 0j, 3.85 + 0j, 0.4 + 0j])


class TestWindow:
    def test_the_window_defaults_to_the_full_cross_section(self, biased):
        cross_section = biased.rf.simulation().cross_section

        assert cross_section.window is None
        assert cross_section.window_z is None

    def test_an_explicit_window_overrides_the_default(self, biased):
        biased.rf(window=(CENTER_Y - 4.0, CENTER_Y + 4.0), window_z=(-1.0, 1.0))

        cross_section = biased.rf.simulation().cross_section

        assert cross_section.window == pytest.approx((CENTER_Y - 4.0, CENTER_Y + 4.0))
        assert cross_section.window_z == pytest.approx((-1.0, 1.0))

    def test_the_rf_stage_writes_into_its_own_directory(self, biased):
        sim = biased.rf.simulation()

        assert sim.output_dir == biased.stage_dir("rf")
        assert sim.output_dir != biased.stage_dir("optical")


class TestInvalidation:
    def test_the_rf_stage_is_downstream_of_the_carriers_stage(self, study):
        assert study.rf in study.carriers._downstream
        assert study.rf in study.charge._downstream

    def test_the_optical_and_rf_stages_do_not_invalidate_each_other(self, study):
        assert study.rf not in study.optical._downstream
        assert study.optical not in study.rf._downstream


class TestMissingExtra:
    def test_running_without_femwell_names_the_extra(self, biased, monkeypatch):
        monkeypatch.setitem(sys.modules, "femwell", None)
        with pytest.raises(ImportError, match=r"gsim\[femwell\]"):
            biased.rf.run()

    def test_the_extra_is_checked_before_anything_is_meshed(self, study, monkeypatch):
        """A user without femwell pays for no mesh and no charge solve."""
        monkeypatch.setitem(sys.modules, "femwell", None)

        def fail(*_args, **_kwargs):
            raise AssertionError("the charge stage must not run")

        monkeypatch.setattr("gsim.modulator.charge.ChargeStage._solve", fail)
        with pytest.raises(ImportError):
            study.rf.run()
