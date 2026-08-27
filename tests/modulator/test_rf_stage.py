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

import numpy as np
import pytest

from gsim.common.stack.staircase import ElectrodeSpec
from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap

from .conftest import CENTER_Y, HALF_WIDTH, PAD_WIDTH, RIB_HEIGHT

SLAB = (CENTER_Y - HALF_WIDTH - PAD_WIDTH, CENTER_Y + HALF_WIDTH + PAD_WIDTH)


def carriers_at(bias_v: float) -> CarrierMap:
    """A Carrier map across the doped slab, depleting with reverse bias."""
    y = np.linspace(SLAB[0], SLAB[1], 61)
    z = np.linspace(0.0, RIB_HEIGHT, 5)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    yy, zz = yy.ravel(), zz.ravel()
    depleted = np.abs(yy - CENTER_Y) < 0.05 * np.sqrt(1.0 + abs(bias_v))
    n_side = yy < CENTER_Y
    return CarrierMap(
        x_um=yy,
        y_um=zz,
        region=["n_rib" if side else "p_rib" for side in n_side],
        electrons_cm3=np.where(depleted | ~n_side, 1e10, 1e18),
        holes_cm3=np.where(depleted | n_side, 1e10, 1e18),
        potential_v=np.zeros(yy.size),
        net_doping_cm3=np.zeros(yy.size),
    )


@pytest.fixture
def biased(study):
    """A Study whose charge Stage already holds a two-point sweep."""
    study.charge._result = BiasSweepResult(
        contact="cathode",
        points=[BiasPoint(bias_v=v, carriers=carriers_at(v)) for v in (0.0, 2.0)],
    )
    study.charge._has_run = True
    return study


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
