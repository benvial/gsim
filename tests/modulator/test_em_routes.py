"""Route selection on the two EM Stages, without any solver runtime.

Both EM Stages answer the same question through either Backend, and the
choice is a Stage setting. What is hermetic about that choice — the
default, the values accepted, what a strip count means on each Stage, the
Staircase the optical Stage builds when it is routed to Palace, and the
error a user selecting a Route they cannot run gets — is under test here.
The Routes actually agreeing on a number is the runtime-gated
``test_palace_route_runtime.py``.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from pydantic import ValidationError

from gsim.modulator import DEFAULT_PALACE_STRIPS, OpticalStage, RFStage
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


class TestRouteSelection:
    def test_both_em_stages_default_to_femwell(self):
        assert OpticalStage().route == "femwell"
        assert RFStage().route == "femwell"

    @pytest.mark.parametrize("stage", [OpticalStage, RFStage])
    def test_palace_is_selectable(self, stage):
        assert stage()(route="palace").route == "palace"

    @pytest.mark.parametrize("stage", [OpticalStage, RFStage])
    def test_an_unknown_route_is_rejected(self, stage):
        with pytest.raises(ValidationError, match="route"):
            stage()(route="comsol")

    def test_changing_the_route_invalidates_the_stage(self, biased):
        biased.rf._result = object()
        biased.rf._has_run = True
        biased.rf(route="palace")
        assert biased.rf.has_run is False


class TestOpticalStripCount:
    def test_the_continuous_profile_is_the_default(self):
        stage = OpticalStage()
        assert stage.n_strips is None
        assert stage.effective_n_strips() is None

    def test_the_palace_route_falls_back_to_a_strip_count(self):
        stage = OpticalStage()(route="palace")
        assert stage.effective_n_strips() == DEFAULT_PALACE_STRIPS

    def test_a_configured_count_wins_on_either_route(self):
        assert OpticalStage()(n_strips=7).effective_n_strips() == 7
        assert OpticalStage()(route="palace", n_strips=7).effective_n_strips() == 7

    def test_the_continuous_stage_builds_no_staircase(self, biased):
        point = biased.carriers.run().points[0]
        with pytest.raises(ValueError, match="n_strips"):
            biased.optical.staircase(point)


class TestOpticalStaircase:
    @pytest.fixture
    def staircase(self, biased):
        biased.optical(route="palace", n_strips=4)
        return biased.optical.staircase(biased.carriers.run().points[-1])

    def test_it_tiles_the_junction_extent_with_the_asked_for_strips(
        self, biased, staircase
    ):
        span = biased.layout.junction_span
        assert len(staircase.strip_names) == 4
        edges = staircase.strips["edges_um"]
        assert edges[0] == pytest.approx(span.h[0])
        assert edges[-1] == pytest.approx(span.h[1])

    def test_it_draws_no_electrodes(self, staircase):
        """The optical window is a box around the rib; the metal is outside it."""
        assert staircase.electrode_names == ()

    def test_its_strips_carry_the_carrier_perturbed_permittivity(self, staircase):
        eps = staircase.strips["eps_complex"]
        assert eps.size == 4
        # Free carriers lower the index and add loss (exp(+i omega t)).
        assert np.all(eps.real < OpticalStage().strip_index ** 2)
        assert np.all(eps.imag <= 0.0)

    def test_it_resolves_to_a_meshable_optical_stack(self, staircase):
        stack = staircase.stack("optical")
        assert set(staircase.strip_names) <= set(stack.layers)

    def test_the_strip_span_is_overridable(self, biased):
        biased.optical(route="palace", n_strips=2, strip_span=SLAB)
        staircase = biased.optical.staircase(biased.carriers.run().points[-1])
        edges = staircase.strips["edges_um"]
        assert edges[0] == pytest.approx(SLAB[0])
        assert edges[-1] == pytest.approx(SLAB[1])


class TestMissingRuntime:
    @pytest.fixture
    def no_palace(self, monkeypatch):
        monkeypatch.setattr(
            "gsim.palace.runtime.resolve_palace_binary", lambda **_kw: None
        )

    @pytest.mark.usefixtures("no_palace")
    @pytest.mark.parametrize("stage_name", ["optical", "rf"])
    def test_palace_without_the_binary_is_actionable(self, biased, stage_name):
        getattr(biased, stage_name)(route="palace")
        with pytest.raises(RuntimeError) as excinfo:
            getattr(biased, stage_name).run()
        message = str(excinfo.value)
        assert "PALACE_BIN" in message
        assert f"study.{stage_name}(route='femwell')" in message

    @pytest.mark.usefixtures("no_palace")
    @pytest.mark.parametrize("stage_name", ["optical", "rf"])
    def test_the_route_is_checked_before_anything_is_meshed(
        self, study, monkeypatch, stage_name
    ):
        """A user whose Route cannot run pays for no mesh and no charge solve."""

        def fail(*_args, **_kwargs):
            raise AssertionError("the charge stage must not run")

        monkeypatch.setattr("gsim.modulator.charge.ChargeStage._solve", fail)
        getattr(study, stage_name)(route="palace")
        with pytest.raises(RuntimeError, match="PALACE_BIN"):
            getattr(study, stage_name).run()

    @pytest.mark.parametrize("stage_name", ["optical", "rf"])
    def test_femwell_still_names_its_extra(self, biased, monkeypatch, stage_name):
        monkeypatch.setitem(sys.modules, "femwell", None)
        with pytest.raises(ImportError, match=r"gsim\[femwell\]"):
            getattr(biased, stage_name).run()
