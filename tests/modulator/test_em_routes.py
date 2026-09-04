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

from .conftest import SLAB


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


class TestSavedFieldsAreTheSelectedMode:
    """Which ParaView cycle holds which Mode is a convention, so it is checked."""

    def _field(self, *, n_from_fields: float):
        from gsim.palace.mode_fields import BoundaryModeField

        eta0 = 376.730313668
        e_t = np.tile([1.0 + 0j, 0.0 + 0j], (6, 1))
        h_t = (n_from_fields / eta0) * np.stack([-e_t[:, 1], e_t[:, 0]], axis=1)
        return BoundaryModeField(
            points_um=np.zeros((6, 2)),
            cells=np.arange(6).reshape(1, 6),
            attribute=np.array([1]),
            e_t=e_t,
            e_n=np.zeros(6, dtype=complex),
            h_t=h_t,
            h_n=None,
        )

    def test_fields_matching_the_mode_table_pass_quietly(self):
        import warnings

        from gsim.modulator.route import PalaceMode, _check_field_is_the_mode

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_field_is_the_mode(
                self._field(n_from_fields=2.4),
                PalaceMode(n_eff=complex(2.5, -0.01), mode_id=2),
                stage_name="rf",
            )

    def test_fields_from_another_mode_are_reported(self):
        from gsim.modulator.route import PalaceMode, _check_field_is_the_mode

        with pytest.warns(UserWarning, match="different mode's fields"):
            _check_field_is_the_mode(
                self._field(n_from_fields=0.02),
                PalaceMode(n_eff=complex(2.5, -0.01), mode_id=2),
                stage_name="rf",
            )

    def test_a_mode_carrying_no_field_says_nothing(self):
        """Nothing to compare is not the same as a mismatch."""
        import warnings

        from gsim.modulator.route import PalaceMode, _check_field_is_the_mode

        field = self._field(n_from_fields=2.4)
        field = type(field)(
            points_um=field.points_um,
            cells=field.cells,
            attribute=field.attribute,
            e_t=np.zeros_like(field.e_t),
            e_n=field.e_n,
            h_t=field.h_t,
            h_n=None,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_field_is_the_mode(
                field, PalaceMode(n_eff=complex(2.5), mode_id=1), stage_name="rf"
            )


class TestCrashedRunSalvage:
    """Palace 0.17 can corrupt its heap on shutdown, after answering.

    A run that exits abnormally with its complete mode table on disk is
    an answer, not a failure; a truncated or absent table stays one.
    """

    class _Sim:
        def __init__(self, output_dir):
            self.output_dir = output_dir

    @staticmethod
    def _write_mode_table(output_dir, n_modes: int) -> None:
        palace_dir = output_dir / "output" / "palace"
        palace_dir.mkdir(parents=True)
        rows = ["m, Re{kn} (1/m), Im{kn} (1/m), Re{n_eff}, Im{n_eff}"]
        rows += [
            f"{m}, {4.2e7 + m:.6e}, -1.0e2, {2.0 + 0.1 * m:.6e}, -1.0e-5"
            for m in range(1, n_modes + 1)
        ]
        (palace_dir / "mode-kn.csv").write_text("\n".join(rows) + "\n")

    def test_a_complete_table_is_used_and_the_crash_reported(self, tmp_path):
        from gsim.modulator.route import _salvage_mode_table

        self._write_mode_table(tmp_path, 4)
        with pytest.warns(UserWarning, match="exited abnormally"):
            text = _salvage_mode_table(
                self._Sim(tmp_path),
                RuntimeError("free(): corrupted unsorted chunks"),
                freq_hz=10e9,
                num_modes=4,
            )
        assert text is not None
        assert len(text.modes) == 4
        assert text.modes[1]["n_eff"].real == pytest.approx(2.1)

    def test_a_truncated_table_is_not_an_answer(self, tmp_path):
        from gsim.modulator.route import _salvage_mode_table

        self._write_mode_table(tmp_path, 2)
        assert (
            _salvage_mode_table(
                self._Sim(tmp_path), RuntimeError("boom"), freq_hz=10e9, num_modes=4
            )
            is None
        )

    def test_no_output_at_all_is_not_an_answer(self, tmp_path):
        from gsim.modulator.route import _salvage_mode_table

        assert (
            _salvage_mode_table(
                self._Sim(tmp_path), RuntimeError("boom"), freq_hz=10e9, num_modes=4
            )
            is None
        )


class TestSolvingThroughACrash:
    """What ``solve_palace_modes`` does around a run that exits abnormally."""

    class _CrashingSim:
        """A sim whose run writes a mode table and then fails."""

        def __init__(self, output_dir, n_modes: int | None):
            self.output_dir = output_dir
            self._n_modes = n_modes

        def set_boundary_mode(self, **_kwargs):
            pass

        def write_config(self, **_kwargs):
            pass

        def run_local(self, **_kwargs):
            if self._n_modes is not None:
                TestCrashedRunSalvage._write_mode_table(self.output_dir, self._n_modes)
            raise RuntimeError("free(): corrupted unsorted chunks")

    def _solve(self, sim, num_modes: int = 4):
        from gsim.modulator.route import solve_palace_modes

        return solve_palace_modes(
            sim, freq_hz=10e9, num_modes=num_modes, target=2.0, save=1, binary="palace"
        )

    def test_a_crash_that_still_answered_is_an_answer(self, tmp_path):
        with pytest.warns(UserWarning, match="exited abnormally"):
            modes = self._solve(self._CrashingSim(tmp_path, 4))

        assert [mode.mode_id for mode in modes] == [1, 2, 3, 4]

    def test_a_crash_that_answered_nothing_is_raised(self, tmp_path):
        with pytest.raises(RuntimeError, match="corrupted unsorted chunks"):
            self._solve(self._CrashingSim(tmp_path, None))

    def test_a_previous_runs_table_is_not_salvaged_as_this_ones(self, tmp_path):
        """The output directory is cleared before the run, not after it."""
        TestCrashedRunSalvage._write_mode_table(tmp_path, 4)

        with pytest.raises(RuntimeError, match="corrupted unsorted chunks"):
            self._solve(self._CrashingSim(tmp_path, None))
