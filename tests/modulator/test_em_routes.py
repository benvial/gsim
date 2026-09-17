"""Route selection on the two EM Stages, without any solver runtime.

Both EM Stages answer the same question through either Backend, and the
choice is a Stage setting. What is hermetic about that choice — the
default, the values accepted, the registry a route is found in, what
a strip count means on each Stage, the Staircase the optical Stage
builds when it is routed to Palace, and the error a user selecting a
Route they cannot run gets — is under test here, as is everything the
Palace route does around a Palace run without the binary: the crash
salvage, the abort report, the impedance paths and the readings off
Palace's tables. The Routes actually agreeing on a number is the
runtime-gated ``test_palace_route_runtime.py``.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from pydantic import ValidationError

from gsim.common.modes import Conductor
from gsim.modulator import DEFAULT_PALACE_STRIPS, OpticalStage, RFStage
from gsim.palace.results import PalaceTextResults

from .conftest import SLAB


class TestRouteRegistry:
    def test_each_name_resolves_to_its_adapter(self):
        from gsim.modulator import FemwellRoute, PalaceRoute
        from gsim.modulator.route import route_for

        assert isinstance(route_for("femwell"), FemwellRoute)
        assert isinstance(route_for("palace"), PalaceRoute)

    def test_every_run_gets_a_fresh_adapter(self):
        from gsim.modulator.route import route_for

        assert route_for("palace") is not route_for("palace")

    def test_an_unregistered_name_is_reported(self):
        from gsim.modulator.route import route_for

        with pytest.raises(ValueError, match="comsol"):
            route_for("comsol")  # type: ignore[arg-type]

    def test_a_registered_fake_is_what_the_stage_gets(self, biased, fake_route):
        biased.rf(route="palace")
        assert isinstance(biased.rf.resolved_route(), fake_route)

    def test_the_adapters_say_what_they_can_express(self):
        from gsim.modulator import FemwellRoute, PalaceRoute

        assert FemwellRoute.continuous_materials is True
        assert PalaceRoute.continuous_materials is False
        assert FemwellRoute.conductor_model == "volume"
        assert PalaceRoute.conductor_model == "pec"


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
        biased.rf.seed(object())
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

    def test_it_tiles_the_doped_slab_with_the_asked_for_strips(self, biased, staircase):
        span = biased.layout.doped_span
        assert len(staircase.strip_names) == 4
        edges = staircase.strips.edges_um
        assert edges[0] == pytest.approx(span[0])
        assert edges[-1] == pytest.approx(span[1])

    def test_it_invents_no_flanking_electrodes(self, staircase):
        """The drawn metal arrives as a surrounding region, not as a flank."""
        assert staircase.electrode_names == ()
        assert "cathode_metal" in {region.name for region in staircase.surroundings}

    def test_it_redraws_the_device_around_the_strips(self, biased, staircase):
        """The staircase is the drawn guide with its doped silicon binned."""
        drawn = {rect.layer_name for rect in biased.section}
        redrawn = {region.name.rsplit("_", 1)[0] for region in staircase.surroundings}
        # The doped regions are what the strips replace; everything else
        # the plane crosses is redrawn beside them.
        assert "slab90" in redrawn
        assert not drawn & {"n_rib", "p_rib"} & redrawn

    def test_its_strips_carry_the_carrier_perturbed_permittivity(
        self, biased, staircase
    ):
        eps = staircase.strips.permittivity
        assert eps.size == 4
        # Free carriers lower the index and add loss (exp(+i omega t)).
        assert np.all(eps.real < biased.optical.unperturbed_index() ** 2)
        assert np.all(eps.imag <= 0.0)

    def test_it_resolves_to_a_meshable_optical_stack(self, staircase):
        stack = staircase.stack()
        assert set(staircase.strip_names) <= set(stack.layers)

    def test_the_strip_span_is_overridable(self, biased):
        biased.optical(route="palace", n_strips=2, strip_span=SLAB)
        staircase = biased.optical.staircase(biased.carriers.run().points[-1])
        edges = staircase.strips.edges_um
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

        from gsim.modulator.palace_route import PalaceMode, _check_field_is_the_mode

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_field_is_the_mode(
                self._field(n_from_fields=2.4),
                PalaceMode(n_eff=complex(2.5, -0.01), mode_id=2),
                stage_name="rf",
            )

    def test_fields_from_another_mode_are_reported(self):
        from gsim.modulator.palace_route import PalaceMode, _check_field_is_the_mode

        with pytest.warns(UserWarning, match="different mode's fields"):
            _check_field_is_the_mode(
                self._field(n_from_fields=0.02),
                PalaceMode(n_eff=complex(2.5, -0.01), mode_id=2),
                stage_name="rf",
            )

    def test_a_mode_carrying_no_field_says_nothing(self):
        """Nothing to compare is not the same as a mismatch."""
        import warnings

        from gsim.modulator.palace_route import PalaceMode, _check_field_is_the_mode

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


def mode_table(n_modes: int) -> PalaceTextResults:
    """A run's results carrying a complete ``mode-kn.csv`` of *n_modes* Modes."""
    rows = [
        {
            "m": str(m),
            "Re{kn} (1/m)": f"{4.2e7 + m:.6e}",
            "Im{kn} (1/m)": "-1.0e2",
            "Re{n_eff}": f"{2.0 + 0.1 * m:.6e}",
            "Im{n_eff}": "-1.0e-5",
        }
        for m in range(1, n_modes + 1)
    ]
    return PalaceTextResults(
        files={}, csv_tables={"mode-kn.csv": rows}, json_data={}, text_data={}
    )


class FakeSim:
    """A boundary-mode simulation whose run is scripted.

    ``run_local`` leaves whatever *leaves* says on the run — the results
    the sim then reads back — and raises *raises* if given, the way a
    crashed or aborted Palace does after (or before) writing its tables.
    """

    def __init__(self, output_dir, *, leaves=None, raises=None, stale=None):
        self.output_dir = output_dir
        self._leaves = leaves
        self._raises = raises
        self._results = stale
        self.solved = []

    def set_boundary_mode(self, **kwargs):
        self.solved.append(kwargs)

    def write_config(self, **_kwargs):
        pass

    def run_local(self, **_kwargs):
        # The real sim clears its run directory before running.
        self._results = self._leaves
        if self._raises is not None:
            raise self._raises
        return self._results

    def read_results(self):
        return self._results

    @property
    def last_run_files(self):
        return {"mode-kn.csv": self.output_dir / "mode-kn.csv"} if self._results else {}

    def read_mode_field(self, mode_id):
        raise FileNotFoundError(f"no saved fields for mode {mode_id}")


class TestCrashedRunSalvage:
    """Palace 0.17 can corrupt its heap on shutdown, after answering.

    A run that exits abnormally with its complete mode table on disk is
    an answer, not a failure; a truncated or absent table stays one.
    """

    def test_a_complete_table_is_used_and_the_crash_reported(self, tmp_path):
        from gsim.modulator.palace_route import _salvage_mode_table

        sim = FakeSim(tmp_path, stale=mode_table(4))
        with pytest.warns(UserWarning, match="exited abnormally"):
            text = _salvage_mode_table(
                sim,
                RuntimeError("free(): corrupted unsorted chunks"),
                freq_hz=10e9,
                num_modes=4,
            )
        assert text is not None
        assert len(text.modes) == 4
        assert text.modes[1]["n_eff"].real == pytest.approx(2.1)

    def test_a_truncated_table_is_not_an_answer(self, tmp_path):
        from gsim.modulator.palace_route import _salvage_mode_table

        sim = FakeSim(tmp_path, stale=mode_table(2))
        assert (
            _salvage_mode_table(sim, RuntimeError("boom"), freq_hz=10e9, num_modes=4)
            is None
        )

    def test_no_output_at_all_is_not_an_answer(self, tmp_path):
        from gsim.modulator.palace_route import _salvage_mode_table

        sim = FakeSim(tmp_path)
        assert (
            _salvage_mode_table(sim, RuntimeError("boom"), freq_hz=10e9, num_modes=4)
            is None
        )


class TestSolvingThroughACrash:
    """What ``solve_palace_modes`` does around a run that exits abnormally."""

    def _solve(self, sim, num_modes: int = 4):
        from gsim.modulator.palace_route import solve_palace_modes

        return solve_palace_modes(
            sim, freq_hz=10e9, num_modes=num_modes, target=2.0, save=1, binary="palace"
        )

    def test_a_crash_that_still_answered_is_an_answer(self, tmp_path):
        crash = RuntimeError("free(): corrupted unsorted chunks")
        with pytest.warns(UserWarning, match="exited abnormally"):
            solve = self._solve(FakeSim(tmp_path, leaves=mode_table(4), raises=crash))

        assert [mode.mode_id for mode in solve.modes] == [1, 2, 3, 4]
        assert solve.results.modes[4]["n_eff"].real == pytest.approx(2.4)

    def test_a_crash_that_answered_nothing_is_raised(self, tmp_path):
        crash = RuntimeError("free(): corrupted unsorted chunks")
        with pytest.raises(RuntimeError, match="corrupted unsorted chunks"):
            self._solve(FakeSim(tmp_path, raises=crash))

    def test_a_previous_runs_table_is_not_salvaged_as_this_ones(self, tmp_path):
        """The sim clears its run before running, so a stale table is gone."""
        crash = RuntimeError("free(): corrupted unsorted chunks")
        with pytest.raises(RuntimeError, match="corrupted unsorted chunks"):
            self._solve(FakeSim(tmp_path, stale=mode_table(4), raises=crash))

    def test_a_clean_run_hands_back_its_modes_and_results(self, tmp_path):
        sim = FakeSim(tmp_path, leaves=mode_table(2))
        solve = self._solve(sim, num_modes=2)
        assert [mode.n_eff.real for mode in solve.modes] == pytest.approx([2.1, 2.2])
        assert solve.results is sim.read_results()
        assert sim.solved[-1]["save"] == 1


class TestAbortedBinaryIsReported:
    """A Palace binary that aborts is reported as a runtime failure.

    A broken Palace runtime — typically a bundled MPI that cannot start —
    kills the binary before it writes any solver output, and the raw
    ``CalledProcessError`` that surfaces carries an exit status and
    nothing a user can act on. The Route turns that into a report naming
    the binary that ran and saying that an abort with no solver output
    means the runtime rather than the model.
    """

    @staticmethod
    def _abort(returncode: int, stderr: str = "") -> subprocess.CalledProcessError:
        return subprocess.CalledProcessError(
            returncode,
            ["/opt/somewhere/palace", "-np", "1", "config.json"],
            output="",
            stderr=stderr,
        )

    def _solve(self, sim):
        from gsim.modulator.palace_route import solve_palace_modes

        return solve_palace_modes(
            sim,
            freq_hz=10e9,
            num_modes=4,
            target=2.0,
            binary="/opt/somewhere/palace",
        )

    def test_the_report_names_the_binary_and_blames_the_runtime(self, tmp_path):
        with pytest.raises(RuntimeError) as excinfo:
            self._solve(FakeSim(tmp_path, raises=self._abort(134)))
        message = str(excinfo.value)
        assert "/opt/somewhere/palace" in message
        assert "exit status 134" in message
        assert "SIGABRT" in message
        assert "any solver output" in message
        assert "runtime" in message
        assert "PALACE_BIN" in message
        assert "route='femwell'" in message

    def test_a_plain_exit_is_not_blamed_on_the_runtime(self, tmp_path):
        """Exit 1 is Palace refusing the run itself; its stderr says why."""
        with pytest.raises(RuntimeError) as excinfo:
            self._solve(
                FakeSim(tmp_path, raises=self._abort(1, "Invalid configuration\n"))
            )
        message = str(excinfo.value)
        assert "exit status 1." in message
        assert "runtime" not in message.split("Point PALACE_BIN", maxsplit=1)[0]
        assert "Invalid configuration" in message

    def test_the_raw_error_is_chained_not_lost(self, tmp_path):
        with pytest.raises(RuntimeError) as excinfo:
            self._solve(FakeSim(tmp_path, raises=self._abort(134)))
        assert isinstance(excinfo.value.__cause__, subprocess.CalledProcessError)
        assert excinfo.value.__cause__.returncode == 134

    def test_a_segfault_is_named_as_one(self, tmp_path):
        with pytest.raises(RuntimeError, match="SIGSEGV"):
            self._solve(FakeSim(tmp_path, raises=self._abort(139)))

    def test_the_last_worded_stderr_line_is_quoted(self, tmp_path):
        """MPI ends its error blocks with a dashed rule; quote past it."""
        stderr = "noise\nopal_shmem_base_select failed\n" + "-" * 40 + "\n"
        with pytest.raises(RuntimeError, match="opal_shmem_base_select failed"):
            self._solve(FakeSim(tmp_path, raises=self._abort(134, stderr)))

    def test_partial_output_is_not_blamed_on_the_runtime(self, tmp_path):
        """A truncated table means the solver ran; the runtime did start."""
        with pytest.raises(RuntimeError) as excinfo:
            self._solve(
                FakeSim(tmp_path, leaves=mode_table(2), raises=self._abort(134))
            )
        message = str(excinfo.value)
        assert "any solver output" not in message
        assert "partial solver output" in message
        assert "exit status 134" in message
        assert str(tmp_path) in message


#: A signal electrode, a return electrode and the meshed domain around
#: them, for sizing the Palace Route's impedance paths (um).
SIGNAL = ((-22.6, -20.6), (0.0, 0.5))
GROUND = ((-19.4, -17.4), (0.0, 0.5))
DOMAIN = ((-25.0, -15.0), (-3.0, 2.0))


def impedance_paths(**overrides):
    """The paths sized for the electrodes above, with any of them replaced."""
    from gsim.modulator.palace_route import line_impedance_paths

    kwargs = {"signal": SIGNAL, "ground": GROUND, "domain": DOMAIN}
    kwargs.update(overrides)
    return line_impedance_paths(**kwargs)


class TestImpedancePaths:
    """The postprocessing paths the RF Stage's Palace Route declares.

    Pure geometry: a signal electrode, a return electrode and the meshed
    domain in, a voltage path and a current loop out. Palace integrates
    E along the first and H around the second, so the first must run
    from one conductor face to the other and the second must enclose the
    signal conductor and nothing else.
    """

    def test_the_voltage_path_crosses_the_gap_face_to_face(self):
        paths = impedance_paths()

        (h0, v0), (h1, v1) = paths.voltage
        assert v0 == v1 == pytest.approx(0.25)
        # From the signal's inner face towards the return's inner face,
        # each end a hair inside the gap rather than on the conductor.
        assert -20.6 < h0 < -20.5
        assert -19.5 < h1 < -19.4
        assert h1 - h0 == pytest.approx(1.2, rel=1e-2)

    def test_each_end_sits_at_its_own_electrodes_mid_height(self):
        """A return on another metal level is still met on its face."""
        paths = impedance_paths(ground=((-19.4, -17.4), (1.0, 1.5)))

        (_, v0), (_, v1) = paths.voltage
        assert v0 == pytest.approx(0.25)
        assert v1 == pytest.approx(1.25)

    def test_a_return_on_the_low_side_is_crossed_the_other_way(self):
        paths = impedance_paths(signal=GROUND, ground=SIGNAL)

        (h0, _), (h1, _) = paths.voltage
        assert -19.4 > h0 > -19.5
        assert -20.6 < h1 < -20.5

    def test_the_current_loop_hugs_the_signal_and_only_the_signal(self):
        paths = impedance_paths()

        h = [p[0] for p in paths.current]
        v = [p[1] for p in paths.current]
        (s0, s1), (t0, t1) = SIGNAL
        # Around the conductor: every corner outside its rectangle...
        assert min(h) < s0
        assert max(h) > s1
        assert min(v) < t0
        assert max(v) > t1
        # ...but well clear of the return and of the domain wall.
        assert max(h) < GROUND[0][0]
        assert min(h) > DOMAIN[0][0]
        assert min(v) > DOMAIN[1][0]
        assert max(v) < DOMAIN[1][1]
        # And tight: the loop's clearance is a fraction of the gap.
        assert max(h) - s1 < 0.01 * (GROUND[0][0] - s1)

    def test_the_loop_is_closed_by_palace_not_by_repeating_a_point(self):
        """Palace joins the last point back to the first itself."""
        paths = impedance_paths()

        assert len(paths.current) == 4
        assert paths.current[0] != paths.current[-1]

    def test_the_loop_stays_inside_a_tight_domain(self):
        """A wall closer than the gap sets the clearance, not the gap."""
        paths = impedance_paths(domain=((-22.601, -15.0), (-0.001, 2.0)))

        assert min(p[0] for p in paths.current) > -22.601
        assert min(p[1] for p in paths.current) > -0.001

    def test_touching_electrodes_are_refused(self):
        with pytest.raises(ValueError, match="no gap"):
            impedance_paths(ground=((-20.6, -18.6), (0.0, 0.5)))

    def test_electrodes_stacked_over_each_other_are_refused(self):
        with pytest.raises(ValueError, match="no gap"):
            impedance_paths(ground=((-22.0, -20.0), (1.0, 1.5)))

    def test_a_signal_the_window_clips_is_refused(self):
        """An electrode on or over the wall has no outline to loop around."""
        with pytest.raises(ValueError, match="not inside the meshed domain"):
            impedance_paths(domain=((-22.6, -15.0), (-3.0, 2.0)))


class TestDeclaringThePaths:
    def test_the_paths_become_a_mode_path_of_the_sim(self, biased):
        from gsim.modulator.palace_route import declare_impedance_paths

        sim = biased.rf.simulation()
        paths = impedance_paths()

        index = declare_impedance_paths(sim, paths)

        assert index == 1
        (path,) = sim.mode_paths
        assert path.voltage_path == [list(p) for p in paths.voltage]
        assert path.current_path == [list(p) for p in paths.current]
        # Two-dimensional points: cross-section coordinates, not layout.
        assert all(len(p) == 2 for p in path.voltage_path)

    def test_declaring_twice_replaces_rather_than_stacks(self, biased):
        from gsim.modulator.palace_route import declare_impedance_paths

        sim = biased.rf.simulation()

        first = declare_impedance_paths(sim, impedance_paths())
        second = declare_impedance_paths(sim, impedance_paths())

        assert len(sim.mode_paths) == 1
        assert first == second == 1

    def test_the_index_is_where_the_sim_put_it(self, biased):
        """A path declared after another is read under its own index."""
        from gsim.modulator.palace_route import declare_impedance_paths

        sim = biased.rf.simulation()
        sim.add_impedance_path("probe", voltage=[[-20.0, 0.1], [-20.0, 0.2]])

        assert declare_impedance_paths(sim, impedance_paths()) == 2


#: ``mode -> (Z_PV, Z_VI)`` of the canned ``mode-Z.csv``.
NATIVE_TABLE = {1: (100.0, 80.0), 2: (200.0, 120.0)}

#: The signal electrode of the geometry above, as a current integral names it.
SIGNAL_CONDUCTOR = Conductor(name="electrode_low", extent=SIGNAL, model="pec")


def impedance_tables(rows=None, *, z_vi: bool = True, index: int = 1):
    """A run's results carrying a ``mode-Z.csv`` under postprocessing *index*."""
    rows = NATIVE_TABLE if rows is None else rows
    table = []
    for m, (z_pv, z_vi_) in rows.items():
        row = {
            "m": str(m),
            f"Z_PV[{index}] (Ohm)": str(z_pv),
            f"L_PV[{index}] (H/m)": "1e-7",
            f"C_PV[{index}] (F/m)": "1e-10",
        }
        if z_vi:
            row[f"Z_VI[{index}] (Ohm)"] = str(z_vi_)
            row[f"L_VI[{index}] (H/m)"] = "1e-7"
            row[f"C_VI[{index}] (F/m)"] = "1e-10"
        table.append(row)
    return PalaceTextResults(
        files={}, csv_tables={"mode-Z.csv": table}, json_data={}, text_data={}
    )


class TestNativeImpedance:
    """Reading the selected Mode's impedance off Palace's own tables."""

    def test_it_is_the_power_current_impedance_of_the_selected_mode(self):
        """``Z_VI^2 / Z_PV`` is ``2P/|I|^2``: the definition both Routes use."""
        from gsim.modulator.palace_route import PalaceMode, native_line_impedance

        reading = native_line_impedance(
            impedance_tables(), PalaceMode(2.1 - 1e-5j, 2), index=1
        )

        assert reading is not None
        assert reading.z0_ohm == pytest.approx(120.0**2 / 200.0)
        assert reading.n_eff == 2.1 - 1e-5j
        assert reading.wall_mode is False

    def test_it_reads_under_the_index_the_path_was_declared_at(self):
        from gsim.modulator.palace_route import PalaceMode, native_line_impedance

        results = impedance_tables(index=3)
        assert native_line_impedance(results, PalaceMode(2.0, 1), index=1) is None
        reading = native_line_impedance(results, PalaceMode(2.0, 1), index=3)
        assert reading is not None
        assert reading.z0_ohm == pytest.approx(80.0**2 / 100.0)

    def test_a_gap_carrying_no_voltage_is_reported(self):
        """``|V| |I| << 2P``: the electrodes sit at one potential."""
        from gsim.modulator.palace_route import PalaceMode, native_line_impedance

        reading = native_line_impedance(
            impedance_tables({1: (2.5e-8, 2.1e-3)}),
            PalaceMode(2.03 - 6e-7j, 1),
            index=1,
        )

        assert reading is not None
        assert reading.wall_mode is True
        assert "between them and the window wall" in reading.diagnostic
        # The impedance itself is still the power-current one.
        assert reading.z0_ohm == pytest.approx(2.1e-3**2 / 2.5e-8)

    def test_a_lossy_line_mode_is_not_reported(self):
        """``Z_PV / Z_PI`` of 0.4 is a lossy line, not a wall mode."""
        from gsim.modulator.palace_route import PalaceMode, native_line_impedance

        reading = native_line_impedance(
            impedance_tables({1: (1.63, 2.62)}), PalaceMode(31.9 - 31.9j, 1), index=1
        )
        assert reading is not None
        assert reading.wall_mode is False

    def test_a_loop_enclosing_no_current_is_no_answer(self):
        """``Z_VI = 0`` would make the impedance zero, which is no reading."""
        from gsim.modulator.palace_route import PalaceMode, native_line_impedance

        assert (
            native_line_impedance(
                impedance_tables({1: (100.0, 0.0)}),
                PalaceMode(2.0, 1),
                index=1,
            )
            is None
        )

    def test_no_table_is_no_answer(self):
        from gsim.modulator.palace_route import PalaceMode, native_line_impedance

        assert native_line_impedance(mode_table(2), PalaceMode(2.0, 1), index=1) is None

    def test_a_table_without_the_current_is_no_answer(self):
        """``Z_PV`` alone is a different definition, not a fallback."""
        from gsim.modulator.palace_route import PalaceMode, native_line_impedance

        assert (
            native_line_impedance(
                impedance_tables(z_vi=False),
                PalaceMode(2.0, 1),
                index=1,
            )
            is None
        )

    def test_a_mode_the_table_does_not_carry_is_no_answer(self):
        from gsim.modulator.palace_route import PalaceMode, native_line_impedance

        assert (
            native_line_impedance(impedance_tables(), PalaceMode(2.0, 3), index=1)
            is None
        )

    def test_the_route_prefers_the_native_value(self, tmp_path):
        """With the table in the run's results, no field file is read."""
        import warnings

        from gsim.modulator.palace_route import (
            PalaceMode,
            PalaceSolve,
            palace_line_impedance,
        )

        mode = PalaceMode(2.0, 1)
        solve = PalaceSolve(modes=[mode], results=impedance_tables())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reading = palace_line_impedance(
                FakeSim(tmp_path),
                solve,
                mode,
                index=1,
                signal=SIGNAL_CONDUCTOR,
                stage_name="rf",
            )

        assert reading.z0_ohm == pytest.approx(80.0**2 / 100.0)
        assert reading.z0_ohm.imag == 0.0
        assert reading.wall_mode is False

    def test_without_a_declared_path_the_fields_are_read_and_their_absence_reported(
        self, tmp_path
    ):
        """The fallback and its NaN contract stay."""
        from gsim.modulator.palace_route import (
            PalaceMode,
            PalaceSolve,
            palace_line_impedance,
        )

        mode = PalaceMode(2.0, 1)
        solve = PalaceSolve(modes=[mode], results=impedance_tables())
        with pytest.warns(UserWarning, match="could not read mode 1's saved fields"):
            reading = palace_line_impedance(
                FakeSim(tmp_path),
                solve,
                mode,
                index=None,
                signal=SIGNAL_CONDUCTOR,
                stage_name="rf",
            )

        assert np.isnan(reading.z0_ohm.real)
        assert reading.wall_mode is None


class TestPalaceRoutePreparesTheLine:
    """The Palace route sizes the paths from the electrodes, not by hand."""

    def _meshed(self, biased, **settings):
        biased.rf(route="palace", frequencies_hz=[10e9], n_strips=3, **settings)
        staircase = biased.rf.staircase()
        sim = biased.rf.simulation(staircase)
        sim.mesh(**biased.rf.mesh)
        return sim, staircase

    def test_the_paths_are_declared_on_the_meshed_simulation(self, biased):
        from gsim.modulator import PalaceRoute

        sim, staircase = self._meshed(biased)
        signal, return_ = biased.rf.line_conductors(staircase)
        route = PalaceRoute()

        route.prepare_line(sim, signal=signal, return_=return_, stage_name="rf")

        assert route.impedance_index == 1
        (path,) = sim.mode_paths
        (h_lo, h_hi), (v_lo, v_hi) = signal.extent
        # The voltage path leaves the signal electrode's inner face at
        # its mid-height; the loop surrounds that electrode.
        assert path.voltage_path[0][1] == pytest.approx(0.5 * (v_lo + v_hi))
        loop_h = [p[0] for p in path.current_path]
        assert min(loop_h) < h_lo
        assert max(loop_h) > h_hi

    def test_a_signal_the_window_clips_leaves_the_reading_to_the_fields(self, biased):
        """A Window clipping an electrode is no reason to refuse the solve."""
        from gsim.modulator import PalaceRoute

        sim, staircase = self._meshed(biased, conductor_model="volume")
        signal, return_ = biased.rf.line_conductors(staircase)
        # Pretend the mesh stops short of the signal electrode's outer face.
        biased.rf(window=(signal.extent[0][0] + 0.5, signal.extent[0][1] + 10.0))
        sim = biased.rf.simulation(staircase)
        sim.mesh(**biased.rf.mesh)
        route = PalaceRoute()

        with pytest.warns(UserWarning, match="read off the saved fields instead"):
            route.prepare_line(sim, signal=signal, return_=return_, stage_name="rf")

        assert route.impedance_index is None
        assert sim.mode_paths == []

    def test_a_line_without_a_single_return_is_read_off_the_fields(self, biased):
        from gsim.modulator import PalaceRoute

        sim, staircase = self._meshed(biased)
        signal, _ = biased.rf.line_conductors(staircase)
        route = PalaceRoute()

        with pytest.warns(UserWarning, match="no single return electrode"):
            route.prepare_line(sim, signal=signal, return_=None, stage_name="rf")

        assert route.impedance_index is None


class TestFemwellRouteReadsTheWallOffTheSimulation:
    """The metallic-boundary flag has one channel: the simulation."""

    @pytest.mark.parametrize("wall", [True, False])
    def test_the_solve_uses_the_simulation_flag(self, monkeypatch, wall):
        from types import SimpleNamespace

        from gsim.modulator import FemwellRoute

        seen = {}

        def fake_solve_modes(_mesh_path, **kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setattr("gsim.femwell.adapter.solve_modes", fake_solve_modes)
        sim = SimpleNamespace(mesh_path="mesh.msh", metallic_boundaries=wall)

        FemwellRoute().solve(
            sim,
            freq_hz=1e9,
            num_modes=1,
            target=None,
            order=1,
            verbose=False,
            stage_name="rf",
            epsilon=np.ones(1),
        )

        assert seen["metallic_boundaries"] is wall
