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

import subprocess
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
        edges = staircase.strips["edges_um"]
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
        eps = staircase.strips["eps_complex"]
        assert eps.size == 4
        # Free carriers lower the index and add loss (exp(+i omega t)).
        assert np.all(eps.real < biased.optical.unperturbed_index() ** 2)
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


class TestAbortedBinaryIsReported:
    """A Palace binary that aborts is reported as a runtime failure.

    A broken Palace runtime — typically a bundled MPI that cannot start —
    kills the binary before it writes any solver output, and the raw
    ``CalledProcessError`` that surfaces carries an exit status and
    nothing a user can act on. The Route turns that into a report naming
    the binary that ran and saying that an abort with no solver output
    means the runtime rather than the model.
    """

    class _AbortingSim:
        """A sim whose binary dies without writing anything."""

        def __init__(self, output_dir, *, returncode: int, stderr: str = ""):
            self.output_dir = output_dir
            self._returncode = returncode
            self._stderr = stderr

        def set_boundary_mode(self, **_kwargs):
            pass

        def write_config(self, **_kwargs):
            pass

        def run_local(self, *, palace_executable, **_kwargs):
            raise subprocess.CalledProcessError(
                self._returncode,
                [str(palace_executable), "-np", "1", "config.json"],
                output="",
                stderr=self._stderr,
            )

    def _solve(self, sim):
        from gsim.modulator.route import solve_palace_modes

        return solve_palace_modes(
            sim,
            freq_hz=10e9,
            num_modes=4,
            target=2.0,
            binary="/opt/somewhere/palace",
        )

    def test_the_report_names_the_binary_and_blames_the_runtime(self, tmp_path):
        with pytest.raises(RuntimeError) as excinfo:
            self._solve(self._AbortingSim(tmp_path, returncode=134))
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
                self._AbortingSim(
                    tmp_path, returncode=1, stderr="Invalid configuration\n"
                )
            )
        message = str(excinfo.value)
        assert "exit status 1." in message
        assert "runtime" not in message.split("Point PALACE_BIN", maxsplit=1)[0]
        assert "Invalid configuration" in message

    def test_the_raw_error_is_chained_not_lost(self, tmp_path):
        with pytest.raises(RuntimeError) as excinfo:
            self._solve(self._AbortingSim(tmp_path, returncode=134))
        assert isinstance(excinfo.value.__cause__, subprocess.CalledProcessError)
        assert excinfo.value.__cause__.returncode == 134

    def test_a_segfault_is_named_as_one(self, tmp_path):
        with pytest.raises(RuntimeError, match="SIGSEGV"):
            self._solve(self._AbortingSim(tmp_path, returncode=139))

    def test_the_last_worded_stderr_line_is_quoted(self, tmp_path):
        """MPI ends its error blocks with a dashed rule; quote past it."""
        with pytest.raises(RuntimeError, match="opal_shmem_base_select failed"):
            self._solve(
                self._AbortingSim(
                    tmp_path,
                    returncode=134,
                    stderr="noise\nopal_shmem_base_select failed\n" + "-" * 40 + "\n",
                )
            )

    def test_partial_output_is_not_blamed_on_the_runtime(self, tmp_path):
        """A truncated table means the solver ran; the runtime did start."""

        class _PartialSim(self._AbortingSim):
            def run_local(self, *, palace_executable, **_kwargs):
                TestCrashedRunSalvage._write_mode_table(self.output_dir, 2)
                super().run_local(palace_executable=palace_executable)

        with pytest.raises(RuntimeError) as excinfo:
            self._solve(_PartialSim(tmp_path, returncode=134))
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
    from gsim.modulator.route import line_impedance_paths

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
    def test_the_paths_become_the_sims_only_postprocessing_port(self, biased):
        from gsim.modulator.route import declare_impedance_paths

        sim = biased.rf.simulation()
        paths = impedance_paths()

        declare_impedance_paths(sim, paths)

        assert len(sim.ports) == 1
        port = sim.ports[0]
        assert port.voltage_path == [list(p) for p in paths.voltage]
        assert port.current_path == [list(p) for p in paths.current]
        # Two-dimensional points: cross-section coordinates, not layout.
        assert all(len(p) == 2 for p in port.voltage_path)

    def test_declaring_twice_replaces_rather_than_stacks(self, biased):
        from gsim.modulator.route import declare_impedance_paths

        sim = biased.rf.simulation()

        declare_impedance_paths(sim, impedance_paths())
        declare_impedance_paths(sim, impedance_paths())

        assert len(sim.ports) == 1

    def test_another_port_on_the_sim_is_refused(self, biased):
        """Entry 1 is where the impedance is read, so nothing may precede it."""
        from gsim.modulator.route import declare_impedance_paths

        sim = biased.rf.simulation()
        sim.add_port("probe", voltage_path=[[-20.0, 0.1], [-20.0, 0.2]])

        with pytest.raises(ValueError, match=r"already carries ports \['probe'\]"):
            declare_impedance_paths(sim, impedance_paths())


#: ``mode -> (Z_PV, Z_VI)`` of the canned ``mode-Z.csv``.
NATIVE_TABLE = {1: (100.0, 80.0), 2: (200.0, 120.0)}


class TestNativeImpedance:
    """Reading the selected Mode's impedance off Palace's own tables."""

    class _Sim:
        def __init__(self, output_dir):
            self.output_dir = output_dir

    @staticmethod
    def _write_tables(
        output_dir, *, rows=None, z_vi: bool = True, where="output/palace"
    ) -> None:
        palace_dir = output_dir / where
        palace_dir.mkdir(parents=True, exist_ok=True)
        rows = NATIVE_TABLE if rows is None else rows
        header = "m, Z_PV[1] (Ohm), L_PV[1] (H/m), C_PV[1] (F/m)"
        lines = [f"{m}, {z_pv}, 1e-7, 1e-10" for m, (z_pv, _) in rows.items()]
        if z_vi:
            header += ", Z_VI[1] (Ohm), L_VI[1] (H/m), C_VI[1] (F/m)"
            lines = [
                f"{m}, {z_pv}, 1e-7, 1e-10, {z_vi_}, 1e-7, 1e-10"
                for m, (z_pv, z_vi_) in rows.items()
            ]
        (palace_dir / "mode-Z.csv").write_text(header + "\n" + "\n".join(lines) + "\n")

    def test_it_is_the_power_current_impedance_of_the_selected_mode(self, tmp_path):
        """``Z_VI^2 / Z_PV`` is ``2P/|I|^2``: the definition both Routes use."""
        from gsim.modulator.route import PalaceMode, native_line_impedance

        self._write_tables(tmp_path)

        z0 = native_line_impedance(
            self._Sim(tmp_path), PalaceMode(2.1 - 1e-5j, 2), stage_name="rf"
        )

        assert z0 == pytest.approx(120.0**2 / 200.0)

    def test_a_gap_carrying_no_voltage_is_reported(self, tmp_path):
        """``|V| |I| << 2P``: the electrodes sit at one potential."""
        from gsim.modulator.route import PalaceMode, native_line_impedance

        self._write_tables(tmp_path, rows={1: (2.5e-8, 2.1e-3)})

        with pytest.warns(UserWarning, match="between them and the window wall"):
            z0 = native_line_impedance(
                self._Sim(tmp_path), PalaceMode(2.03 - 6e-7j, 1), stage_name="rf"
            )

        # The impedance itself is still the power-current one.
        assert z0 == pytest.approx(2.1e-3**2 / 2.5e-8)

    def test_a_lossy_line_mode_is_not_reported(self, tmp_path):
        """``Z_PV / Z_PI`` of 0.4 is a lossy line, not a wall mode."""
        import warnings

        from gsim.modulator.route import PalaceMode, native_line_impedance

        self._write_tables(tmp_path, rows={1: (1.63, 2.62)})

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            native_line_impedance(
                self._Sim(tmp_path), PalaceMode(31.9 - 31.9j, 1), stage_name="rf"
            )

    def test_a_loop_enclosing_no_current_is_no_answer(self, tmp_path):
        """``Z_VI = 0`` would make the impedance zero, which is no reading."""
        from gsim.modulator.route import PalaceMode, native_line_impedance

        self._write_tables(tmp_path, rows={1: (100.0, 0.0)})

        assert (
            native_line_impedance(
                self._Sim(tmp_path), PalaceMode(2.0, 1), stage_name="rf"
            )
            is None
        )

    def test_a_table_left_beside_the_run_is_not_this_runs(self, tmp_path):
        """Only the solver's own directory is cleared per run, so only it is read."""
        from gsim.modulator.route import PalaceMode, native_line_impedance

        self._write_tables(tmp_path, where=".")

        assert (
            native_line_impedance(
                self._Sim(tmp_path), PalaceMode(2.0, 1), stage_name="rf"
            )
            is None
        )

    def test_no_table_is_no_answer(self, tmp_path):
        from gsim.modulator.route import PalaceMode, native_line_impedance

        assert (
            native_line_impedance(
                self._Sim(tmp_path), PalaceMode(2.0, 1), stage_name="rf"
            )
            is None
        )

    def test_a_table_without_the_current_is_no_answer(self, tmp_path):
        """``Z_PV`` alone is a different definition, not a fallback."""
        from gsim.modulator.route import PalaceMode, native_line_impedance

        self._write_tables(tmp_path, z_vi=False)

        assert (
            native_line_impedance(
                self._Sim(tmp_path), PalaceMode(2.0, 1), stage_name="rf"
            )
            is None
        )

    def test_a_mode_the_table_does_not_carry_is_no_answer(self, tmp_path):
        from gsim.modulator.route import PalaceMode, native_line_impedance

        self._write_tables(tmp_path)

        assert (
            native_line_impedance(
                self._Sim(tmp_path), PalaceMode(2.0, 3), stage_name="rf"
            )
            is None
        )

    def test_the_route_prefers_the_native_value(self, tmp_path):
        """With the table on disk, no field file is read and nothing warns."""
        import warnings

        from gsim.modulator.route import PalaceMode, palace_line_impedance

        self._write_tables(tmp_path)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            z0 = palace_line_impedance(
                self._Sim(tmp_path),
                PalaceMode(2.0, 1),
                h_span=(-22.6, -20.6),
                v_span=(0.0, 0.5),
                stage_name="rf",
            )

        assert z0 == pytest.approx(80.0**2 / 100.0)
        assert z0.imag == 0.0

    def test_without_the_table_the_fields_are_read_and_their_absence_reported(
        self, tmp_path
    ):
        """The fallback and its NaN contract stay."""
        from gsim.modulator.route import PalaceMode, palace_line_impedance

        with pytest.warns(UserWarning, match="could not read mode 1's saved fields"):
            z0 = palace_line_impedance(
                self._Sim(tmp_path),
                PalaceMode(2.0, 1),
                h_span=(-22.6, -20.6),
                v_span=(0.0, 0.5),
                stage_name="rf",
            )

        assert np.isnan(z0.real)


class TestTheStageDeclaresThePaths:
    def test_the_palace_route_solves_with_the_paths_declared(self, biased, monkeypatch):
        """The RF Stage sizes the paths from its Staircase, not by hand."""
        from pathlib import Path

        from gsim.modulator import route

        seen: dict[str, list] = {}

        def fake_extent(_sim):
            return ((-30.0, -10.0), (-3.0, 2.0))

        def fake_solve(sim, **_kwargs):
            seen["ports"] = list(sim.ports)
            return [route.PalaceMode(n_eff=2.0 - 1e-3j, mode_id=1)]

        monkeypatch.setattr(route, "mesh_extent", fake_extent)
        monkeypatch.setattr(route, "solve_palace_modes", fake_solve)
        monkeypatch.setattr(
            route, "palace_line_impedance", lambda *_a, **_k: complex(50.0)
        )

        biased.rf(route="palace", frequencies_hz=[10e9], n_strips=3)
        staircase = biased.rf.staircase()
        sim = biased.rf.simulation(staircase)
        with pytest.warns(UserWarning, match="cannot check window containment"):
            n_eff, z0 = biased.rf._solve_palace(sim, staircase, Path("palace"))

        assert n_eff == [2.0 - 1e-3j]
        assert z0 == [50.0]
        (port,) = seen["ports"]
        (h_lo, h_hi), (v_lo, v_hi) = staircase.electrode_extent(
            biased.rf.signal_electrode()
        )
        # The voltage path leaves the signal electrode's inner face at
        # its mid-height; the loop surrounds that electrode.
        assert port.voltage_path[0][1] == pytest.approx(0.5 * (v_lo + v_hi))
        loop_h = [p[0] for p in port.current_path]
        assert min(loop_h) < h_lo
        assert max(loop_h) > h_hi

    def test_paths_that_cannot_be_sized_leave_the_solve_to_the_fields(
        self, biased, monkeypatch
    ):
        """A Window clipping an electrode is no reason to refuse the solve."""
        import warnings
        from pathlib import Path

        from gsim.modulator import route

        seen: dict[str, list] = {}

        def fake_solve(sim, **_kwargs):
            seen["ports"] = list(sim.ports)
            return [route.PalaceMode(n_eff=2.0 - 1e-3j, mode_id=1)]

        # The mesh stops short of the signal electrode's outer face.
        monkeypatch.setattr(
            route, "mesh_extent", lambda _sim: ((-22.0, -10.0), (-3.0, 2.0))
        )
        monkeypatch.setattr(route, "solve_palace_modes", fake_solve)
        monkeypatch.setattr(
            route, "palace_line_impedance", lambda *_a, **_k: complex(50.0)
        )

        biased.rf(route="palace", frequencies_hz=[10e9], n_strips=3)
        staircase = biased.rf.staircase()
        sim = biased.rf.simulation(staircase)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            n_eff, z0 = biased.rf._solve_palace(sim, staircase, Path("palace"))
        assert any(
            "read off the saved fields instead" in str(w.message) for w in caught
        )

        assert n_eff == [2.0 - 1e-3j]
        assert z0 == [50.0]
        assert seen["ports"] == []
