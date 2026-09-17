"""A boundary-mode simulation owns its run directory.

Where Palace's tables land, that the previous run's are gone before the
next one, and that what is read back is this run's — none of which a
caller should have to spell.
"""

from __future__ import annotations

import pytest

from gsim.palace import BoundaryModeSim
from gsim.palace.base import PalaceSimMixin
from gsim.palace.boundarymode import RUN_SUBDIR

MODE_TABLE = (
    "m, Re{kn} (1/m), Im{kn} (1/m), Re{n_eff}, Im{n_eff}\n"
    "1, 4.2e7, -1.0e2, 2.1, -1.0e-5\n"
)


def sim_at(tmp_path) -> BoundaryModeSim:
    sim = BoundaryModeSim()
    sim.set_output_dir(tmp_path)
    return sim


def write_table(directory, name="mode-kn.csv", text=MODE_TABLE) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(text)


class TestRunDirectory:
    def test_it_sits_under_the_output_directory(self, tmp_path):
        assert sim_at(tmp_path).run_dir == tmp_path / RUN_SUBDIR

    def test_without_an_output_directory_it_is_an_error(self):
        with pytest.raises(ValueError, match="set_output_dir"):
            _ = BoundaryModeSim().run_dir

    def test_no_run_leaves_no_files_and_no_results(self, tmp_path):
        sim = sim_at(tmp_path)
        assert sim.last_run_files == {}
        assert sim.read_results() is None

    def test_this_runs_tables_are_read_back(self, tmp_path):
        sim = sim_at(tmp_path)
        write_table(sim.run_dir)

        assert set(sim.last_run_files) == {"mode-kn.csv"}
        results = sim.read_results()
        assert results is not None
        assert results.modes[1]["n_eff"].real == pytest.approx(2.1)

    def test_a_table_left_beside_the_run_is_not_this_runs(self, tmp_path):
        """Only the run directory is read, so nothing else can shadow it."""
        sim = sim_at(tmp_path)
        write_table(tmp_path)

        assert sim.last_run_files == {}
        assert sim.read_results() is None


class TestRunningLocally:
    @pytest.fixture
    def scripted_palace(self, monkeypatch):
        """Stand the mixin's run in for a Palace that writes what it is told."""
        script = {"writes": True}

        def fake_run_local(self, **_kwargs):
            if script["writes"]:
                write_table(self.run_dir)
            return {}

        monkeypatch.setattr(PalaceSimMixin, "run_local", fake_run_local)
        return script

    @pytest.mark.usefixtures("scripted_palace")
    def test_the_previous_runs_tables_are_cleared_first(self, tmp_path):
        sim = sim_at(tmp_path)
        write_table(sim.run_dir, name="mode-Z.csv", text="m, Z_PV[1] (Ohm)\n1, 50\n")

        results = sim.run_local(palace_executable="palace", verbose=False)

        assert set(sim.last_run_files) == {"mode-kn.csv"}
        assert results.characteristic_impedance(index=1, mode=1) is None
        assert results.modes[1]["n_eff"].real == pytest.approx(2.1)

    def test_a_run_that_leaves_nothing_is_an_error(self, tmp_path, scripted_palace):
        scripted_palace["writes"] = False
        with pytest.raises(RuntimeError, match="no text results"):
            sim_at(tmp_path).run_local(palace_executable="palace", verbose=False)

    @pytest.mark.usefixtures("scripted_palace")
    def test_the_returned_results_are_what_read_results_reads(self, tmp_path):
        sim = sim_at(tmp_path)
        results = sim.run_local(palace_executable="palace", verbose=False)
        again = sim.read_results()
        assert again is not None
        assert again.modes == results.modes
