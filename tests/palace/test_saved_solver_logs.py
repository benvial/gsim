"""Retain solver output on success and failure without requiring console output."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gsim.palace import ElectrostaticSim, runtime
from gsim.palace.runtime import PalaceLaunch, run_palace


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("returncode", [0, 7])
def test_saved_log_contains_both_streams(
    tmp_path: Path, verbose: bool, returncode: int
) -> None:
    script = tmp_path / "solver.py"
    script.write_text(
        "import sys\nprint('field solve', flush=True)\n"
        "print('estimator solve', file=sys.stderr, flush=True)\n"
        f"sys.exit({returncode})\n"
    )
    launch = PalaceLaunch((sys.executable, str(script)), dict(os.environ))
    target = tmp_path / "logs" / "solver.log"
    options = dict(cwd=tmp_path, num_processes=1, verbose=verbose, log_path=target)
    if returncode:
        with pytest.raises(subprocess.CalledProcessError) as error:
            run_palace(launch, **options)
        assert error.value.returncode == returncode
        assert "field solve" in error.value.stderr
    else:
        run_palace(launch, **options)
    assert target.read_text() == "field solve\nestimator solve\n"


def test_saved_log_preserves_utf8_solver_output(tmp_path: Path) -> None:
    script = tmp_path / "solver.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.buffer.write('\\u03c9/2\\u03c0 k\\u2099\\n'.encode('utf-8'))\n"
        "sys.stdout.buffer.flush()\n"
    )
    launch = PalaceLaunch((sys.executable, str(script)), dict(os.environ))
    target = tmp_path / "solver.log"
    run_palace(launch, cwd=tmp_path, num_processes=1, verbose=False, log_path=target)

    assert target.read_bytes() == "\u03c9/2\u03c0 k\u2099\n".encode()


def test_unwritable_log_destination_is_not_reported_as_a_missing_executable(
    tmp_path: Path,
) -> None:
    script = tmp_path / "solver.py"
    script.write_text("print('field solve')\n")
    launch = PalaceLaunch((sys.executable, str(script)), dict(os.environ))
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")

    with pytest.raises(OSError) as error:
        run_palace(
            launch,
            cwd=tmp_path,
            num_processes=1,
            verbose=False,
            log_path=blocker / "solver.log",
        )

    assert "Palace executable not found" not in str(error.value)


def test_failure_report_keeps_only_the_output_tail(tmp_path: Path) -> None:
    script = tmp_path / "solver.py"
    script.write_text(
        "import sys\nfor i in range(500):\n    print(f'line {i}')\nsys.exit(3)\n"
    )
    launch = PalaceLaunch((sys.executable, str(script)), dict(os.environ))
    target = tmp_path / "solver.log"

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_palace(
            launch, cwd=tmp_path, num_processes=1, verbose=False, log_path=target
        )

    tail = error.value.stderr.splitlines()
    assert len(tail) == runtime._FAILURE_OUTPUT_TAIL_LINES
    assert tail[-1] == "line 499"
    assert len(target.read_text().splitlines()) == 500


@pytest.mark.parametrize("initial_guess", [None, False, True])
def test_initial_guess_is_optional_and_reaches_the_config(
    initial_guess: bool | None,
) -> None:
    sim = ElectrostaticSim()
    sim.set_solver(initial_guess=initial_guess)
    linear = sim.solver.linear.to_palace_config()
    if initial_guess is None:
        assert "InitialGuess" not in linear
    else:
        assert linear["InitialGuess"] is initial_guess
