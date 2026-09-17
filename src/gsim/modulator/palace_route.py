"""The Palace Route: the second first-class Backend, run as a binary.

Palace's *text* results report Modes as effective indices rather than as
field vectors, so this Route cannot say how much of a Mode sits on the
Window wall or outside the Strips, and says so. What it can do is
integrate the line's voltage and current itself: the Route declares the
paths on the simulation before the solve, reads the characteristic
impedance off Palace's own tables under the index the simulation
assigned, and tells the wall Mode from the line Mode by the voltage the
gap carries against the Mode's power (ADR 0005). Its electrode metal
defaults to a perfect conductor (ADR 0003), because a metal Region takes
Palace's eigenvalue search over.

The Palace binary is resolved here, once per Stage run, and threaded
through nothing.
"""

from __future__ import annotations

import math
import signal
import subprocess
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from gsim.common.modes import Conductor, Extent, LineReading
from gsim.modulator.route import EMRoute, Route

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from gsim.common.stack.staircase import (
        ConductorModel,
        StaircaseCrossSection,
        SurroundingRegion,
    )
    from gsim.palace import BoundaryModeSim
    from gsim.palace.results import PalaceTextResults

__all__ = [
    "FIELD_INDEX_RTOL",
    "IMPEDANCE_PORT",
    "MIN_VOLTAGE_POWER_RATIO",
    "PATH_CLEARANCE_FRACTION",
    "ImpedancePaths",
    "PalaceMode",
    "PalaceRoute",
    "PalaceSolve",
    "conductor_clearance",
    "containment_unmeasurable",
    "declare_impedance_paths",
    "field_line_impedance",
    "line_impedance_paths",
    "mesh_extent",
    "native_line_impedance",
    "palace_line_impedance",
    "require_palace_binary",
    "solve_palace_modes",
]


@dataclass(frozen=True)
class PalaceMode:
    """One Mode of a Palace ``BoundaryMode`` solve.

    Palace's *text* results report Modes as effective indices rather than
    as field vectors, so a Palace Mode carries its index and its
    solver-assigned number and nothing else. What else the solve
    measured is on disk under the same number: the characteristic
    impedance its postprocessing paths integrate (``mode-Z.csv``, read
    by :func:`native_line_impedance`) and, when the solve was asked to
    save them, the fields themselves (ParaView, read by
    :func:`field_line_impedance`). Neither is in hand at selection time,
    so anything a Stage measures while choosing a Mode (the
    Window-containment ratio) stays a femwell-Route capability rather
    than something to invent.

    Attributes:
        n_eff: Complex effective index (``exp(+i omega t)`` convention).
        mode_id: Palace's own mode number, 1-based. This is also the
            ParaView cycle its saved fields land in.
    """

    n_eff: complex
    mode_id: int


@dataclass(frozen=True)
class PalaceSolve:
    """One Palace ``BoundaryMode`` run: its Modes and the tables it wrote.

    Attributes:
        modes: The solved Modes, in Palace's own mode order.
        results: The run's text results, off which a declared impedance
            path's reading is taken.
    """

    modes: list[PalaceMode]
    results: PalaceTextResults


def _palace_hint(stage_name: str) -> str:
    """The error a user selecting the Palace Route without Palace gets."""
    return (
        f"The {stage_name} stage is routed to Palace, but no Palace binary "
        "was found. Point PALACE_BIN at one, put 'palace' on PATH, or go "
        f"back to the default route with study.{stage_name}(route='femwell')."
    )


def require_palace_binary(*, stage_name: str) -> Path:
    """Locate the Palace binary the Palace Route runs.

    Args:
        stage_name: Stage asking, named in the error.

    Returns:
        Path to a runnable Palace executable.

    Raises:
        RuntimeError: When no binary is available, naming the ways to
            provide one and the way back to the femwell Route.
    """
    from gsim.palace.runtime import resolve_palace_binary

    try:
        binary = resolve_palace_binary()
    except Exception as err:  # pragma: no cover - resolver is environment bound
        raise RuntimeError(_palace_hint(stage_name)) from err
    if binary is None:
        raise RuntimeError(_palace_hint(stage_name))
    return binary


def solve_palace_modes(
    sim: BoundaryModeSim,
    *,
    freq_hz: float,
    num_modes: int,
    binary: Path,
    target: float = 0.0,
    save: int = 0,
    verbose: bool = False,
) -> PalaceSolve:
    """Solve one ``BoundaryMode`` problem with Palace on an existing mesh.

    The simulation must already be meshed. Meshing is the Route-neutral
    part of a solve, so a caller comparing the two Routes can mesh once
    and hand the same simulation to both. The simulation owns its run
    directory: it clears the previous run's tables before this one and
    hands back this run's.

    Args:
        sim: A meshed ``BoundaryModeSim``.
        freq_hz: Frequency to solve at (Hz).
        num_modes: Number of Modes to compute.
        target: Effective-index target centring the shift-and-invert
            search; ``0.0`` leaves it to Palace.
        binary: Palace executable, from :func:`require_palace_binary`.
        save: Number of Modes whose fields are written to ParaView, in
            mode order. A Stage reading fields back — the RF Stage, when
            no impedance path could be declared — asks for every Mode it
            might select, because which one that is is not known until
            they are all solved.
        verbose: Stream Palace's output.

    Returns:
        The solved Modes and the run's results.

    Raises:
        RuntimeError: When Palace returns no mode table, or when the
            binary exits abnormally with nothing to salvage — reported
            through :func:`_abort_report` rather than as the raw exit
            status.
    """
    sim.set_boundary_mode(
        freq=float(freq_hz),
        num_modes=int(num_modes),
        target=target,
        save=int(save),
    )
    sim.write_config(photonic=True)
    try:
        text = sim.run_local(palace_executable=binary, verbose=verbose)
    except (RuntimeError, subprocess.CalledProcessError) as err:
        salvaged = _salvage_mode_table(sim, err, freq_hz=freq_hz, num_modes=num_modes)
        if salvaged is None:
            if isinstance(err, subprocess.CalledProcessError):
                raise RuntimeError(
                    _abort_report(err, sim=sim, binary=binary, freq_hz=freq_hz)
                ) from err
            raise
        text = salvaged
    modes = getattr(text, "modes", {})
    if not modes:
        raise RuntimeError(
            f"Palace produced no mode table at f = {freq_hz:g} Hz; its output "
            f"is in {sim.output_dir}."
        )
    return PalaceSolve(
        modes=[
            PalaceMode(n_eff=complex(modes[mode_id]["n_eff"]), mode_id=int(mode_id))
            for mode_id in sorted(modes)
        ],
        results=text,
    )


def _death_signal(returncode: int) -> int | None:
    """The signal a process died on, if its return code says it did.

    Args:
        returncode: What the process exited with — negative when
            ``subprocess`` saw the signal itself, 128 + signum when a
            shell in between reported it.

    Returns:
        The signal number, or ``None`` for a plain exit code.
    """
    if returncode < 0:
        return -returncode
    if returncode > 128:
        return returncode - 128
    return None


def _abort_report(
    err: subprocess.CalledProcessError,
    *,
    sim: BoundaryModeSim,
    binary: Path | str,
    freq_hz: float,
) -> str:
    """Turn a dead Palace binary's exit status into something actionable.

    A broken Palace runtime — typically a bundled MPI that cannot start —
    kills the binary before the solver writes anything, and the bare
    ``CalledProcessError`` carries an exit status and nothing else. What
    a user can act on is which binary ran and whether the solver got as
    far as producing output: a signal death with none means the runtime,
    not the model, and the fix is a different binary rather than a
    different Cross-section. A plain nonzero exit is Palace refusing the
    run on its own terms, so that one is left to say why through its
    stderr rather than blamed on the runtime.

    Args:
        err: What ``run_local`` raised.
        sim: The simulation that was run, holding its output directory.
        binary: The Palace executable that ran.
        freq_hz: The frequency being solved, named in the report.

    Returns:
        The report text.
    """
    code = err.returncode
    signum = _death_signal(code)
    signal_note = ""
    if signum is not None:
        try:
            signal_note = f" ({signal.Signals(signum).name})"
        except ValueError:
            signal_note = f" (signal {signum})"

    output_dir = sim.output_dir
    wrote_output = bool(sim.last_run_files)

    # MPI closes its error blocks with a line of dashes, so the last
    # *worded* line is the one that says anything.
    stderr_note = ""
    stderr = err.stderr if isinstance(err.stderr, str) else ""
    if lines := [
        line for line in stderr.splitlines() if any(c.isalnum() for c in line)
    ]:
        stderr_note = f" Its last stderr line: {lines[-1].strip()!r}."

    if wrote_output:
        diagnosis = (
            f"It got far enough to write partial solver output (in "
            f"{output_dir}), so the runtime did start; the solve itself "
            "died before finishing."
        )
    elif signum is not None:
        diagnosis = (
            "It was killed before writing any solver output, which means "
            "the Palace runtime — typically its bundled MPI — failed "
            "before the solver started, not that the model is wrong."
        )
    else:
        diagnosis = (
            "It exited before writing any solver output; its stderr "
            "should say why it refused the run."
        )
    return (
        f"Palace aborted at f = {freq_hz:g} Hz with exit status "
        f"{code}{signal_note}. The binary that ran is {binary}. "
        f"{diagnosis}{stderr_note} Point PALACE_BIN at a Palace whose "
        "runtime works here, or re-solve on the default route with "
        "route='femwell'."
    )


def _salvage_mode_table(
    sim: BoundaryModeSim, err: Exception, *, freq_hz: float, num_modes: int
) -> PalaceTextResults | None:
    """Read a crashed Palace run's mode table back, if it is complete.

    Palace 0.17 intermittently corrupts its heap while shutting down a
    ``BoundaryMode`` solve (``free(): corrupted unsorted chunks``), after
    the solve itself has finished and its results are on disk. The crash
    is non-deterministic on an identical mesh and config, so a run that
    exits abnormally may still have answered the question it was asked —
    and the answer on disk is used rather than thrown away, which is what
    lets a gate depend on a live solve at all. The simulation clears its
    run directory before each run, so what it reads back is this run's.

    Args:
        sim: The simulation that was run.
        err: What ``run_local`` raised, named in the warning.
        freq_hz: The frequency, named in the warning.
        num_modes: Modes the solve was asked for; a table with fewer is
            a truncated write, not an answer.

    Returns:
        The parsed text results when the table is complete, else ``None``
        so the caller re-raises.
    """
    try:
        text = sim.read_results()
    except Exception:
        return None
    if text is None or len(getattr(text, "modes", {})) < num_modes:
        return None
    warnings.warn(
        f"Palace exited abnormally at f = {freq_hz:g} Hz but its complete "
        f"mode table was on disk, so the run's answer is used ({err}). "
        "Palace 0.17 is known to corrupt its heap on shutdown of a "
        "boundary-mode solve.",
        stacklevel=3,
    )
    return text


def containment_unmeasurable(stage_name: str) -> str:
    """Why the Window-containment check does not run on the Palace Route.

    ADR 0002 makes a Stage warn when a solved Mode still carries field at
    its Window boundary, because a too-small Window is the failure mode
    per-Stage Windows create. The check is measured on every candidate
    Mode as the Stage chooses between them, and Palace's *text* results —
    which is all a Stage has at that point — carry only effective
    indices. A check that quietly does not run is worse than one that
    says so.

    Args:
        stage_name: Stage the message names.

    Returns:
        The warning text.
    """
    return (
        f"The {stage_name} stage's palace route cannot check window "
        "containment: the mode table it selects from carries only effective "
        "indices, so boundary_field_ratio is NaN and a mode clipped by its "
        "window will not announce itself (ADR 0002). Re-solve with "
        f"study.{stage_name}(route='femwell') to have the window checked."
    )


#: How far the index a saved Mode's fields imply may sit from the index
#: its mode table reports before the two are called different Modes.
#: Loose on purpose: the relation behind :func:`field_index_ratio` is
#: exact only for a TEM Mode, so this catches a wrong file rather than a
#: quasi-TEM Mode's own departure from it.
FIELD_INDEX_RTOL: float = 2.0


def _check_field_is_the_mode(field: Any, mode: PalaceMode, *, stage_name: str) -> None:
    """Warn when the fields read back are not the selected Mode's.

    Which ParaView cycle holds which Mode is a convention — Palace
    writes them in mode order, so cycle ``m`` is Mode ``m`` — and a
    convention is the kind of thing that changes without an error. The
    fields carry their own index, so they can be asked whether they are
    the Mode they were fetched for.

    Args:
        field: The fields that were read back.
        mode: The Mode they were fetched for.
        stage_name: Stage asking, named in the warning.
    """
    from gsim.palace.mode_fields import field_index_ratio

    implied = field_index_ratio(field)
    expected = abs(mode.n_eff)
    if not math.isfinite(implied) or expected <= 0.0:
        return
    if 1.0 / (1.0 + FIELD_INDEX_RTOL) <= implied / expected <= 1.0 + FIELD_INDEX_RTOL:
        return
    warnings.warn(
        f"The {stage_name} stage read back fields for palace mode "
        f"{mode.mode_id} whose own effective index is about {implied:.3g}, "
        f"against the {expected:.3g} its mode table reports: these are most "
        "likely a different mode's fields, and the characteristic impedance "
        "taken from them belongs to that one. Palace writes one paraview "
        "cycle per saved mode in mode order; check that it still does.",
        stacklevel=2,
    )


#: How far inside the gap a postprocessing path keeps from a conductor
#: face, as a fraction of the smallest clearance around it. The paths
#: are sampled on the meshed domain, and under the ``"pec"`` model a
#: conductor's interior is not in it (ADR 0003); a hair inside the gap
#: is on the domain, and a hair is what the voltage integral misses.
PATH_CLEARANCE_FRACTION: float = 1e-3

#: Name of the postprocessing port the RF Stage declares on its solve.
IMPEDANCE_PORT: str = "line"


@dataclass(frozen=True)
class ImpedancePaths:
    """Where Palace integrates the line's voltage and current.

    Palace's ``BoundaryMode`` postprocessing takes two paths per
    impedance entry: an open one along which it integrates ``E`` for
    the mode voltage, and a closed one around which it integrates ``H``
    for the current. Both are in the Cross-section's own ``(h, v)``
    coordinates (um).

    Attributes:
        voltage: Two points, from the signal conductor's face across the
            gap to the return conductor's face.
        current: The corners of a loop enclosing the signal conductor and
            nothing else. Palace joins the last corner back to the first,
            so the first is not repeated.
    """

    voltage: tuple[tuple[float, float], ...]
    current: tuple[tuple[float, float], ...]


def line_impedance_paths(
    *,
    signal: Extent,
    ground: Extent,
    domain: Extent,
    clearance_fraction: float = PATH_CLEARANCE_FRACTION,
) -> ImpedancePaths:
    """Size the impedance postprocessing paths from the electrode layout.

    The voltage path crosses the gap between the two electrodes, from
    the signal electrode's face at its mid-height to the return
    electrode's face at its own, so it is the same path whichever side
    of the Junction the drive is on. The current loop is the signal
    electrode's outline pushed out by a clearance: tight enough that
    the doped Strips standing against the electrode contribute nothing
    to the enclosed current, which is what the Marks-Williams contour of
    the field-based fallback measures too.

    Args:
        signal: ``((h_min, h_max), (v_min, v_max))`` of the signal
            electrode (um).
        ground: The same for the return electrode.
        domain: The same for the meshed domain the paths are sampled on.
        clearance_fraction: Fraction of the smallest clearance around the
            signal electrode — the gap, its thickness, its distance to
            each domain wall — that the paths keep from every face.

    Returns:
        The two paths.

    Raises:
        ValueError: When the electrodes leave no gap to cross along the
            in-plane axis, or the signal electrode is not strictly inside
            the domain (an electrode the Window clips has no outline to
            loop around).
    """
    (s_lo, s_hi), (t_lo, t_hi) = signal
    (g_lo, g_hi), (u_lo, u_hi) = ground
    (d_lo, d_hi), (e_lo, e_hi) = domain

    if not (d_lo < s_lo < s_hi < d_hi and e_lo < t_lo < t_hi < e_hi):
        raise ValueError(
            f"The signal electrode spans {signal} but is not inside the meshed "
            f"domain {domain}, so no path around it can be sampled."
        )
    if g_lo > s_hi:
        gap, faces = g_lo - s_hi, (s_hi, g_lo)
    elif g_hi < s_lo:
        gap, faces = s_lo - g_hi, (s_lo, g_hi)
    else:
        raise ValueError(
            f"The electrodes at {signal[0]} and {ground[0]} leave no gap along "
            "the in-plane axis to integrate the line voltage across."
        )

    clearance = clearance_fraction * min(
        gap, t_hi - t_lo, s_lo - d_lo, d_hi - s_hi, t_lo - e_lo, e_hi - t_hi
    )
    # A hair into the gap from each face, on the domain rather than on
    # (or, under the "pec" model, inside) the conductor.
    inset = clearance if faces[0] < faces[1] else -clearance
    voltage = (
        (faces[0] + inset, 0.5 * (t_lo + t_hi)),
        (faces[1] - inset, 0.5 * (u_lo + u_hi)),
    )
    current = (
        (s_lo - clearance, t_lo - clearance),
        (s_hi + clearance, t_lo - clearance),
        (s_hi + clearance, t_hi + clearance),
        (s_lo - clearance, t_hi + clearance),
    )
    return ImpedancePaths(voltage=voltage, current=current)


def mesh_extent(sim: BoundaryModeSim) -> Extent:
    """The rectangle a meshed simulation's 2D mesh covers.

    Args:
        sim: A meshed ``BoundaryModeSim``.

    Returns:
        ``((h_min, h_max), (v_min, v_max))`` in the mesh's own
        cross-section coordinates (um).

    Raises:
        ValueError: When the simulation has not been meshed.
    """
    import meshio

    points = meshio.read(str(sim.mesh_path)).points
    h = points[:, 0]
    v = points[:, 1]
    return ((float(h.min()), float(h.max())), (float(v.min()), float(v.max())))


def declare_impedance_paths(
    sim: BoundaryModeSim, paths: ImpedancePaths, *, nsamples: int = 100
) -> int:
    """Register the paths as a postprocessing path of the solve.

    A ``BoundaryMode`` path is postprocessing only — it loads nothing
    and leaves the eigenproblem as it was. The simulation says which
    index Palace will report it under, and that index is what
    :func:`native_line_impedance` reads.

    Args:
        sim: The simulation about to be solved; meshed or not, since the
            paths take no part in meshing.
        paths: What :func:`line_impedance_paths` sized.
        nsamples: Quadrature order of each line integral.

    Returns:
        The index Palace reports the line's impedance under.
    """
    return sim.add_impedance_path(
        IMPEDANCE_PORT,
        voltage=[list(point) for point in paths.voltage],
        current=[list(point) for point in paths.current],
        nsamples=nsamples,
    )


#: Below this ``Z_PV / Z_PI`` — which is ``(|V| |I| / 2P)^2`` — the voltage
#: across the gap carries none of the Mode's power, and the Mode is not
#: the line Mode between the two electrodes. A quasi-TEM line Mode has
#: ``|V| |I| ~ 2P``; a lossy one sits within a factor of a few of it; a
#: Mode running between both electrodes together and the Window wall
#: puts the electrodes at one potential and sits many decades under.
MIN_VOLTAGE_POWER_RATIO: float = 1e-2


def native_line_impedance(
    results: PalaceTextResults, mode: PalaceMode, *, index: int
) -> LineReading | None:
    """The selected Mode's reading off Palace's own ``mode-Z.csv``.

    Palace reports two magnitudes per entry: ``Z_PV = |V|^2 / 2P`` from
    the voltage path alone, and ``Z_VI = |V| / |I|`` once a current path
    is declared. Their ratio ``Z_VI^2 / Z_PV = 2P / |I|^2`` is the
    power-current impedance the femwell Route and the field-based
    fallback both compute, so that is what is read; a table carrying
    ``Z_PV`` alone is a different definition, not a nearer answer, and
    is declined the same as no table.

    The voltage cancels out of that ratio, so it is read on its own for
    the wall-Mode diagnostic: a Mode whose gap voltage carries almost
    none of its power is not running between the two electrodes.

    Args:
        results: The run's text results.
        mode: The selected Mode.
        index: The postprocessing index the path was declared under.

    Returns:
        The reading — its impedance real, since the tables carry
        magnitudes — or ``None`` when the tables are absent, carry no
        current path, no current (a loop enclosing none), or not this
        Mode.
    """
    z_pv = results.characteristic_impedance(
        index=index, mode=mode.mode_id, quantity="Z_PV"
    )
    z_vi = results.characteristic_impedance(
        index=index, mode=mode.mode_id, quantity="Z_VI"
    )
    if z_pv is None or z_vi is None or z_pv <= 0.0 or z_vi <= 0.0:
        return None
    z_pi = z_vi * z_vi / z_pv
    wall_mode = z_pv < MIN_VOLTAGE_POWER_RATIO * z_pi
    if wall_mode:
        diagnostic = (
            f"whose voltage across the gap carries {z_pv / z_pi:.2g} of the "
            "power its current implies: the two electrodes sit at one "
            "potential, so this is a mode between them and the window wall "
            "rather than the line mode between them"
        )
    else:
        diagnostic = (
            f"whose voltage across the gap carries {z_pv / z_pi:.2g} of the "
            "power its current implies: the line mode between the electrodes"
        )
    return LineReading(
        n_eff=complex(mode.n_eff),
        z0_ohm=complex(z_pi),
        wall_mode=wall_mode,
        diagnostic=diagnostic,
    )


def field_line_impedance(
    sim: BoundaryModeSim,
    mode: PalaceMode,
    *,
    h_span: tuple[float, float],
    v_span: tuple[float, float],
    stage_name: str,
) -> complex:
    """Characteristic impedance of a Palace Mode, off its saved fields.

    The Marks-Williams power-current integral the femwell Route runs,
    run on the fields Palace wrote for the selected Mode: the complex
    Poynting flux over the whole Cross-section, over the current
    Ampere's law reads around the signal conductor.

    Reading fields back is the one part of a Palace solve that depends
    on a file the solver may not have written, so a missing or
    unreadable one is reported as NaN with the reason rather than
    raising in the middle of a frequency sweep.

    Args:
        sim: The simulation that was run, which reads its saved fields
            back.
        mode: The selected Mode, whose ``mode_id`` names its saved
            fields.
        h_span: ``(min, max)`` of the signal conductor along the
            Cross-section's in-plane axis (um).
        v_span: ``(min, max)`` along its vertical axis (um).
        stage_name: Stage asking, named in the warning.

    Returns:
        The complex characteristic impedance in ohms, or NaN when the
        fields could not be read.
    """
    from gsim.palace.mode_fields import z0_power_current as palace_z0

    try:
        field = sim.read_mode_field(mode.mode_id)
        z0 = palace_z0(field, h_span=h_span, v_span=v_span)
    except Exception as err:
        warnings.warn(
            f"The {stage_name} stage's palace route could not read mode "
            f"{mode.mode_id}'s saved fields, so its characteristic impedance "
            f"comes back NaN: {err} Palace's output is in {sim.output_dir}.",
            stacklevel=2,
        )
        return complex(math.nan, math.nan)
    # Outside the guard above: the check warns rather than raises, and a
    # caller running with warnings as errors must see the wrong-Mode
    # diagnostic rather than have it caught here and reported as an
    # unreadable file.
    _check_field_is_the_mode(field, mode, stage_name=stage_name)
    return z0


def palace_line_impedance(
    sim: BoundaryModeSim,
    solve: PalaceSolve,
    mode: PalaceMode,
    *,
    index: int | None,
    signal: Conductor,
    stage_name: str,
) -> LineReading:
    """The selected Mode's reading: tables first, saved fields as fallback.

    Palace's own answer first: the power-current impedance its
    postprocessing paths measured (:func:`native_line_impedance`), read
    off this run's tables under the index the path was declared at, with
    the wall-Mode diagnostic the gap voltage gives. A run without the
    tables — a solve that declared no path, or an older Palace — falls
    back to the saved fields (:func:`field_line_impedance`), whose NaN
    contract stands and which say nothing about which Mode this is.

    Args:
        sim: The simulation that was run.
        solve: The run, holding its results.
        mode: The selected Mode.
        index: The postprocessing index the impedance path was declared
            under, or ``None`` when none could be declared.
        signal: The signal electrode, whose outline the fallback
            integrates around.
        stage_name: Stage asking, named in the fallback's warning.

    Returns:
        The reading: impedance real off the tables, complex off the
        fields, NaN when neither could be read.
    """
    if index is not None:
        native = native_line_impedance(solve.results, mode, index=index)
        if native is not None:
            return native
    h_span, v_span = signal.extent
    z0 = field_line_impedance(
        sim, mode, h_span=h_span, v_span=v_span, stage_name=stage_name
    )
    return LineReading(
        n_eff=complex(mode.n_eff),
        z0_ohm=z0,
        wall_mode=None,
        diagnostic="the impedance was read off the saved fields, which carry "
        "no gap voltage to compare against the power",
    )


def conductor_clearance(
    surroundings: Sequence[SurroundingRegion],
    *,
    window: tuple[float, float],
    window_z: tuple[float, float],
    stage_name: str,
) -> None:
    """Refuse a Staircase whose metal is sliced by the Window.

    A drawn conductor is meshed as an outline with its interior left out
    of the domain (ADR 0003). When the Window cuts through one, that
    outline runs along the Window's own outer wall, and Palace's meshing
    does not survive it — the solver aborts rather than reporting
    anything. The femwell Route meshes it, so this is a Route limitation
    and not a modelling one, which is why it is checked here and not in
    the Staircase.

    Args:
        surroundings: The Regions redrawn around the Strips.
        window: In-plane Window the Staircase is clipped to (um).
        window_z: Vertical Window (um).
        stage_name: Stage asking, named in the error.

    Raises:
        ValueError: When a conductor crosses the Window boundary on
            either axis.
    """
    for region in surroundings:
        if region.layer_type not in ("conductor", "via"):
            continue
        for extent, bounds, axis in (
            (region.h, window, "window"),
            (region.z, window_z, "window_z"),
        ):
            inside = extent[0] >= bounds[0] and extent[1] <= bounds[1]
            outside = extent[1] <= bounds[0] or extent[0] >= bounds[1]
            if inside or outside:
                continue
            raise ValueError(
                f"The {stage_name} stage's palace route cannot solve this "
                f"staircase: the drawn conductor '{region.name}' spans "
                f"{extent[0]:.4g}..{extent[1]:.4g} um, which the {axis} "
                f"{bounds[0]:.4g}..{bounds[1]:.4g} um cuts through, so its "
                "perfect-conductor outline would run along the window's own "
                f"wall. Widen the window to contain it (study.{stage_name}"
                f"({axis}=...)) or narrow it to leave the conductor out, or "
                f"solve with study.{stage_name}(route='femwell'), which "
                "meshes it."
            )


class PalaceRoute(Route):
    """The Route interface on Palace.

    Keeps, across one Stage run, the binary it resolved, the index its
    impedance path was declared under, and the last run's results.
    """

    name: ClassVar[EMRoute] = "palace"
    conductor_model: ClassVar[ConductorModel] = "pec"
    continuous_materials: ClassVar[bool] = False

    def __init__(self) -> None:
        """A fresh route: nothing resolved, nothing declared, nothing run."""
        self._binary: Path | None = None
        self._index: int | None = None
        self._last: PalaceSolve | None = None
        self._said_unmeasurable = False

    @property
    def impedance_index(self) -> int | None:
        """Index the line's impedance path was declared under, or None."""
        return self._index

    def require(self, *, stage_name: str) -> None:
        """Locate the Palace binary, once, for this run.

        Raises:
            RuntimeError: When no binary is available.
        """
        self._binary = require_palace_binary(stage_name=stage_name)

    def _executable(self, stage_name: str) -> Path:
        """The resolved binary, resolving it if :meth:`require` never ran."""
        if self._binary is None:
            self._binary = require_palace_binary(stage_name=stage_name)
        return self._binary

    def check_staircase(
        self,
        staircase: StaircaseCrossSection,
        *,
        window: tuple[float, float],
        window_z: tuple[float, float],
        stage_name: str,
    ) -> None:
        """Refuse a Staircase whose drawn metal the Window cuts through."""
        conductor_clearance(
            staircase.surroundings,
            window=window,
            window_z=window_z,
            stage_name=stage_name,
        )

    def prepare_line(
        self,
        sim: BoundaryModeSim,
        *,
        signal: Conductor,
        return_: Conductor | None,
        stage_name: str,
    ) -> None:
        """Declare the voltage path and current loop on the simulation.

        Sized from the electrodes and the meshed domain
        (:func:`line_impedance_paths`). A line whose paths cannot be
        sized — no single return electrode, or a signal electrode the
        Window clips — is not refused: the impedance is then read off the
        saved fields, and the solve saves every Mode for it.

        Warns:
            UserWarning: When the paths cannot be declared.
        """
        self._index = None
        if return_ is None:
            reason = "the staircase has no single return electrode beside the signal"
        else:
            try:
                paths = line_impedance_paths(
                    signal=signal.extent,
                    ground=return_.extent,
                    domain=mesh_extent(sim),
                )
            except ValueError as err:
                reason = str(err)
            else:
                self._index = declare_impedance_paths(sim, paths)
                return
        warnings.warn(
            f"The {stage_name} stage's palace route cannot declare its "
            f"impedance paths, so the characteristic impedance is read off "
            f"the saved fields instead: {reason}",
            stacklevel=3,
        )

    def solve(
        self,
        sim: BoundaryModeSim,
        *,
        freq_hz: float,
        num_modes: int,
        target: float | None,
        order: int,  # noqa: ARG002 - Palace's element order is its own
        verbose: bool,
        stage_name: str,
        epsilon: ArrayLike | None = None,
    ) -> Sequence[Any]:
        """Run Palace on the meshed simulation at one frequency.

        Fields are saved only when the impedance has to be read off them
        — every Mode then, because which one is the line Mode is not
        known until they are all solved.

        Raises:
            ValueError: When a continuous permittivity is asked for,
                which Palace cannot carry.

        Warns:
            UserWarning: Once per run, that the Window containment
                cannot be checked on this Route (ADR 0002).
        """
        if epsilon is not None:
            raise ValueError(
                "The palace route takes piecewise-constant materials per "
                "region and cannot carry a continuous permittivity."
            )
        if not self._said_unmeasurable:
            self._said_unmeasurable = True
            warnings.warn(containment_unmeasurable(stage_name), stacklevel=3)
        self._last = solve_palace_modes(
            sim,
            freq_hz=freq_hz,
            num_modes=num_modes,
            binary=self._executable(stage_name),
            target=target if target is not None else 0.0,
            save=0 if self._index is not None else num_modes,
            verbose=verbose,
        )
        return self._last.modes

    def boundary_ratio(self, mode: Any) -> float:  # noqa: ARG002 - no fields in hand
        """NaN: Palace's mode table carries no field to measure."""
        return math.nan

    def strip_fraction_outside(
        self,
        mode: Any,  # noqa: ARG002 - no fields in hand
        span: tuple[float, float],  # noqa: ARG002 - nothing to measure over
        *,
        stage_name: str,
    ) -> float:
        """NaN, and say so: no mode fields come back to measure."""
        warnings.warn(
            f"The {stage_name} stage's palace route cannot check how much of "
            "the mode sits outside the strip extent either, for the same "
            "reason: no mode fields come back. Re-solve with "
            f"study.{stage_name}(route='femwell') at the same strip count to "
            "have both checks run on the identical staircase.",
            stacklevel=3,
        )
        return math.nan

    def read_line(
        self,
        sim: BoundaryModeSim,
        mode: Any,
        *,
        freq_hz: float,  # noqa: ARG002 - Palace's tables are per run
        signal: Conductor,
        return_: Conductor | None,  # noqa: ARG002 - the gap voltage stands in
        stage_name: str,
    ) -> LineReading:
        """Tables first, saved fields as the fallback.

        Palace's own ``mode-Z.csv`` under the declared index gives the
        power-current impedance and, through the voltage across the gap,
        the wall-Mode diagnostic. Without the tables the impedance is
        the Marks-Williams integral on the saved fields and nothing says
        which Mode this is.
        """
        if self._last is None:
            raise RuntimeError("read_line() called before solve().")
        return palace_line_impedance(
            sim,
            self._last,
            mode,
            index=self._index,
            signal=signal,
            stage_name=stage_name,
        )
