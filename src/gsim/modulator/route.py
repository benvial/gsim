"""Which Backend answers an EM Stage's question.

A Route is a per-Stage, per-run choice between the Backends that can
answer the same question, and both EM Stages of a modulator Study accept
one. femwell is the default because it needs no external binary and can
carry a continuously varying permittivity; Palace is the second
first-class Route the originating spec asks for, and the one the
cross-Route agreement check is written against.

The two Backends differ in what they can express, and that difference is
the reason a Route is a choice rather than an implementation detail:
Palace takes piecewise-constant materials per mesh Region and nothing
else, so a carrier distribution reaches it as a Staircase. Where a Stage
can also solve a continuous profile, selecting the Palace Route moves it
onto the Staircase representation of the same physics.

Nothing here imports a Backend at module scope: a Study that never
selects a Route pays for neither.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from gsim.palace import BoundaryModeSim

__all__ = [
    "DEFAULT_PALACE_STRIPS",
    "FIELD_INDEX_RTOL",
    "EMRoute",
    "PalaceMode",
    "containment_unmeasurable",
    "metallic_boundary_unexpressed",
    "mode_boundary_ratio",
    "palace_binary",
    "palace_line_impedance",
    "require_palace_binary",
    "require_route",
    "solve_palace_modes",
]

#: The Backends an EM Stage can be routed to.
EMRoute = Literal["femwell", "palace"]

#: Strip count the Palace Route falls back to when a Stage that can also
#: solve a continuous profile has not been given one. Palace cannot
#: express a continuous profile at all, so the choice is a strip count or
#: an error, and a workable default beats an error.
DEFAULT_PALACE_STRIPS: int = 5


@dataclass(frozen=True)
class PalaceMode:
    """One Mode of a Palace ``BoundaryMode`` solve.

    Palace's *text* results report Modes as effective indices rather than
    as field vectors, so a Palace Mode carries its index and its
    solver-assigned number and nothing else. Its fields are not lost —
    a solve asked to save them writes them to ParaView, and
    :func:`palace_line_impedance` reads one back through
    :mod:`gsim.palace.mode_fields` — but they are not in hand at
    selection time, so anything a Stage measures while choosing a Mode
    (the Window-containment ratio) stays a femwell-Route capability
    rather than something to invent.

    Attributes:
        n_eff: Complex effective index (``exp(+i omega t)`` convention).
        mode_id: Palace's own mode number, 1-based. This is also the
            ParaView cycle its saved fields land in.
    """

    n_eff: complex
    mode_id: int


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


def palace_binary(binary: Path | None, *, stage_name: str) -> Path:
    """Narrow :func:`require_route`'s answer to a Palace executable.

    A Stage resolves its binary once, at the top of a run, and threads it
    down as ``Path | None`` because the femwell Route needs none. This
    turns that back into the ``Path`` the Palace calls take, resolving
    again — and naming the right Stage — if it was never resolved.

    Args:
        binary: What :func:`require_route` returned, if anything.
        stage_name: Stage asking, named in the error.

    Returns:
        A runnable Palace executable.

    Raises:
        RuntimeError: When no binary is available.
    """
    if binary is not None:
        return binary
    return require_palace_binary(stage_name=stage_name)


def require_route(route: EMRoute, *, stage_name: str) -> Path | None:
    """Check a Route's runtime before the Stage spends anything.

    Called before meshing and before the charge solve, so a user who
    selected a Route they cannot run pays only for the error message.

    Args:
        route: The selected Route.
        stage_name: Stage asking, named in the error.

    Returns:
        The Palace executable on the Palace Route, so the Stage resolves
        it once rather than once per solve; ``None`` on the femwell
        Route, which needs no binary.

    Raises:
        ImportError: When the femwell extra is not installed.
        RuntimeError: When no Palace binary is available.
    """
    if route == "femwell":
        from gsim.femwell.runtime import require_femwell, require_skfem

        require_femwell()
        require_skfem()
        return None
    return require_palace_binary(stage_name=stage_name)


def solve_palace_modes(
    sim: BoundaryModeSim,
    *,
    freq_hz: float,
    num_modes: int,
    binary: Path,
    target: float = 0.0,
    save: int = 0,
    verbose: bool = False,
) -> list[PalaceMode]:
    """Solve one ``BoundaryMode`` problem with Palace on an existing mesh.

    The simulation must already be meshed. Meshing is the Route-neutral
    part of a solve, so a caller comparing the two Routes can mesh once
    and hand the same simulation to both.

    Args:
        sim: A meshed ``BoundaryModeSim``.
        freq_hz: Frequency to solve at (Hz).
        num_modes: Number of Modes to compute.
        target: Effective-index target centring the shift-and-invert
            search; ``0.0`` leaves it to Palace.
        binary: Palace executable, from :func:`require_route` or
            :func:`require_palace_binary`.
        save: Number of Modes whose fields are written to ParaView, in
            mode order. A Stage reading fields back — the RF Stage, for
            its characteristic impedance — asks for every Mode it might
            select, because which one that is is not known until they
            are all solved.
        verbose: Stream Palace's output.

    Returns:
        The solved Modes, in Palace's own mode order.

    Raises:
        RuntimeError: When Palace returns no mode table.
    """
    from gsim.palace.results import load_text_results

    sim.set_boundary_mode(
        freq=float(freq_hz),
        num_modes=int(num_modes),
        target=target,
        save=int(save),
    )
    sim.write_config(photonic=True)
    results: Any = sim.run_local(palace_executable=binary, verbose=verbose)
    text = results if hasattr(results, "modes") else load_text_results(results)
    modes = getattr(text, "modes", {})
    if not modes:
        raise RuntimeError(
            f"Palace produced no mode table at f = {freq_hz:g} Hz; its output "
            f"is in {sim.output_dir}."
        )
    return [
        PalaceMode(n_eff=complex(modes[mode_id]["n_eff"]), mode_id=int(mode_id))
        for mode_id in sorted(modes)
    ]


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


def metallic_boundary_unexpressed(stage_name: str) -> str:
    """Why the Palace Route does not shield its Window.

    ``metallic_boundaries`` puts a perfect conductor on the outer edge of
    a Stage's Window, which is what makes a line solve a *shielded* line
    solve. Nothing in the Palace pipeline expresses it: the native-2D
    mesher tags a conductor's own outline, never the domain wall, and
    Palace's own default for a boundary attribute it was given no
    condition for is PMC — the magnetic wall, the opposite one. So the
    two Routes do not solve the same boundary-value problem, and their
    Modes cannot be expected to match.

    Args:
        stage_name: Stage the message names.

    Returns:
        The warning text.
    """
    return (
        f"The {stage_name} stage's palace route cannot put a metallic wall "
        "around its window: nothing in the palace pipeline expresses "
        "metallic_boundaries, and palace's own default for the outer "
        "boundary is PMC rather than PEC. The femwell route shields the "
        "same window, so the two routes are solving different problems and "
        "their modes will not agree. Widen the window with "
        f"study.{stage_name}(window=..., window_z=...) until the wall stops "
        "mattering, which is the only way to bring them together today: "
        "turning the femwell route's wall off instead would leave its "
        "perfect electrodes as open slots rather than as conductors."
    )


def mode_boundary_ratio(mode: Any) -> float:
    """Peak field at the Window boundary over peak field overall.

    Args:
        mode: A solved Mode from either Route.

    Returns:
        The ratio for a femwell Mode, and NaN for a Palace one — the
        Palace mode table a Stage selects from carries no field, so the
        containment check is a femwell-Route capability rather than a
        number to fabricate.
    """
    if isinstance(mode, PalaceMode):
        return math.nan
    from gsim.femwell.adapter import boundary_field_ratio

    return boundary_field_ratio(mode)


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


def palace_line_impedance(
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
        sim: The simulation that was run, holding its output directory.
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
    from gsim.palace.mode_fields import load_boundary_mode_field
    from gsim.palace.mode_fields import z0_power_current as palace_z0

    output_dir = sim.output_dir
    try:
        if output_dir is None:
            raise RuntimeError("the simulation has no output directory.")  # noqa: TRY301
        field = load_boundary_mode_field(output_dir, mode_id=mode.mode_id)
        _check_field_is_the_mode(field, mode, stage_name=stage_name)
        return palace_z0(field, h_span=h_span, v_span=v_span)
    except Exception as err:
        warnings.warn(
            f"The {stage_name} stage's palace route could not read mode "
            f"{mode.mode_id}'s saved fields, so its characteristic impedance "
            f"comes back NaN: {err} Palace's output is in {output_dir}.",
            stacklevel=2,
        )
        return complex(math.nan, math.nan)
