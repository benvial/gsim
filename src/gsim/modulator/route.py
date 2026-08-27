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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from gsim.palace import BoundaryModeSim

__all__ = [
    "DEFAULT_PALACE_STRIPS",
    "EMRoute",
    "PalaceMode",
    "containment_unmeasurable",
    "mode_boundary_ratio",
    "palace_binary",
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

    Palace reports its Modes as effective indices in a text result rather
    than as field vectors, so a Palace Mode carries its index and its
    solver-assigned number and nothing else. Anything reading fields —
    the Marks-Williams impedance extraction, the Window-containment
    check — is a femwell-Route capability, and the Stages report those
    quantities as NaN under this Route rather than inventing them.

    Attributes:
        n_eff: Complex effective index (``exp(+i omega t)`` convention).
        mode_id: Palace's own mode number, 1-based.
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
        verbose: Stream Palace's output.

    Returns:
        The solved Modes, in Palace's own mode order.

    Raises:
        RuntimeError: When Palace returns no mode table.
    """
    from gsim.palace.results import load_text_results

    sim.set_boundary_mode(freq=float(freq_hz), num_modes=int(num_modes), target=target)
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
    per-Stage Windows create. Palace's boundary-mode results carry no
    fields, so the check cannot run there — and a check that quietly does
    not run is worse than one that says so.

    Args:
        stage_name: Stage the message names.

    Returns:
        The warning text.
    """
    return (
        f"The {stage_name} stage's palace route cannot check window "
        "containment: Palace's boundary-mode results carry no mode fields, "
        "so boundary_field_ratio is NaN and a mode clipped by its window "
        "will not announce itself (ADR 0002). Re-solve with "
        f"study.{stage_name}(route='femwell') to have the window checked."
    )


def mode_boundary_ratio(mode: Any) -> float:
    """Peak field at the Window boundary over peak field overall.

    Args:
        mode: A solved Mode from either Route.

    Returns:
        The ratio for a femwell Mode, and NaN for a Palace one — Palace's
        text results carry no field, so the containment check is a
        femwell-Route capability rather than a number to fabricate.
    """
    if isinstance(mode, PalaceMode):
        return math.nan
    from gsim.femwell.adapter import boundary_field_ratio

    return boundary_field_ratio(mode)
