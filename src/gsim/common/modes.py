"""Choosing the physical line Mode out of a set of solved Modes.

An RF solve returns several Modes: the quasi-TEM line Mode a designer
wants, plus evanescent and spurious ones the discretization produces.
Picking the physical one is a library concern rather than a heuristic
every caller re-invents, so :func:`select_line_mode` owns the default
rule, the substitution point for unusual lines, and the ambiguity
warning.

The default rule keeps Modes that propagate (``Re(n_eff)`` above the
light line, ``|Im(n_eff)|`` smaller than ``Re(n_eff)``) and picks the
slowest-travelling of them — the right choice for a line with a single
signal conductor, where the loaded quasi-TEM Mode carries the highest
effective index. Lines that break that assumption pass their own
candidate rule instead.

Modes are read duck-typed: anything with an ``n_eff`` attribute (femwell
``Mode``), a mapping with an ``"n_eff"`` key (the Palace result rows), or
a bare complex number works.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = [
    "LineModeRule",
    "NoLineModeError",
    "mode_index",
    "propagating_modes",
    "select_line_mode",
]

#: A candidate rule: given every solved Mode, return the physical ones.
LineModeRule = Callable[[Sequence[Any]], Sequence[Any]]


class NoLineModeError(ValueError):
    """No solved Mode qualifies as the physical line Mode."""


def mode_index(mode: Any) -> complex:
    """Effective index of a solved Mode.

    Args:
        mode: A solver Mode (an object with ``n_eff``, a mapping with an
            ``"n_eff"`` key, or a bare complex effective index).

    Returns:
        The complex effective index.
    """
    if isinstance(mode, Mapping):
        return complex(mode["n_eff"])
    n_eff = getattr(mode, "n_eff", None)
    if n_eff is not None:
        return complex(n_eff)
    return complex(mode)


def propagating_modes[ModeT](
    modes: Sequence[ModeT],
    *,
    min_index: float = 1.0,
) -> list[ModeT]:
    """Keep the Modes that propagate, dropping evanescent and spurious ones.

    Args:
        modes: Every solved Mode.
        min_index: Lower bound on ``Re(n_eff)``; Modes at or below it are
            not guided by the line (default: the vacuum light line).

    Returns:
        The propagating Modes, in the order they were solved.
    """
    kept = []
    for mode in modes:
        n_eff = mode_index(mode)
        if n_eff.real > min_index and abs(n_eff.imag) < n_eff.real:
            kept.append(mode)
    return kept


def _describe(modes: Sequence[Any]) -> str:
    """One-line summary of the effective indices that were solved."""
    if not modes:
        return "no modes were solved"
    indices = ", ".join(f"{mode_index(m):.4g}" for m in modes)
    return f"{len(modes)} mode(s) solved with n_eff = {indices}"


def select_line_mode[ModeT](
    modes: Sequence[ModeT],
    *,
    rule: LineModeRule | None = None,
    min_index: float = 1.0,
    degeneracy_rtol: float = 0.03,
) -> ModeT:
    """Select the physical line Mode from a set of solved Modes.

    Args:
        modes: Every Mode the solve returned.
        rule: Candidate rule replacing the default
            :func:`propagating_modes`. It receives every solved Mode and
            returns the physically admissible ones; the slowest of those
            (highest ``Re(n_eff)``) is selected.
        min_index: Lower bound on ``Re(n_eff)`` for the default rule.
        degeneracy_rtol: Relative spread in ``Re(n_eff)`` within which two
            candidates count as ambiguous and a warning is issued.

    Returns:
        The selected Mode, as handed in.

    Raises:
        NoLineModeError: When no candidate survives the rule, naming the
            effective indices that were solved.

    Warns:
        UserWarning: When two or more candidates sit within
            ``degeneracy_rtol`` of the selected effective index, so the
            choice between them is not physically meaningful.
    """
    candidates: Sequence[ModeT] = (
        propagating_modes(modes, min_index=min_index)
        if rule is None
        else list(rule(modes))
    )
    if not candidates:
        raise NoLineModeError(
            f"No propagating line mode among the solved modes: {_describe(modes)}. "
            "Solve more modes, move the eigenvalue guess (n_guess) toward the "
            "expected line index, or pass a rule= selecting the mode yourself."
        )

    selected = max(candidates, key=lambda mode: mode_index(mode).real)
    selected_index = mode_index(selected).real
    if selected_index != 0.0:
        degenerate = [
            mode
            for mode in candidates
            if mode is not selected
            and abs(mode_index(mode).real - selected_index) / abs(selected_index)
            <= degeneracy_rtol
        ]
        if degenerate:
            warnings.warn(
                f"Line mode selection is degenerate: n_eff = "
                f"{mode_index(selected):.6g} was selected but "
                f"{', '.join(f'{mode_index(m):.6g}' for m in degenerate)} "
                f"sit within {degeneracy_rtol:.1%} of it. Inspect the mode "
                "fields and pass rule= to select explicitly.",
                stacklevel=2,
            )
    return selected
