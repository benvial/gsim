"""Selecting the physical line Mode out of a set of solved Modes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gsim.common.modes import (
    NoLineModeError,
    propagating_modes,
    select_line_mode,
)


def mode(n_eff: complex):
    """A solver Mode stand-in carrying only its effective index."""
    return SimpleNamespace(n_eff=complex(n_eff))


class TestDefaultRule:
    def test_picks_the_slowest_propagating_mode(self):
        modes = [mode(3.2 - 0.01j), mode(2.1 - 0.005j)]
        assert select_line_mode(modes).n_eff == 3.2 - 0.01j

    def test_rejects_evanescent_and_spurious_modes(self):
        physical = mode(3.0 - 0.02j)
        modes = [
            mode(0.4 - 0.001j),  # below the light line: not guided
            mode(6.0 - 9.0j),  # |Im| > Re: spurious / lossy junk
            physical,
        ]
        assert select_line_mode(modes) is physical

    def test_accepts_plain_complex_numbers_and_mappings(self):
        assert select_line_mode([1.5 + 0j, 3.0 + 0j]) == 3.0 + 0j
        picked = select_line_mode([{"n_eff": 2.0 + 0j}, {"n_eff": 4.0 + 0j}])
        assert picked["n_eff"] == 4.0 + 0j

    def test_candidates_are_available_on_their_own(self):
        modes = [mode(0.5 + 0j), mode(3.0 - 0.1j)]
        assert [m.n_eff for m in propagating_modes(modes)] == [3.0 - 0.1j]


class TestCustomRule:
    def test_caller_rule_replaces_the_candidate_set(self):
        modes = [mode(3.2 + 0j), mode(2.1 + 0j)]

        def slowest_only(candidates):
            return [min(candidates, key=lambda m: m.n_eff.real)]

        assert select_line_mode(modes, rule=slowest_only).n_eff == 2.1 + 0j

    def test_caller_rule_may_keep_modes_the_default_rejects(self):
        modes = [mode(0.4 + 0j)]
        assert select_line_mode(modes, rule=list).n_eff == 0.4 + 0j

    def test_empty_result_from_a_caller_rule_is_reported(self):
        with pytest.raises(NoLineModeError):
            select_line_mode([mode(3.0 + 0j)], rule=lambda _modes: [])


class TestDegeneracy:
    def test_near_degenerate_candidates_warn_and_name_the_indices(self):
        modes = [mode(3.00 + 0j), mode(2.99 + 0j)]
        with pytest.warns(UserWarning, match="2.99"):
            picked = select_line_mode(modes)
        assert picked.n_eff == 3.00 + 0j

    def test_well_separated_candidates_are_silent(self):
        import warnings

        modes = [mode(3.0 + 0j), mode(2.0 + 0j)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert select_line_mode(modes).n_eff == 3.0 + 0j

    def test_tolerance_is_configurable(self):
        modes = [mode(3.0 + 0j), mode(2.0 + 0j)]
        with pytest.warns(UserWarning, match="degenerate"):
            select_line_mode(modes, degeneracy_rtol=0.5)


class TestNoCandidates:
    def test_error_names_every_solved_mode(self):
        modes = [mode(0.4 + 0j), mode(0.2 - 3.0j)]
        with pytest.raises(NoLineModeError) as excinfo:
            select_line_mode(modes)
        message = str(excinfo.value)
        assert "0.4" in message
        assert "2 mode" in message

    def test_error_when_nothing_was_solved_at_all(self):
        with pytest.raises(NoLineModeError, match="no modes"):
            select_line_mode([])

    def test_is_a_value_error(self):
        assert issubclass(NoLineModeError, ValueError)
