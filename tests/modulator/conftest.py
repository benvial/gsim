"""Shared device fixtures for the gsim.modulator tests.

The device is the lateral PN phase shifter ``demo_phase_shifter`` draws:
a rib with four doped regions (n_pad | n_rib | p_rib | p_pad) along the
in-plane axis and a metal electrode landing on each outer pad. Its drawn
dimensions are re-exported under the names the assertions were written
against, read off the builder rather than restated here.
"""

from __future__ import annotations

import pytest

from gsim.modulator.demo import (
    DEFAULT_CENTER_UM,
    DEFAULT_ELECTRODE_THICKNESS_UM,
    DEFAULT_HALF_WIDTH_UM,
    DEFAULT_LENGTH_UM,
    DEFAULT_PAD_WIDTH_UM,
    DEFAULT_RIB_HEIGHT_UM,
    demo_phase_shifter,
)

CENTER_Y = DEFAULT_CENTER_UM
HALF_WIDTH = DEFAULT_HALF_WIDTH_UM
PAD_WIDTH = DEFAULT_PAD_WIDTH_UM
RIB_HEIGHT = DEFAULT_RIB_HEIGHT_UM
ELECTRODE_THICKNESS = DEFAULT_ELECTRODE_THICKNESS_UM
LENGTH_UM = DEFAULT_LENGTH_UM


def build_demo():
    """Draw the lateral PN phase shifter the tests are written against."""
    return demo_phase_shifter()


@pytest.fixture(scope="module")
def demo():
    """The demo phase shifter, drawn once per module."""
    return build_demo()


@pytest.fixture(scope="module")
def phase_shifter(demo):
    """The lateral PN phase shifter component and its doped stack."""
    return demo.component, demo.stack


@pytest.fixture
def device(demo):
    """The device description matching what :func:`build_demo` drew."""
    return demo.device


@pytest.fixture
def study(phase_shifter, device, tmp_path):
    """A Study over the phase shifter, writing meshes into ``tmp_path``."""
    from gsim.modulator import Study

    component, stack = phase_shifter
    return Study(
        component=component,
        stack=stack,
        device=device,
        plane="x=0",
        output_dir=tmp_path / "study",
    )
