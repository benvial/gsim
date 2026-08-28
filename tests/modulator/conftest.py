"""Shared device fixtures for the gsim.modulator tests.

The device is the lateral PN phase shifter ``demo_phase_shifter`` draws:
a rib with four doped regions (n_pad | n_rib | p_rib | p_pad) along the
in-plane axis and a metal electrode landing on each outer pad. Its drawn
dimensions are re-exported under the names the assertions were written
against, read off the builder rather than restated here.
"""

from __future__ import annotations

import numpy as np
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
from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap

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


#: In-plane extent of the doped slab, pads included (um).
SLAB = (CENTER_Y - HALF_WIDTH - PAD_WIDTH, CENTER_Y + HALF_WIDTH + PAD_WIDTH)


def carriers_at(bias_v: float) -> CarrierMap:
    """A Carrier map across the doped slab, depleting with reverse bias."""
    y = np.linspace(SLAB[0], SLAB[1], 61)
    z = np.linspace(0.0, RIB_HEIGHT, 5)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    yy, zz = yy.ravel(), zz.ravel()
    depleted = np.abs(yy - CENTER_Y) < 0.05 * np.sqrt(1.0 + abs(bias_v))
    n_side = yy < CENTER_Y
    return CarrierMap(
        x_um=yy,
        y_um=zz,
        region=["n_rib" if side else "p_rib" for side in n_side],
        electrons_cm3=np.where(depleted | ~n_side, 1e10, 1e18),
        holes_cm3=np.where(depleted | n_side, 1e10, 1e18),
        potential_v=np.zeros(yy.size),
        net_doping_cm3=np.zeros(yy.size),
    )


@pytest.fixture
def biased(study):
    """A Study whose charge Stage already holds a two-point sweep."""
    study.charge._result = BiasSweepResult(
        contact="cathode",
        points=[BiasPoint(bias_v=v, carriers=carriers_at(v)) for v in (0.0, 2.0)],
    )
    study.charge._has_run = True
    return study
