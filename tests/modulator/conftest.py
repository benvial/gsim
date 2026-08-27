"""Shared device fixtures for the gsim.modulator tests.

The device is the lateral PN phase shifter of the TW-MZM notebook: a rib
with four doped regions (n_pad | n_rib | p_rib | p_pad) along the in-plane
axis and a metal electrode landing on each outer pad.
"""

from __future__ import annotations

import gdsfactory as gf
import pytest

from gsim.common.cross_section import build_doped_cross_section
from gsim.common.stack.extractor import Layer
from gsim.common.stack.materials import make_doped_materials

CENTER_Y = -20.0
HALF_WIDTH = 0.3
PAD_WIDTH = 0.3
RIB_HEIGHT = 0.22
ELECTRODE_THICKNESS = 0.5
LENGTH_UM = 10.0

REGION_SPANS = {
    "n_pad": (CENTER_Y - HALF_WIDTH - PAD_WIDTH, CENTER_Y - HALF_WIDTH),
    "n_rib": (CENTER_Y - HALF_WIDTH, CENTER_Y),
    "p_rib": (CENTER_Y, CENTER_Y + HALF_WIDTH),
    "p_pad": (CENTER_Y + HALF_WIDTH, CENTER_Y + HALF_WIDTH + PAD_WIDTH),
}
ELECTRODE_SPANS = {
    "cathode_metal": REGION_SPANS["n_pad"],
    "anode_metal": REGION_SPANS["p_pad"],
}


def build_phase_shifter():
    """Return ``(component, stack)`` for the lateral PN phase shifter."""
    gf.gpdk.PDK.activate()
    comp = gf.Component()
    wg = comp << gf.c.rectangle((LENGTH_UM, 0.4), centered=True, layer=(1, 0))
    wg.y = CENTER_Y
    slab = comp << gf.c.rectangle((LENGTH_UM, 100.0), centered=True, layer=(3, 0))
    slab.y = -5.0

    layer_specs = {}
    for i, (name, (y0, y1)) in enumerate(REGION_SPANS.items()):
        gds_layer = (30, i)
        rect = comp << gf.c.rectangle((LENGTH_UM, y1 - y0), layer=gds_layer)
        rect.y = (y0 + y1) / 2
        layer_specs[name] = Layer(
            name=name,
            gds_layer=gds_layer,
            zmin=0.0,
            zmax=RIB_HEIGHT,
            thickness=RIB_HEIGHT,
            material=name,
            layer_type="dielectric",
            mesh_resolution="fine",
        )
    for j, (name, (y0, y1)) in enumerate(ELECTRODE_SPANS.items()):
        gds_layer = (41, j)
        rect = comp << gf.c.rectangle((LENGTH_UM, y1 - y0), layer=gds_layer)
        rect.y = (y0 + y1) / 2
        layer_specs[name] = Layer(
            name=name,
            gds_layer=gds_layer,
            zmin=RIB_HEIGHT,
            zmax=RIB_HEIGHT + ELECTRODE_THICKNESS,
            thickness=ELECTRODE_THICKNESS,
            material="aluminum",
            layer_type="conductor",
            mesh_resolution="fine",
        )

    materials = make_doped_materials(
        [(name, 1.6e3) for name in REGION_SPANS], permittivity=11.9
    )
    stack, _section = build_doped_cross_section(
        comp,
        axis="x",
        value=0.0,
        substrate_thickness=2.0,
        doping={"layer_specs": layer_specs, "materials": materials},
        verbose=False,
    )
    return comp, stack


@pytest.fixture(scope="module")
def phase_shifter():
    """The lateral PN phase shifter component and its doped stack."""
    return build_phase_shifter()


@pytest.fixture
def device():
    """The device description matching :func:`build_phase_shifter`."""
    from gsim.modulator import Device

    return Device(
        p_regions=["p_rib", "p_pad"],
        n_regions=["n_rib", "n_pad"],
        p_doping_cm3=1e18,
        n_doping_cm3=1e18,
    )


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
