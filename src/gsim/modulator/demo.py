"""A lateral PN Phase shifter, drawn from nothing, for examples and tests.

This is scaffolding, not an entry point. A real Study is built over a
device the user already drew: their own component, their own layer stack,
and the device description naming which Regions are p and which are n. The
builder here exists so that a notebook or a test which needs *some*
device to talk about does not carry sixty lines of component construction
before it gets to the point.

What it draws is the canonical lateral PN Phase shifter: a silicon rib
split into four doped Regions along the junction axis
(``n_pad | n_rib | p_rib | p_pad``), with a metal electrode landing on
each outer pad. The dimensions are all arguments, so the shape can be
stretched, but nothing about it is calibrated to any foundry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gsim.modulator.device import Device

if TYPE_CHECKING:
    import gdsfactory as gf

    from gsim.common.stack.extractor import LayerStack

__all__ = ["DemoPhaseShifter", "demo_phase_shifter"]

#: Position of the Junction on the layout's junction axis (um).
DEFAULT_CENTER_UM: float = -20.0

#: Width of each junction flank (um).
DEFAULT_HALF_WIDTH_UM: float = 0.3

#: Width of each contact pad (um).
DEFAULT_PAD_WIDTH_UM: float = 0.3

#: Height of the silicon rib (um).
DEFAULT_RIB_HEIGHT_UM: float = 0.22

#: Thickness of the metal over each pad (um).
DEFAULT_ELECTRODE_THICKNESS_UM: float = 0.5

#: Drawn length of the device (um). The solves are all Cross-section
#: physics, so this only has to be long enough to cut a plane from.
DEFAULT_LENGTH_UM: float = 10.0

#: Width of the drawn waveguide core (um).
DEFAULT_WAVEGUIDE_WIDTH_UM: float = 0.4

#: Region names the builder draws, low side of the junction axis first.
REGION_NAMES: tuple[str, str, str, str] = ("n_pad", "n_rib", "p_rib", "p_pad")

#: Electrode Region name over each outer pad, in the same order.
ELECTRODE_NAMES: dict[str, str] = {
    "n_pad": "cathode_metal",
    "p_pad": "anode_metal",
}


@dataclass(frozen=True)
class DemoPhaseShifter:
    """A drawn demo device and the description that interprets it.

    Attributes:
        component: The drawn device.
        stack: The layer stack its Regions are named in.
        device: The device description matching what was drawn, ready to
            hand to :func:`~gsim.modulator.preset.pn_phase_shifter`.
        center_um: Position of the Junction on the layout's junction axis (um).
        half_width_um: Width of each junction flank (um).
        pad_width_um: Width of each contact pad (um).
        rib_height_um: Height of the silicon rib (um).
        electrode_thickness_um: Thickness of the metal over each pad (um).
        waveguide_width_um: Width of the drawn waveguide core (um).
        length_um: Drawn length of the device (um); every solve is
            Cross-section physics, so this only has to be long enough to
            cut a plane from.
    """

    component: gf.Component
    stack: LayerStack
    device: Device
    center_um: float
    half_width_um: float
    pad_width_um: float
    rib_height_um: float
    electrode_thickness_um: float
    waveguide_width_um: float
    length_um: float


def _region_spans(
    *, center_um: float, half_width_um: float, pad_width_um: float
) -> dict[str, tuple[float, float]]:
    """Extent of each doped Region along the junction axis (um)."""
    return {
        "n_pad": (center_um - half_width_um - pad_width_um, center_um - half_width_um),
        "n_rib": (center_um - half_width_um, center_um),
        "p_rib": (center_um, center_um + half_width_um),
        "p_pad": (center_um + half_width_um, center_um + half_width_um + pad_width_um),
    }


def demo_phase_shifter(
    *,
    center_um: float = DEFAULT_CENTER_UM,
    half_width_um: float = DEFAULT_HALF_WIDTH_UM,
    pad_width_um: float = DEFAULT_PAD_WIDTH_UM,
    rib_height_um: float = DEFAULT_RIB_HEIGHT_UM,
    electrode_thickness_um: float = DEFAULT_ELECTRODE_THICKNESS_UM,
    length_um: float = DEFAULT_LENGTH_UM,
    waveguide_width_um: float = DEFAULT_WAVEGUIDE_WIDTH_UM,
    substrate_thickness_um: float = 2.0,
    permittivity: float = 11.9,
    sigma_s_per_m: float = 1.6e3,
    p_doping_cm3: float = 1e18,
    n_doping_cm3: float = 1e18,
) -> DemoPhaseShifter:
    """Draw a lateral PN Phase shifter and describe it.

    Scaffolding for examples and tests. A Study over a real device takes
    the user's own component and stack; this builder is only here so that
    example code has a device to point at.

    Args:
        center_um: Position of the Junction on the layout's junction axis (um).
        half_width_um: Width of each junction flank (um).
        pad_width_um: Width of each contact pad (um).
        rib_height_um: Height of the silicon rib (um).
        electrode_thickness_um: Thickness of the metal over each pad (um).
        length_um: Drawn length of the device (um).
        waveguide_width_um: Width of the drawn waveguide core (um).
        substrate_thickness_um: Substrate below ``z = 0`` (um).
        permittivity: Relative permittivity of the doped silicon.
        sigma_s_per_m: Background conductivity of the doped Regions (S/m).
        p_doping_cm3: Acceptor concentration of the p Regions (cm^-3).
        n_doping_cm3: Donor concentration of the n Regions (cm^-3).

    Returns:
        The drawn component, its stack, the matching device description,
        and the dimensions they were drawn with.
    """
    import gdsfactory as gf

    from gsim.common.cross_section import build_doped_cross_section
    from gsim.common.stack.extractor import Layer
    from gsim.common.stack.materials import make_doped_materials

    gf.gpdk.PDK.activate()
    component = gf.Component()
    waveguide = component << gf.c.rectangle(
        (length_um, waveguide_width_um), centered=True, layer=(1, 0)
    )
    waveguide.y = center_um
    slab = component << gf.c.rectangle((length_um, 100.0), centered=True, layer=(3, 0))
    slab.y = -5.0

    spans = _region_spans(
        center_um=center_um,
        half_width_um=half_width_um,
        pad_width_um=pad_width_um,
    )
    layer_specs: dict[str, Layer] = {}
    for index, name in enumerate(REGION_NAMES):
        low, high = spans[name]
        gds_layer = (30, index)
        rect = component << gf.c.rectangle((length_um, high - low), layer=gds_layer)
        rect.y = 0.5 * (low + high)
        layer_specs[name] = Layer(
            name=name,
            gds_layer=gds_layer,
            zmin=0.0,
            zmax=rib_height_um,
            thickness=rib_height_um,
            material=name,
            layer_type="dielectric",
            mesh_resolution="fine",
        )
    for index, (pad, electrode) in enumerate(ELECTRODE_NAMES.items()):
        low, high = spans[pad]
        gds_layer = (41, index)
        rect = component << gf.c.rectangle((length_um, high - low), layer=gds_layer)
        rect.y = 0.5 * (low + high)
        layer_specs[electrode] = Layer(
            name=electrode,
            gds_layer=gds_layer,
            zmin=rib_height_um,
            zmax=rib_height_um + electrode_thickness_um,
            thickness=electrode_thickness_um,
            material="aluminum",
            layer_type="conductor",
            mesh_resolution="fine",
        )

    materials = make_doped_materials(
        [(name, sigma_s_per_m) for name in REGION_NAMES],
        permittivity=permittivity,
    )
    stack, _section = build_doped_cross_section(
        component,
        axis="x",
        value=0.0,
        substrate_thickness=substrate_thickness_um,
        doping={"layer_specs": layer_specs, "materials": materials},
        verbose=False,
    )

    device = Device(
        p_regions=["p_rib", "p_pad"],
        n_regions=["n_rib", "n_pad"],
        p_doping_cm3=p_doping_cm3,
        n_doping_cm3=n_doping_cm3,
    )
    return DemoPhaseShifter(
        component=component,
        stack=stack,
        device=device,
        center_um=center_um,
        half_width_um=half_width_um,
        pad_width_um=pad_width_um,
        rib_height_um=rib_height_um,
        electrode_thickness_um=electrode_thickness_um,
        waveguide_width_um=waveguide_width_um,
        length_um=length_um,
    )
