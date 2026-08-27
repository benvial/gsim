"""gsim.modulator — the electro-optic modulator workflow, end to end.

One device description drives every Stage of a traveling-wave modulator
Study: charge transport through the Phase shifter, and (as they land) the
carrier coupling, the optical and RF Modes, and the line figures of merit.
Each Stage is configured through its own callable section, derives its own
Cross-section Window, and caches its result until something upstream of it
changes.

Backends are optional installs: importing this package never requires
them. The full workflow installs with ``pip install 'gsim[modulator]'``.

Example::

    from gsim.modulator import Device, Study

    study = Study(
        component=comp,
        stack=stack,
        device=Device(
            p_regions=["p_rib", "p_pad"],
            n_regions=["n_rib", "n_pad"],
        ),
    )
    study.charge(biases=[0.0, -1.0, -2.0])
    sweep = study.charge.run()
"""

from gsim.modulator.charge import ChargeStage
from gsim.modulator.device import Device
from gsim.modulator.layout import (
    Contact,
    DeviceLayout,
    Interface,
    Span,
    derive_layout,
)
from gsim.modulator.stage import Stage, StageNotRunError
from gsim.modulator.study import Study

__all__ = [
    "ChargeStage",
    "Contact",
    "Device",
    "DeviceLayout",
    "Interface",
    "Span",
    "Stage",
    "StageNotRunError",
    "Study",
    "derive_layout",
]
