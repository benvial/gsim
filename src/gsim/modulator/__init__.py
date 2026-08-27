"""gsim.modulator — the electro-optic modulator workflow, end to end.

One device description drives every Stage of a traveling-wave modulator
Study: charge transport through the Phase shifter, the carrier coupling
those Bias points imply, the optical and RF Modes, and (as they land) the
line figures of merit. Each Stage is configured through its own
callable section, derives its own Cross-section Window where it meshes at
all, and caches its result until something upstream of it changes.

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
    response = study.carriers.run()
    study.optical(wavelength_um=1.55)
    modes = study.optical.run()
    study.rf(frequencies_hz=[10e9, 40e9], n_strips=5)
    line = study.rf.run()
"""

from gsim.modulator.carriers import (
    CarrierResponse,
    CarrierResponseSweep,
    CarriersStage,
    MaterialResponse,
)
from gsim.modulator.charge import ChargeStage
from gsim.modulator.device import Device
from gsim.modulator.layout import (
    Contact,
    DeviceLayout,
    Interface,
    Span,
    derive_layout,
)
from gsim.modulator.optical import OpticalMode, OpticalStage, OpticalSweep
from gsim.modulator.rf import RFStage
from gsim.modulator.stage import Stage, StageNotRunError
from gsim.modulator.study import Study

__all__ = [
    "CarrierResponse",
    "CarrierResponseSweep",
    "CarriersStage",
    "ChargeStage",
    "Contact",
    "Device",
    "DeviceLayout",
    "Interface",
    "MaterialResponse",
    "OpticalMode",
    "OpticalStage",
    "OpticalSweep",
    "RFStage",
    "Span",
    "Stage",
    "StageNotRunError",
    "Study",
    "derive_layout",
]
