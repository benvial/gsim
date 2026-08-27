"""The charge Stage: Poisson + drift-diffusion across a Bias sweep.

The Stage assembles a :class:`gsim.tcad.ChargeTransportSim` from the
device description — its Window, its Contacts, its Interfaces and its
doping profiles all derived — meshes it once, and sweeps the bias. The
result is the backend's own Bias sweep, so nothing downstream has to learn
a second result type.

DEVSIM is optional: the Stage checks for it before it meshes, so a missing
extra costs nothing but the error message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field, PrivateAttr

from gsim.modulator.stage import Stage
from gsim.tcad.runtime import require_devsim

if TYPE_CHECKING:
    from gsim.tcad.results import BiasSweepResult
    from gsim.tcad.sim import ChargeTransportSim

__all__ = ["ChargeStage"]


class ChargeStage(Stage):
    """Charge transport through the Phase shifter, bias point by bias point.

    Attributes:
        biases: Bias voltages applied to the swept Contact (V).
        contact: Swept Contact; defaults to the Contact on the n side, so
            positive voltages reverse-bias the Junction.
        window: In-plane charge Window (um); derived from the device
            description when unset.
        window_z: Vertical Window (um); unclipped when unset.
        mesh: Keyword arguments forwarded to the mesh pipeline.
        airbox: Background region around the clipped domain.
        temperature: Lattice temperature (K).
        settings: Extra settings applied to the charge-transport sim.
    """

    stage_name: ClassVar[str] = "charge"

    #: The charge-transport sim of the last run, kept so its DEVSIM state
    #: is released before the next one is built.
    _sim: Any = PrivateAttr(default=None)

    biases: list[float] = Field(default_factory=lambda: [0.0])
    contact: str | None = None
    window: tuple[float, float] | None = None
    window_z: tuple[float, float] | None = None
    mesh: dict[str, Any] = Field(
        default_factory=lambda: {
            "preset": "coarse",
            "refined_mesh_size": 0.02,
            "max_mesh_size": 40.0,
            "verbose": False,
        }
    )
    airbox: dict[str, Any] = Field(
        default_factory=lambda: {
            "margin_x": 2.0,
            "margin_y": 2.0,
            "z_above": 1.5,
            "z_below": 1.0,
            "material": "sio2",
        }
    )
    temperature: float = Field(default=300.0, gt=0.0)
    settings: dict[str, Any] = Field(default_factory=dict)

    def simulation(self) -> ChargeTransportSim:
        """Assemble the charge-transport sim this Stage would run.

        Contacts, Interfaces, doping profiles and the Window all come from
        the device description; nothing is declared twice.

        Returns:
            The configured (unmeshed) :class:`ChargeTransportSim`.
        """
        from gsim.tcad.doping import StepDoping
        from gsim.tcad.sim import ChargeTransportSim

        study = self._require_study()
        layout = study.layout
        device = study.device

        sim = ChargeTransportSim(temperature=self.temperature, **self.settings)
        sim.set_output_dir(study.stage_dir(self.stage_name))
        sim.set_stack(study.stack)
        sim.set_geometry(study.component)
        sim.set_airbox(**self.airbox)
        sim.set_cross_section(
            study.plane,
            window=self.window if self.window is not None else layout.window,
            window_z=self.window_z,
        )

        for contact in layout.contacts:
            sim.add_contact(
                name=contact.name,
                layer_a=contact.region,
                layer_b=contact.electrode,
            )
        for interface in layout.interfaces:
            sim.add_interface(
                name=interface.name,
                layer_a=interface.regions[0],
                layer_b=interface.regions[1],
            )
        if device.doping is not None:
            for profile in device.doping:
                sim.add_doping(profile)
        else:
            for region in device.doped_regions:
                sim.add_doping(
                    StepDoping(
                        region=region,
                        dopant_type=device.dopant_type(region),
                        concentration_cm3=device.concentration_cm3(region),
                    )
                )
        return sim

    def swept_contact(self) -> str:
        """Name of the Contact the sweep drives.

        Returns:
            The configured Contact, or the n-side one by default.
        """
        if self.contact is not None:
            return self.contact
        return str(self._require_study().layout.contact_on("n").name)

    def _solve(self) -> BiasSweepResult:
        """Mesh the charge Window and sweep the bias."""
        require_devsim()
        if self._sim is not None:
            # DEVSIM's device, mesh and circuit namespaces are global: the
            # previous run's device has to go before this one is built.
            self._sim.reset_device()
        sim = self.simulation()
        self._sim = sim
        sim.mesh(**self.mesh)
        return sim.sweep(list(self.biases), contact=self.swept_contact())
