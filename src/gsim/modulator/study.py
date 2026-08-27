"""One device, one set of conditions, every Stage's configuration and result.

A Study holds the component, the layer stack and the device description,
derives from them everything the Stages need, and exposes each Stage as a
callable section (ADR 0001)::

    study = Study(component=comp, stack=stack, device=device)
    study.charge(biases=[0.0, -1.0, -2.0])
    sweep = study.charge.run()

Stages solve lazily and cache: running one twice costs one solve, and
re-configuring a Stage clears its result and every downstream Stage's.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from gsim.modulator.carriers import CarriersStage
from gsim.modulator.charge import ChargeStage
from gsim.modulator.device import Device
from gsim.modulator.layout import DeviceLayout, derive_layout
from gsim.modulator.optical import OpticalStage

if TYPE_CHECKING:
    import gdsfactory as gf

    from gsim.common.stack.extractor import LayerStack
    from gsim.modulator.stage import Stage

__all__ = ["Study"]

#: Stage order; a Stage's result is cleared by any change upstream of it.
STAGE_ORDER: tuple[str, ...] = ("charge", "carriers", "optical")


def _parse_plane(plane: str) -> tuple[Literal["x", "y", "z"], float]:
    """Split a ``"x=<value>"`` plane spec into its axis and coordinate."""
    axis, _, value = plane.partition("=")
    axis = axis.strip().lower()
    if axis not in ("x", "y", "z") or not value:
        raise ValueError(
            f"Invalid cross-section plane {plane!r}; use 'x=<value>' or 'y=<value>'."
        )
    return axis, float(value)  # type: ignore[return-value]


class Study:
    """A modulator device investigated across one set of conditions.

    Re-assigning :attr:`device` or :attr:`plane` drops the derived layout
    and every Stage's result, the same way re-configuring a Stage does.

    Attributes:
        component: The drawn device.
        stack: The layer stack its Regions are named in.
        device: The device description every derivation starts from.
        plane: Cross-section plane spec (e.g. ``"x=0"``).
        verbose: Print one line per Stage on entry and exit.
        charge: The charge Stage section.
        carriers: The carrier-coupling Stage section.
        optical: The optical-Mode Stage section.
    """

    def __init__(
        self,
        *,
        component: gf.Component,
        stack: LayerStack,
        device: Device | dict[str, Any],
        plane: str = "x=0",
        output_dir: str | Path | None = None,
        verbose: bool = False,
    ) -> None:
        """Build a Study over one device.

        Args:
            component: The drawn device.
            stack: The layer stack its Regions are named in.
            device: The device description (or its keyword arguments).
            plane: Cross-section plane spec, e.g. ``"x=0"``.
            output_dir: Directory for meshes and solver files; a temporary
                directory is used when omitted.
            verbose: Print one line per Stage on entry and exit.
        """
        self.component = component
        self.stack = stack
        self._device = (
            device if isinstance(device, Device) else Device.model_validate(device)
        )
        self._plane = plane
        self.verbose = verbose
        self._output_dir = Path(output_dir) if output_dir is not None else None
        self._layout: DeviceLayout | None = None

        self.charge = ChargeStage()
        self.carriers = CarriersStage()
        self.optical = OpticalStage()
        self._wire_stages()

    # ------------------------------------------------------------------
    # Stage wiring
    # ------------------------------------------------------------------

    @property
    def stages(self) -> dict[str, Stage]:
        """Every Stage of this Study, in dependency order."""
        return {name: getattr(self, name) for name in STAGE_ORDER}

    def _wire_stages(self) -> None:
        """Attach each Stage to this Study and to the Stages after it."""
        stages = self.stages
        ordered = list(stages.values())
        for index, stage in enumerate(ordered):
            stage.wire(
                study=self,
                downstream=ordered[index + 1 :],
                is_verbose=lambda: self.verbose,
            )

    def invalidate(self) -> None:
        """Drop the derived layout and every Stage's result."""
        self._layout = None
        for stage in self.stages.values():
            stage.invalidate()

    # ------------------------------------------------------------------
    # Device description
    # ------------------------------------------------------------------

    @property
    def device(self) -> Device:
        """The device description every derivation starts from."""
        return self._device

    @device.setter
    def device(self, value: Device | dict[str, Any]) -> None:
        """Replace the description, dropping everything derived from it."""
        self._device = (
            value if isinstance(value, Device) else Device.model_validate(value)
        )
        self.invalidate()

    @property
    def plane(self) -> str:
        """Cross-section plane spec, e.g. ``"x=0"``."""
        return self._plane

    @plane.setter
    def plane(self, value: str) -> None:
        """Move the Cross-section, dropping everything derived from it."""
        _parse_plane(value)
        self._plane = value
        self.invalidate()

    # ------------------------------------------------------------------
    # Derived device
    # ------------------------------------------------------------------

    @property
    def layout(self) -> DeviceLayout:
        """Contacts, Interfaces, the Junction and the charge Window.

        Derived from the device description against the drawn
        Cross-section, and cached.
        """
        if self._layout is None:
            axis, value = _parse_plane(self.plane)
            self._layout = derive_layout(
                self.component,
                self.stack,
                self.device,
                axis=axis,
                value=value,
            )
        return self._layout

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @property
    def output_dir(self) -> Path:
        """Directory holding the Study's meshes and solver files."""
        if self._output_dir is None:
            self._output_dir = Path(tempfile.mkdtemp(prefix="gsim-modulator-"))
        self._output_dir.mkdir(parents=True, exist_ok=True)
        return self._output_dir

    def stage_dir(self, stage_name: str) -> Path:
        """Directory a Stage writes into.

        Args:
            stage_name: Name of the Stage.

        Returns:
            The (created) per-Stage output directory.
        """
        path = self.output_dir / stage_name
        path.mkdir(parents=True, exist_ok=True)
        return path
