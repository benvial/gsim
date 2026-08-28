"""What the two EM Stages share: a Staircase, and the Cross-section of it.

The optical and the RF Stage ask different questions, but both answer them
on a Staircase — the Bias point's Carrier map reduced to Strips tiling the
Junction extent, drawn as a component of its own and meshed through the
native ``BoundaryMode`` pipeline. The Strips, the extent they tile, the
substrate under them and the plane cut through the middle of them are the
same decisions in both Stages, so they are made once here.

What differs is the material response the Strips carry — the plasma
dispersion at the optical wavelength, or the Drude conductivity up to the
top RF frequency — and that stays with the Stage that knows about it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field

from gsim.common.stack.staircase import STRIP_LENGTH_UM
from gsim.modulator.meshing import STAGE_AIRBOX, STAGE_MESH
from gsim.modulator.route import EMRoute
from gsim.modulator.stage import Stage

if TYPE_CHECKING:
    from pathlib import Path

    from gsim.common.stack.staircase import (
        ElectrodeSpec,
        StaircaseCrossSection,
    )
    from gsim.palace import BoundaryModeSim
    from gsim.tcad.results import CarrierMap

__all__ = ["EMStage"]


class EMStage(Stage):
    """A Stage that solves an EM Mode on a Staircase.

    Subclasses declare which material response of the Staircase they
    stack in :attr:`stack_kind`, and add the settings their own physics
    needs on top of the ones here.

    Attributes:
        route: Backend answering this Stage — ``"femwell"`` (the default)
            or ``"palace"``.
        strip_span: ``(min, max)`` extent the Strips tile along the
            junction axis (um); the Junction extent — the rib — when
            unset.
        substrate_thickness_um: Substrate below the Staircase (um).
        window: In-plane Window (um) the Cross-section is clipped to.
        window_z: Vertical Window (um).
        mesh: Keyword arguments forwarded to the mesh pipeline.
        airbox: Background region around what the Stage meshes.
    """

    #: Which material response of a Staircase this Stage stacks.
    stack_kind: ClassVar[Literal["rf", "optical"]] = "optical"

    route: EMRoute = "femwell"
    strip_span: tuple[float, float] | None = None
    substrate_thickness_um: float = Field(default=2.0, gt=0.0)
    window: tuple[float, float] | None = None
    window_z: tuple[float, float] | None = None
    mesh: dict[str, Any] = Field(default_factory=STAGE_MESH.copy)
    airbox: dict[str, Any] = Field(default_factory=STAGE_AIRBOX.copy)

    def build_staircase(
        self,
        carriers: CarrierMap,
        *,
        n_strips: int,
        electrodes: ElectrodeSpec | None,
        **response: Any,
    ) -> StaircaseCrossSection:
        """Reduce a Carrier map to a meshable Staircase.

        The Strips tile the Junction extent unless ``strip_span`` widens
        them, sit at the Junction's own height, and take the carriers
        Stage's plasma-dispersion coefficients and mobilities — so both
        EM Stages read the one coupling, evaluated on strip averages.

        Args:
            carriers: The Bias point's Carrier map.
            n_strips: Number of Strips to tile the extent with.
            electrodes: The drawn conductors flanking the Strips, or
                ``None`` for a Staircase carrying none.
            **response: What this Stage's own physics adds to the Strip
                materials — the optical wavelength and unperturbed index,
                or the RF permittivity and top frequency.

        Returns:
            The Staircase Cross-section, drawn on its own component.
        """
        from gsim.common.stack.staircase import build_staircase_cross_section

        study = self._require_study()
        span = study.layout.junction_span
        return build_staircase_cross_section(
            carriers,
            n_strips=n_strips,
            junction=self.strip_span if self.strip_span is not None else span.h,
            zmin=span.z[0],
            zmax=span.z[1],
            length=STRIP_LENGTH_UM,
            electrodes=electrodes,
            dispersion=study.carriers.dispersion,
            mu_n_cm2=study.carriers.mu_n_cm2,
            mu_p_cm2=study.carriers.mu_p_cm2,
            axis="x",
            value=STRIP_LENGTH_UM / 2.0,
            substrate_thickness=self.substrate_thickness_um,
            **response,
        )

    def build_staircase_simulation(
        self,
        staircase: StaircaseCrossSection,
        *,
        output_dir: str | Path,
        freq_hz: float,
        num_modes: int,
        target: float = 0.0,
    ) -> BoundaryModeSim:
        """Assemble the Staircase Cross-section this Stage meshes.

        The Staircase is a component of its own — Strips and electrodes
        drawn in the Cross-section's transverse coordinates and extruded
        along the propagation direction — so the plane cuts through the
        middle of it rather than through the drawn device, and the airbox
        rather than a derived Window is what puts cladding around the
        Strips.

        Args:
            staircase: The Staircase to mesh.
            output_dir: Directory the mesh and solver files land in.
            freq_hz: Frequency recorded in the boundary-mode block.
            num_modes: Number of Modes the block asks for.
            target: Effective-index target centering the search.

        Returns:
            The configured (unmeshed) ``BoundaryModeSim``.
        """
        from gsim.palace import BoundaryModeSim

        sim = BoundaryModeSim()
        sim.set_output_dir(output_dir)
        sim.set_stack(staircase.stack(self.stack_kind))
        sim.set_geometry(staircase.component)
        sim.set_airbox(**self.airbox)
        sim.set_cross_section(
            f"x={STRIP_LENGTH_UM / 2.0}",
            window=self.window,
            window_z=self.window_z,
        )
        sim.set_boundary_mode(freq=freq_hz, num_modes=num_modes, target=target)
        return sim
