"""What the two EM Stages share: a Staircase, and the Cross-section of it.

The optical and the RF Stage ask different questions, but both answer them
on a Staircase — the Bias point's Carrier map reduced to Strips tiling the
Junction extent, drawn as a component of its own and meshed through the
native ``BoundaryMode`` pipeline. The Strips, the extent they tile, the
substrate under them and the plane cut through the middle of them are the
same decisions in both Stages, so they are made once here.

What differs is the material each Stage adds to a Strip — the optical
wavelength and unperturbed index, or the RF lattice permittivity and top
frequency — and that is the one typed value each Stage answers
:meth:`EMStage.strip_material` with. The coupling itself is the carriers
Stage's, handed to the Staircase whole.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import Field

from gsim.common.stack.staircase import STRIP_LENGTH_UM, StaircaseDrawing
from gsim.modulator.meshing import STAGE_AIRBOX, STAGE_MESH
from gsim.modulator.route import EMRoute
from gsim.modulator.stage import Stage

if TYPE_CHECKING:
    from pathlib import Path

    from gsim.common.stack.staircase import (
        ElectrodeSpec,
        StaircaseCrossSection,
        StripMaterial,
        SurroundingRegion,
    )
    from gsim.palace import BoundaryModeSim
    from gsim.tcad.results import CarrierMap

__all__ = ["EMStage"]


class EMStage(Stage):
    """A Stage that solves an EM Mode on a Staircase.

    Subclasses say what they add to a Strip's material in
    :meth:`strip_material`, and add the settings their own physics needs
    on top of the ones here.

    Attributes:
        route: Backend answering this Stage — ``"femwell"`` (the default)
            or ``"palace"``.
        strip_span: ``(min, max)`` extent the Strips tile along the
            junction axis (um); :meth:`default_strip_span` when unset,
            which each Stage answers for itself.
        substrate_thickness_um: Substrate below the Staircase (um).
        window: In-plane Window (um) the Cross-section is clipped to.
        window_z: Vertical Window (um).
        mesh: Keyword arguments forwarded to the mesh pipeline.
        airbox: Background region around what the Stage meshes.
    """

    route: EMRoute = "femwell"
    strip_span: tuple[float, float] | None = None
    substrate_thickness_um: float = Field(default=2.0, gt=0.0)
    window: tuple[float, float] | None = None
    window_z: tuple[float, float] | None = None
    mesh: dict[str, Any] = Field(default_factory=STAGE_MESH.copy)
    airbox: dict[str, Any] = Field(default_factory=STAGE_AIRBOX.copy)

    def default_strip_span(self) -> tuple[float, float]:
        """The extent this Stage tiles when ``strip_span`` says nothing.

        The Junction extent — the rib — which is what a Staircase
        standing on its own is a model of. A Stage whose Staircase is
        drawn inside the device's own Cross-section wants the doped slab
        instead, and says so by overriding this.

        Returns:
            ``(min, max)`` along the junction axis (um).
        """
        return self._require_study().layout.junction_span.h

    def strip_extent(self, carriers: CarrierMap | None = None) -> tuple[float, float]:
        """``(min, max)`` extent the Strips tile along the junction axis (um).

        Strips cannot outrun the Carrier map they average, so a map
        narrower than the extent asked for (a charge Window clipped
        tighter than the doped Regions) narrows it to what the map
        covers, and says so. That holds whether the extent was derived or
        chosen: the preset chooses it, and a Stage that failed only for
        the callers who said what they wanted would fail for most of
        them.

        Args:
            carriers: The Carrier map the Strips will average, to bound
                the extent by what it covers; unbounded when omitted.

        Returns:
            The extent the Strips tile.
        """
        study = self._require_study()
        if self.strip_span is not None:
            wanted = (float(self.strip_span[0]), float(self.strip_span[1]))
            source = "the extent asked for"
        else:
            wanted = self.default_strip_span()
            source = "the extent derived for this stage"
        if carriers is None:
            return wanted

        from gsim.common.stack.staircase import carrier_map_extent

        covered = carrier_map_extent(carriers, study.layout.junction_span.z)
        clipped = (max(wanted[0], covered[0]), min(wanted[1], covered[1]))
        if clipped != wanted:
            warnings.warn(
                f"The {self.stage_name} stage's strips tile "
                f"{clipped[0]:.4g}..{clipped[1]:.4g} um rather than {source} "
                f"{wanted[0]:.4g}..{wanted[1]:.4g} um: the carrier map only "
                f"covers {covered[0]:.4g}..{covered[1]:.4g} um, and strips "
                "cannot reach past the map they average. The doped silicon "
                "outside them keeps its drawn material and carries no "
                "carrier response. Widen the charge window with "
                "study.charge(window=...), or choose the extent yourself "
                f"with study.{self.stage_name}(strip_span=...).",
                stacklevel=3,
            )
        return clipped

    def strip_material(self) -> StripMaterial:
        """What this Stage adds to a Strip's material.

        The optical wavelength and unperturbed index, or the RF lattice
        permittivity and top frequency: the one typed value the
        Staircase builder takes per Stage. Implemented by each EM Stage.
        """
        raise NotImplementedError

    def build_staircase(
        self,
        carriers: CarrierMap,
        *,
        n_strips: int,
        electrodes: ElectrodeSpec | None,
        surroundings: Sequence[SurroundingRegion] = (),
        span: tuple[float, float] | None = None,
    ) -> StaircaseCrossSection:
        """Reduce a Carrier map to a meshable Staircase.

        The Strips tile :meth:`strip_extent`, sit at the Junction's own
        height, and take the carriers Stage's coupling whole — so both EM
        Stages read the one coupling, evaluated on strip averages — with
        this Stage's own :meth:`strip_material` added.

        Args:
            carriers: The Bias point's Carrier map.
            n_strips: Number of Strips to tile the extent with.
            electrodes: The drawn conductors flanking the Strips, or
                ``None`` for a Staircase carrying none.
            surroundings: The drawn device's own Regions to redraw around
                the Strips; empty leaves the Strips alone in the
                background medium.
            span: The extent to tile, when the caller has already
                resolved it; :meth:`strip_extent` decides otherwise.

        Returns:
            The Staircase Cross-section, drawn on its own component.
        """
        from gsim.common.stack.staircase import build_staircase_cross_section

        study = self._require_study()
        junction = study.layout.junction_span
        return build_staircase_cross_section(
            carriers,
            n_strips=n_strips,
            junction=span if span is not None else self.strip_extent(carriers),
            zmin=junction.z[0],
            zmax=junction.z[1],
            response=study.carriers.response,
            material=self.strip_material(),
            electrodes=electrodes,
            surroundings=surroundings,
            drawing=StaircaseDrawing(substrate_thickness=self.substrate_thickness_um),
        )

    def build_staircase_simulation(
        self,
        staircase: StaircaseCrossSection,
        *,
        output_dir: str | Path,
        freq_hz: float,
        num_modes: int,
        target: float = 0.0,
        window: tuple[float, float] | None = None,
        window_z: tuple[float, float] | None = None,
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
            window: In-plane Window (um) to clip the Staircase to; the
                Stage's own ``window`` when omitted.
            window_z: Vertical Window (um); likewise.

        Returns:
            The configured (unmeshed) ``BoundaryModeSim``.
        """
        from gsim.palace import BoundaryModeSim

        sim = BoundaryModeSim()
        sim.set_output_dir(output_dir)
        sim.set_stack(staircase.stack())
        sim.set_geometry(staircase.component)
        sim.set_airbox(**self.airbox)
        sim.set_cross_section(
            f"x={STRIP_LENGTH_UM / 2.0}",
            window=window if window is not None else self.window,
            window_z=window_z if window_z is not None else self.window_z,
        )
        sim.set_boundary_mode(freq=freq_hz, num_modes=num_modes, target=target)
        return sim
