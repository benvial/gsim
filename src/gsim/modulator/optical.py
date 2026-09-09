"""The optical Stage: the carrier-perturbed Mode at every Bias point.

The Stage answers one question — what does the Phase shifter's optical
Mode do as the bias moves — and it answers it on a Window of its own. A
Window sized to span the Contacts is not a Window that contains a 1.55 um
Mode, so the optical Stage derives a box around the rib instead, meshes
it, and carries the Carrier maps of the charge solve onto that mesh with
an explicit transfer (ADR 0002). Nothing here reuses the charge mesh.

Each Bias point becomes one complex effective index, from which the two
numbers a designer wants follow: the index shift relative to zero bias,
which sets modulation efficiency, and the bias-dependent loss.

Either Backend can answer, and the choice changes how the carriers reach
the solver. femwell — the default Route — carries a continuous
``eps(x, y)`` projected onto the mesh elements of the drawn device.
Palace takes piecewise-constant materials per Region and nothing else, so
selecting it moves the Stage onto a Staircase: the doped silicon replaced
by Strips tiling it, and the rest of the drawn Cross-section — the slab,
the metal on the pads, whatever else the plane crosses — redrawn around
them, in the same Window. Asking for a strip count puts the femwell Route
on that same Staircase too, which is what makes the two Routes comparable
at all: they differ then in their materials and in nothing else.

The selected Route's runtime is checked before the Stage meshes, so a
missing extra or a missing binary costs nothing but the error message.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from gsim.common.stack.staircase import DEFAULT_SI_INDEX
from gsim.modulator.em import EMStage
from gsim.modulator.route import DEFAULT_PALACE_STRIPS, require_route

if TYPE_CHECKING:
    from pathlib import Path

    import meshio

    from gsim.common.stack.staircase import (
        StaircaseCrossSection,
        SurroundingRegion,
    )
    from gsim.modulator.carriers import CarrierResponse, CarrierResponseSweep
    from gsim.palace import BoundaryModeSim
    from gsim.tcad.results import CarrierMap

__all__ = ["OpticalMode", "OpticalStage", "OpticalSweep"]

#: dB per cm of propagation loss per unit of ``|Im(n_eff)| / lambda[cm]``.
_DB_PER_CM = 40.0 * np.pi / np.log(10.0)


class OpticalMode(BaseModel):
    """The Phase shifter's optical Mode at one Bias point.

    Attributes:
        bias_v: Applied bias on the swept Contact (V).
        n_eff: Complex effective index (``exp(+i omega t)``: a lossy Mode
            has ``Im(n_eff) < 0``).
        index_shift: ``Re(n_eff)`` minus its value at the reference bias.
        loss_db_cm: Propagation loss at this bias (dB/cm).
        boundary_field_ratio: Peak field at the Window boundary over peak
            field overall; large values mean the Window is too small.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bias_v: float
    n_eff: complex
    index_shift: float
    loss_db_cm: float
    boundary_field_ratio: float


class OpticalSweep(BaseModel):
    """The optical Mode across the whole Bias sweep.

    Attributes:
        contact: The Contact the charge sweep drove.
        wavelength_um: Vacuum wavelength the Modes were solved at (um).
        reference_bias_v: The bias the index shift is measured from.
        points: One solved Mode per Bias point, in sweep order.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    contact: str
    wavelength_um: float
    reference_bias_v: float
    points: list[OpticalMode] = Field(default_factory=list)

    @property
    def voltages(self) -> NDArray[np.float64]:
        """Applied biases (V) in sweep order."""
        return np.asarray([p.bias_v for p in self.points], dtype=np.float64)

    @property
    def n_eff(self) -> NDArray[np.complex128]:
        """Complex effective indices in sweep order."""
        return np.asarray([p.n_eff for p in self.points], dtype=np.complex128)

    @property
    def index_shift(self) -> NDArray[np.float64]:
        """Index shift from the reference bias, in sweep order."""
        return np.asarray([p.index_shift for p in self.points], dtype=np.float64)

    @property
    def loss_db_cm(self) -> NDArray[np.float64]:
        """Propagation loss (dB/cm) in sweep order."""
        return np.asarray([p.loss_db_cm for p in self.points], dtype=np.float64)


class OpticalStage(EMStage):
    """The carrier-perturbed optical Mode, bias point by bias point.

    The two representations are not interchangeable, and what the
    Staircase costs is a strip count. On the demo Phase shifter at
    1.55 um, against the drawn device solved with a continuous
    ``eps(x, y)``: two Strips land within 0.3% on ``Re n_eff`` but 59% on
    the index shift, sixteen Strips within 0.01% and 2%. The index shift
    is the quantity ``VpiL`` is computed from and by far the slower of
    the two to converge — the index is a whole guide's worth of material
    and the shift is a sliver's — so a Staircase chosen for ``VpiL``
    wants many more Strips than one chosen for the index. Both errors
    shrink with the strip count, which is the whole claim, and what
    ``tests/modulator/test_representation_gate.py`` holds the Route to.

    Attributes:
        route: Backend answering this Stage — ``"femwell"`` (the default)
            or ``"palace"``. The Palace Route cannot express a continuous
            permittivity, so selecting it puts the Stage on a Staircase
            (see ``n_strips``) and leaves the Window-containment ratio
            NaN, Palace's results carrying no mode fields.
        n_strips: Number of Strips the carrier response is reduced to.
            ``None`` — the default — keeps the continuous ``eps(x, y)``
            of the drawn device, which only the femwell Route can solve;
            the Palace Route staircases with
            :data:`~gsim.modulator.route.DEFAULT_PALACE_STRIPS` instead.
            Setting a count puts either Route on the Staircase, so the
            two solve the identical problem.
        strip_span: ``(min, max)`` extent the Strips tile along the
            junction axis (um); the doped slab — every doped Region, pads
            included — when unset, which is both the guide's core and the
            widest extent the Carrier map covers. Unused when the
            continuous profile is solved.
        strip_index: Unperturbed refractive index of the Strips, which
            the plasma dispersion perturbs. ``None`` — the default —
            reads it off the drawn Junction's own material at
            ``wavelength_um``, which is what the continuous profile
            perturbs, so the two representations start from one index.
            Unused when the continuous profile is solved.
        substrate_thickness_um: Substrate below the Staircase (um).
            Unused when the continuous profile is solved.
        wavelength_um: Vacuum wavelength of the solve (um).
        num_modes: Number of Modes to solve at each Bias point; the
            slowest propagating one is reported.
        window: In-plane optical Window (um); derived as a box around the
            rib when unset. A Staircase is clipped to the same Window as
            the continuous profile, so the two Routes mesh one domain.
        window_z: Vertical Window (um); derived from the guiding layer
            when unset.
        mode_margin_um: Half-width of the derived Window either side of
            the Junction (um).
        z_above_um: Margin above the guiding layer in the derived vertical
            Window (um).
        z_below_um: Margin below it (um).
        perturbed_regions: Regions whose permittivity the Carrier maps
            perturb; defaults to the device's doped Regions.
        min_index: Lower bound on ``Re(n_eff)`` for a Mode to count as
            guided; raise it to the cladding index to reject radiation
            Modes.
        boundary_field_tol: Warn above this boundary-field ratio, the sign
            of a Window too small for the Mode.
        metallic_boundaries: Enforce a perfect conductor on the domain
            boundary. On by default because the drawn metal inside an
            optical Window is part of that boundary: a conductor is
            meshed as an outline with its interior left out (ADR 0003),
            and femwell applies this one condition to every facet at
            once, so turning it off leaves the electrodes as open slots
            while the Palace Route still reads them as perfect
            conductors — the two Routes would solve different
            boundary-value problems. The outer wall takes the same
            condition, which a contained Mode does not notice; the
            containment check is what says whether it is contained.
        strip_field_tol: Warn above this fraction of the Mode's power
            falling outside the Strip extent — the sign of Strips too
            narrow to carry the carrier response where the Mode actually
            is. Measured only on the femwell Route, Palace's results
            carrying no mode fields.
        order: Finite-element order of the mode solve.
        n_guess: Effective-index guess centering the eigenvalue search.
    """

    stage_name: ClassVar[str] = "optical"
    stack_kind: ClassVar[Literal["rf", "optical"]] = "optical"

    n_strips: int | None = Field(default=None, ge=1)
    strip_index: float | None = Field(default=None, gt=0.0)
    wavelength_um: float = Field(default=1.55, gt=0.0)
    num_modes: int = Field(default=1, ge=1)
    mode_margin_um: float = Field(default=2.0, gt=0.0)
    z_above_um: float = Field(default=1.0, ge=0.0)
    z_below_um: float = Field(default=1.0, ge=0.0)
    perturbed_regions: list[str] | None = None
    min_index: float = Field(default=1.0, ge=0.0)
    boundary_field_tol: float = Field(default=0.01, gt=0.0)
    strip_field_tol: float = Field(default=0.5, gt=0.0, le=1.0)
    metallic_boundaries: bool = True
    order: int = Field(default=1, ge=1)
    n_guess: float | None = None

    # ------------------------------------------------------------------
    # Derivation
    # ------------------------------------------------------------------

    def default_strip_span(self) -> tuple[float, float]:
        """The doped slab — every doped Region, pads included.

        This Staircase is drawn inside the device's own Cross-section
        (ADR 0004), so its Strips stand for the doped silicon and nothing
        else. Tiling the rib alone would leave the pads as unperturbed
        drawn silicon in the middle of the guide, and the doped slab is
        both the guide's core and the widest extent the Carrier map
        covers.

        Returns:
            ``(min, max)`` along the junction axis (um).
        """
        return self._require_study().layout.doped_span

    def mode_window(self) -> tuple[float, float]:
        """The in-plane Window a Mode of this Stage is solved in (um).

        Returns:
            The configured ``window``, or a box around the Junction
            ``mode_margin_um`` wide either side (ADR 0002).
        """
        if self.window is not None:
            return self.window
        return self._require_study().layout.window_around_junction(
            margin_um=self.mode_margin_um
        )

    def mode_window_z(self) -> tuple[float, float]:
        """The vertical Window a Mode of this Stage is solved in (um).

        Returns:
            The configured ``window_z``, or the guiding layer cleared by
            ``z_above_um`` and ``z_below_um``.
        """
        if self.window_z is not None:
            return self.window_z
        return self._require_study().layout.window_z_around_guide(
            above_um=self.z_above_um, below_um=self.z_below_um
        )

    def perturbed_region_names(self) -> list[str]:
        """Regions the Carrier maps perturb.

        Returns:
            The configured Regions, or the device's doped Regions.
        """
        if self.perturbed_regions is not None:
            return list(self.perturbed_regions)
        return list(self._require_study().device.doped_regions)

    def _regions_on_mesh(self, available: Iterable[str]) -> list[str]:
        """The perturbed Regions the optical mesh actually carries.

        The optical Window is a box around the rib, so a doped Region far
        enough from the Junction — a contact pad, usually — is simply not
        on this mesh, and is left unperturbed rather than treated as an
        error. A Region named explicitly is a different matter: the user
        asked for it, so its absence is reported.

        Args:
            available: Every 2D Region name on the optical mesh.

        Returns:
            The Regions to perturb, in configured order.

        Raises:
            ValueError: When a named Region is off the mesh, or when the
                Window leaves no perturbed Region on it at all.
        """
        on_mesh = set(available)
        wanted = self.perturbed_region_names()
        if self.perturbed_regions is not None:
            missing = [name for name in wanted if name not in on_mesh]
            if missing:
                raise ValueError(
                    f"Perturbed region(s) {missing} are not on the "
                    f"{self.stage_name} window. On it: {sorted(on_mesh)}."
                )
            return wanted
        present = [name for name in wanted if name in on_mesh]
        if not present:
            raise ValueError(
                f"The {self.stage_name} window contains none of the doped "
                f"regions {wanted}, so the carriers would perturb nothing. "
                f"Widen it with study.{self.stage_name}(mode_margin_um=...) "
                "or set window= explicitly."
            )
        return present

    def effective_n_strips(self) -> int | None:
        """Strip count this Stage will actually solve with.

        Returns:
            The configured count; ``None`` when the continuous
            ``eps(x, y)`` is solved instead, which only the femwell Route
            can do, so the Palace Route falls back to
            :data:`~gsim.modulator.route.DEFAULT_PALACE_STRIPS`.
        """
        if self.n_strips is not None:
            return int(self.n_strips)
        return DEFAULT_PALACE_STRIPS if self.route == "palace" else None

    def unperturbed_index(self) -> float:
        """Refractive index the Strips carry before the carriers move it.

        The continuous profile perturbs each drawn Region's own index; a
        Staircase that starts from a textbook silicon index instead
        differs from it by a constant offset at every bias — small, but
        it is the whole of what separates the two representations once
        the geometry matches.

        Returns:
            The configured ``strip_index``, or the index of the drawn
            Junction's material at this Stage's wavelength.
            :data:`~gsim.common.stack.staircase.DEFAULT_SI_INDEX` stands
            in for a material the stack cannot resolve.
        """
        if self.strip_index is not None:
            return float(self.strip_index)

        from gsim.common.stack.materials import (
            MaterialProperties,
            resolve_material_at_wavelength,
        )

        study = self._require_study()
        # Either side of the metallurgical boundary answers: the two are
        # the same silicon, differing in dopant and not in host index.
        region = study.layout.junction.regions[0]
        layer = study.stack.layers.get(region)
        if layer is None:
            return DEFAULT_SI_INDEX
        overrides = {
            name: (
                props
                if isinstance(props, MaterialProperties)
                else MaterialProperties.model_validate(props)
            )
            for name, props in (study.stack.materials or {}).items()
        }
        resolved = resolve_material_at_wavelength(
            layer.material, self.wavelength_um, overrides=overrides
        )
        if resolved is None or resolved.permittivity_scalar is None:
            return DEFAULT_SI_INDEX
        return float(np.sqrt(float(resolved.permittivity_scalar)))

    def surroundings(
        self, span: tuple[float, float] | None = None
    ) -> tuple[SurroundingRegion, ...]:
        """The drawn device redrawn around the Strips.

        The Strips carry the Carrier map and nothing else, so a Staircase
        made of Strips alone is a silicon wire in the background medium —
        not the drawn guide. Every other Region on the drawn
        Cross-section is cut against the Strip footprint and redrawn
        beside them: the undoped slab the rib sits on, the Traveling-wave
        metal landing on the pads, and whatever else the plane crosses.

        The drawn conductors matter most. Metal inside the optical Window
        is meshed as a perfect conductor either Route honours, and
        omitting it moved this device's index by 0.18 — six times what
        the strip count moves it.

        Args:
            span: The Strip extent the Regions are cut against;
                :meth:`strip_extent` decides when omitted.

        Returns:
            The surrounding Regions, empty when the drawn Cross-section
            has nothing on it but the doped silicon the Strips replace.
        """
        from gsim.common.stack.staircase import surroundings_from_section

        study = self._require_study()
        junction = study.layout.junction_span
        return surroundings_from_section(
            study.section,
            strip_span=span if span is not None else self.strip_extent(),
            strip_z=junction.z,
            stack=study.stack,
        )

    def staircase(self, point: CarrierResponse) -> StaircaseCrossSection:
        """Reduce one Bias point's Carrier map to a meshable Staircase.

        The Strips tile :meth:`strip_extent` — the doped slab unless
        ``strip_span`` says otherwise — and take the plasma-dispersion
        coefficients of the carriers Stage: the same coupling the
        continuous profile reads, evaluated on strip averages instead of
        on mesh elements, and at this Stage's own ``wavelength_um``, so
        both paths carry the same loss. Around the Strips the Staircase
        redraws the device itself (see :meth:`surroundings`), so the two
        representations differ in their materials and not in their
        geometry. No flanking electrodes are invented: the drawn ones are
        already among the surrounding Regions.

        Args:
            point: The Bias point to staircase.

        Returns:
            The Staircase Cross-section, drawn on its own component.

        Raises:
            ValueError: When this Stage is solving the continuous
                profile, so there is no strip count to tile with.
        """
        n_strips = self.effective_n_strips()
        if n_strips is None:
            raise ValueError(
                f"The {self.stage_name} stage is solving the continuous "
                "permittivity, so it builds no staircase. Ask for one with "
                f"study.{self.stage_name}(n_strips=...)."
            )
        span = self.strip_extent(point.carriers)
        return self.build_staircase(
            point.carriers,
            n_strips=n_strips,
            electrodes=None,
            span=span,
            surroundings=self.surroundings(span),
            # The coefficients are the carriers Stage's, but the
            # wavelength they are read at is this Stage's: the model's own
            # is where it was fitted, not where the Mode is solved.
            wavelength_um=self.wavelength_um,
            n0=self.unperturbed_index(),
        )

    def staircase_simulation(
        self, staircase: StaircaseCrossSection, *, output_dir: str | Path
    ) -> BoundaryModeSim:
        """Assemble the Staircase cross-section this Stage meshes.

        The Staircase is a component of its own, so the plane cuts through
        the middle of it rather than through the drawn device. It is
        clipped to the same Window the continuous profile is solved in,
        because the Staircase now carries the same Regions: two Routes
        meshing different domains would not be comparable whatever their
        materials agreed on.

        Args:
            staircase: The Staircase to mesh.
            output_dir: Directory this Bias point's mesh and solver files
                land in; one per point, because the Strip materials move
                with the bias.

        Returns:
            The configured (unmeshed) ``BoundaryModeSim``.
        """
        from scipy.constants import speed_of_light as c0

        sim = self.build_staircase_simulation(
            staircase,
            output_dir=output_dir,
            freq_hz=c0 / (self.wavelength_um * 1e-6),
            num_modes=self.num_modes,
            target=self.n_guess if self.n_guess is not None else 0.0,
            window=self.mode_window(),
            window_z=self.mode_window_z(),
        )
        sim.metallic_boundaries = self.metallic_boundaries
        return sim

    def simulation(self) -> BoundaryModeSim:
        """Assemble the cross-section this Stage meshes.

        The Window is the optical one — a box around the rib derived from
        the Junction, never the charge Stage's slab (ADR 0002) — and the
        mesh lands in the Stage's own output directory.

        Returns:
            The configured (unmeshed) ``BoundaryModeSim``.
        """
        from scipy.constants import speed_of_light as c0

        from gsim.palace import BoundaryModeSim

        study = self._require_study()

        sim = BoundaryModeSim()
        sim.set_output_dir(study.stage_dir(self.stage_name))
        sim.set_stack(study.stack)
        sim.set_geometry(study.component)
        sim.set_airbox(**self.airbox)
        sim.set_cross_section(
            study.plane,
            window=self.mode_window(),
            window_z=self.mode_window_z(),
        )
        sim.set_boundary_mode(
            freq=c0 / (self.wavelength_um * 1e-6), num_modes=self.num_modes
        )
        sim.metallic_boundaries = self.metallic_boundaries
        return sim

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def _element_epsilon(
        self,
        mesh: meshio.Mesh,
        base_epsilon: dict[str, complex],
        carriers: CarrierMap,
    ) -> NDArray[np.complex128]:
        """Per-element permittivity of one Bias point on the optical mesh.

        Every element starts from its Region's stack permittivity; the
        perturbed Regions then take the index and absorption the
        transferred Carrier map implies, so the carrier loss replaces the
        material's own rather than adding to it.
        """
        from gsim.common.carrier_transfer import transfer_carriers
        from gsim.common.carriers import permittivity_perturbation

        study = self._require_study()
        regions = self._regions_on_mesh(base_epsilon)
        transferred = transfer_carriers(
            carriers, mesh, at="elements", fill=0.0, regions=regions
        )
        response = study.carriers.response(
            transferred.electrons_cm3, transferred.holes_cm3
        )

        unmapped = sorted(set(transferred.region) - set(base_epsilon))
        if unmapped:
            raise ValueError(
                f"The optical mesh has element(s) in region(s) {unmapped} with "
                "no permittivity; every 2D region must resolve to a material."
            )
        epsilon = np.asarray(
            [base_epsilon[name] for name in transferred.region],
            dtype=np.complex128,
        )
        wanted = set(regions)
        for index, name in enumerate(transferred.region):
            if name not in wanted:
                continue
            epsilon[index] = permittivity_perturbation(
                n0=float(np.sqrt(base_epsilon[name].real)),
                dn=float(response.index_shift[index]),
                dalpha_cm=float(response.absorption_cm[index]),
                wavelength_um=self.wavelength_um,
            )
        return epsilon

    def _check_strip_coverage(
        self, mode: Any, bias_v: float, span: tuple[float, float]
    ) -> None:
        """Warn when the Mode mostly sits off the carrier-bearing Strips.

        The Strips are the only Regions of a Staircase the Carrier map
        reaches. A Mode whose power is largely outside them is answered
        by the surrounding Regions, which carry the drawn materials and
        no carriers at all — so the index shift the bias sweep reports is
        the shift of whatever fraction of the Mode the Strips do hold.
        """
        from gsim.femwell.adapter import field_fraction_outside

        fraction = field_fraction_outside(mode, span)
        if fraction <= self.strip_field_tol:
            return
        warnings.warn(
            f"The {self.stage_name} stage's staircase at V = {bias_v:g} "
            f"carries {fraction:.1%} of the mode's power outside the strip "
            f"extent {span[0]:.3g}..{span[1]:.3g} um (tolerance "
            f"{self.strip_field_tol:.1%}); only the strips carry the carrier "
            "response, so the index shift is that of the fraction inside "
            f"them. Widen the strips with study.{self.stage_name}"
            "(strip_span=...) — up to what the carrier map covers — and "
            "widen the charge window with study.charge(window=...) to make "
            "a wider span legal.",
            stacklevel=2,
        )

    def _check_conductor_clearance(
        self, surroundings: Sequence[SurroundingRegion]
    ) -> None:
        """Refuse a Staircase whose metal is sliced by the Window.

        A drawn conductor is meshed as an outline with its interior left
        out of the domain (ADR 0003). When the Window cuts through one,
        that outline runs along the Window's own outer wall, and the
        Palace Route's meshing does not survive it — the solver aborts
        rather than reporting anything. The femwell Route meshes it, so
        this is a Route limitation and not a modelling one, which is why
        it is checked here and not in the Staircase.

        Args:
            surroundings: The Regions redrawn around the Strips.

        Raises:
            ValueError: When a conductor crosses the Window boundary on
                either axis.
        """
        window, window_z = self.mode_window(), self.mode_window_z()
        for region in surroundings:
            if region.layer_type not in ("conductor", "via"):
                continue
            for extent, bounds, axis in (
                (region.h, window, "window"),
                (region.z, window_z, "window_z"),
            ):
                inside = extent[0] >= bounds[0] and extent[1] <= bounds[1]
                outside = extent[1] <= bounds[0] or extent[0] >= bounds[1]
                if inside or outside:
                    continue
                raise ValueError(
                    f"The {self.stage_name} stage's palace route cannot "
                    f"solve this staircase: the drawn conductor "
                    f"'{region.name}' spans {extent[0]:.4g}..{extent[1]:.4g} "
                    f"um, which the {axis} {bounds[0]:.4g}..{bounds[1]:.4g} "
                    "um cuts through, so its perfect-conductor outline "
                    "would run along the window's own wall. Widen the "
                    f"window to contain it (study.{self.stage_name}"
                    f"({axis}=...)) or narrow it to leave the conductor "
                    f"out, or solve with study.{self.stage_name}"
                    "(route='femwell'), which meshes it."
                )

    def _check_containment(self, ratio: float, bias_v: float) -> None:
        """Warn when a solved Mode still has field at the Window boundary."""
        if ratio > self.boundary_field_tol:
            warnings.warn(
                f"The {self.stage_name} stage's mode at V = {bias_v:g} still "
                f"carries {ratio:.1%} of its peak field at the window "
                f"boundary (tolerance {self.boundary_field_tol:.1%}); its "
                "effective index is a clipped mode's. Widen the window with "
                f"study.{self.stage_name}(mode_margin_um=...) / "
                "(z_above_um=..., z_below_um=...) or set it explicitly.",
                stacklevel=2,
            )

    def _sweep_from(
        self,
        contact: str,
        solved: list[tuple[float, complex, float]],
    ) -> OpticalSweep:
        """Assemble the sweep both Routes report.

        Args:
            contact: The Contact the charge sweep drove.
            solved: ``(bias_v, n_eff, boundary_field_ratio)`` per Bias
                point, in sweep order.

        Returns:
            The optical sweep, its index shift measured from zero bias
            when the sweep visited it and from its first point otherwise.
        """
        reference_bias, reference_index, _ = next(
            (entry for entry in solved if entry[0] == 0.0), solved[0]
        )
        wavelength_cm = self.wavelength_um * 1e-4
        return OpticalSweep(
            contact=contact,
            wavelength_um=self.wavelength_um,
            reference_bias_v=reference_bias,
            points=[
                OpticalMode(
                    bias_v=bias_v,
                    n_eff=n_eff,
                    index_shift=n_eff.real - reference_index.real,
                    loss_db_cm=_DB_PER_CM * abs(n_eff.imag) / wavelength_cm,
                    boundary_field_ratio=ratio,
                )
                for bias_v, n_eff, ratio in solved
            ],
        )

    def _bias_sweep(self) -> CarrierResponseSweep:
        """The Bias sweep to solve, running the upstream Stages if needed.

        Raises:
            ValueError: When the sweep is empty, so there is no Mode to
                solve.
        """
        responses: CarrierResponseSweep = self._require_study().carriers.run()
        if not responses.points:
            raise ValueError(
                "The bias sweep has no points, so there is no mode to solve. "
                "Configure study.charge(biases=[...])."
            )
        return responses

    def _solve_continuous(self) -> OpticalSweep:
        """Solve the drawn device with a continuous ``eps(x, y)``.

        One mesh serves the whole sweep: the geometry does not move with
        the bias, only the per-element permittivity the Carrier maps
        imply.
        """
        import meshio

        from gsim.common.modes import select_line_mode
        from gsim.femwell.adapter import (
            boundary_field_ratio,
            epsilon_by_region,
            solve_modes,
        )

        study = self._require_study()
        responses = self._bias_sweep()

        sim = self.simulation()
        sim.mesh(**self.mesh)
        mesh_path = sim.mesh_path
        mesh = meshio.read(str(mesh_path))
        base_epsilon = epsilon_by_region(
            mesh, study.stack, wavelength_um=self.wavelength_um
        )

        solved: list[tuple[float, complex, float]] = []
        for point in responses.points:
            modes = solve_modes(
                mesh_path,
                epsilon=self._element_epsilon(mesh, base_epsilon, point.carriers),
                wavelength_um=self.wavelength_um,
                num_modes=self.num_modes,
                order=self.order,
                metallic_boundaries=self.metallic_boundaries,
                n_guess=self.n_guess,
            )
            mode = select_line_mode(modes, min_index=self.min_index)
            ratio = boundary_field_ratio(mode)
            self._check_containment(ratio, point.bias_v)
            solved.append((point.bias_v, complex(mode.n_eff), ratio))
        return self._sweep_from(responses.contact, solved)

    def _staircase_modes(
        self,
        sim: BoundaryModeSim,
        staircase: StaircaseCrossSection,
        *,
        binary: Path | None,
        verbose: bool,
    ) -> Sequence[Any]:
        """Solve one meshed Staircase on the selected Route.

        Both Routes read the Strip materials off the same Staircase and
        the same mesh; they differ only in who does the algebra.

        Args:
            sim: The meshed Staircase simulation.
            staircase: The Staircase it was built from.
            binary: Palace executable, on the Palace Route; ``None`` on
                the femwell Route, which needs none.
            verbose: Stream the Backend's own output.

        Returns:
            Every Mode the Route solved.
        """
        from scipy.constants import speed_of_light as c0

        if self.route == "palace":
            from gsim.modulator.route import palace_binary, solve_palace_modes

            return solve_palace_modes(
                sim,
                freq_hz=c0 / (self.wavelength_um * 1e-6),
                num_modes=self.num_modes,
                binary=palace_binary(binary, stage_name=self.stage_name),
                target=self.n_guess if self.n_guess is not None else 0.0,
                verbose=verbose,
            )

        import meshio

        from gsim.femwell.adapter import epsilon_by_region, solve_modes

        mesh = meshio.read(str(sim.mesh_path))
        modes: Sequence[Any] = solve_modes(
            sim.mesh_path,
            epsilon=epsilon_by_region(
                mesh, staircase.stack("optical"), wavelength_um=self.wavelength_um
            ),
            wavelength_um=self.wavelength_um,
            num_modes=self.num_modes,
            order=self.order,
            metallic_boundaries=self.metallic_boundaries,
            n_guess=self.n_guess,
        )
        return modes

    def _solve_staircase(self, binary: Path | None) -> OpticalSweep:
        """Solve the Staircase of every Bias point, on either Route.

        Each Bias point gets its own Staircase, and its own mesh under its
        own directory: the Strip edges do not move with the bias but the
        Strip materials do, and Palace reads its materials off the meshed
        stack rather than from an array handed in per solve.

        Args:
            binary: Palace executable, on the Palace Route.
        """
        from gsim.common.modes import select_line_mode
        from gsim.modulator.route import (
            containment_unmeasurable,
            mode_boundary_ratio,
        )

        study = self._require_study()
        responses = self._bias_sweep()
        stage_dir = study.stage_dir(self.stage_name)
        verbose = self._is_verbose()
        if self.route == "palace":
            if self.n_strips is None:
                warnings.warn(
                    f"The {self.stage_name} stage's palace route cannot carry "
                    "a continuous permittivity, so it is solving a staircase "
                    f"of {DEFAULT_PALACE_STRIPS} strips instead of the "
                    "continuous eps(x, y) the femwell route would have used. "
                    f"Choose the count with study.{self.stage_name}"
                    "(n_strips=...).",
                    stacklevel=2,
                )
            warnings.warn(containment_unmeasurable(self.stage_name), stacklevel=2)
            warnings.warn(
                f"The {self.stage_name} stage's palace route cannot check "
                "how much of the mode sits outside the strip extent either, "
                "for the same reason: no mode fields come back. Re-solve "
                f"with study.{self.stage_name}(route='femwell') at the same "
                "strip count to have both checks run on the identical "
                "staircase.",
                stacklevel=2,
            )

        solved: list[tuple[float, complex, float]] = []
        for index, point in enumerate(responses.points):
            staircase = self.staircase(point)
            if self.route == "palace":
                self._check_conductor_clearance(staircase.surroundings)
            point_dir = stage_dir / f"bias_{index:02d}"
            point_dir.mkdir(parents=True, exist_ok=True)
            sim = self.staircase_simulation(staircase, output_dir=point_dir)
            sim.mesh(**self.mesh)

            modes = self._staircase_modes(
                sim, staircase, binary=binary, verbose=verbose
            )
            mode = select_line_mode(modes, min_index=self.min_index)
            ratio = mode_boundary_ratio(mode)
            self._check_containment(ratio, point.bias_v)
            if self.route != "palace":
                self._check_strip_coverage(mode, point.bias_v, staircase.strip_span)
            solved.append((point.bias_v, complex(mode.n_eff), ratio))
        return self._sweep_from(responses.contact, solved)

    def _solve(self) -> OpticalSweep:
        """Solve the Mode at every Bias point, on the selected Route."""
        # Before the charge solve and before meshing: a user whose Route
        # cannot run should pay nothing to find that out.
        binary = require_route(self.route, stage_name=self.stage_name)
        if self.effective_n_strips() is None:
            return self._solve_continuous()
        return self._solve_staircase(binary)
