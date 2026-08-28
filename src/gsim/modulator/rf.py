"""The RF Stage: the traveling-wave line parameters versus frequency.

The Stage answers what the Traveling-wave electrode does to an RF signal
once the Phase shifter's carriers load it. Palace and femwell both take
piecewise-constant materials per Region, so the Carrier map of the chosen
Bias point is reduced to a Staircase — Strips tiling the Junction extent
(or any wider extent asked for), each carrying the Drude conductivity of
its average carrier concentrations, flanked by the electrode's two drawn
conductors — and the electrode-loaded Cross-section is solved once per
requested frequency.

The Staircase is built here rather than by the caller: its Strips, their
materials and the electrodes all follow from the device description and
the carriers Stage's mobilities, so nothing outside assembles a second
component.

Each frequency gives one complex effective index, from which the RF index
and the RF loss follow, and one characteristic impedance from the
Marks-Williams power-current integral over the signal conductor — the
electrode on the Contact the RF drive is applied to, identified from the
device description rather than passed as element indices.

Either Backend can solve the Staircase, and either reports the
impedance: the femwell Route reads its Mode's fields directly, and the
Palace Route reads the selected Mode's saved fields back off disk and
runs the same integral on them. Only femwell can check the Window while
it selects, because that check happens before anything is saved.

What the two Routes can express of the electrode metal differs, and that
is what ``conductor_model`` chooses between (ADR 0003). Meshed as
Regions of conducting material — the ``"volume"`` model — the electrodes
have a permittivity whose imaginary part is of order ``1e7`` at RF, and
Palace's shift-and-invert search returns those metal-dominated modes
rather than the quasi-TEM one. Meshed as perfect conductors — the
``"pec"`` model — their interior is left out of the domain and their
outline carries the boundary condition, which both Routes express
identically and neither is derailed by. The Palace Route therefore
defaults to ``"pec"`` and the femwell Route, which can carry the metal's
own loss, to ``"volume"``.

The selected Route's runtime is checked before the Stage meshes, so a
missing extra or a missing binary costs nothing but the error message.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
from pydantic import Field, PrivateAttr, field_validator

from gsim.common.modes import LineModeRule
from gsim.common.stack.staircase import (
    DEFAULT_ELECTRODES,
    ConductorModel,
    ElectrodeSpec,
)
from gsim.modulator.em import EMStage
from gsim.modulator.route import require_route

if TYPE_CHECKING:
    from pathlib import Path

    from gsim.common.stack.staircase import StaircaseCrossSection
    from gsim.common.twmzm_report import RFLineParams
    from gsim.modulator.carriers import CarrierResponse, CarrierResponseSweep
    from gsim.palace import BoundaryModeSim

__all__ = ["RFStage"]

#: Biases this far apart (V) count as the same Bias point.
BIAS_TOL_V: float = 1e-9

#: Strips the Junction extent should span before the Staircase is warned
#: about: fewer, and the depletion edge sits inside one Strip.
MIN_STRIPS_ACROSS_RIB: int = 2


class RFStage(EMStage):
    """The line parameters of the Traveling-wave electrode, versus frequency.

    Attributes:
        route: Backend answering this Stage — ``"femwell"`` (the default)
            or ``"palace"``. Both solve the same Staircase on the same
            mesh and both report the characteristic impedance; only
            femwell reads its Mode's fields while it selects, so the
            Palace Route reports NaN for the Window-containment ratio.
        conductor_model: How the electrode metal is expressed in the
            mesh (ADR 0003) — ``"volume"`` as Regions of lossy metal,
            ``"pec"`` as perfect conductors whose interior is left out of
            the meshed domain. Unset, each Route takes the model it can
            express: ``"volume"`` on femwell, which carries the metal's
            own loss, and ``"pec"`` on Palace, whose eigenvalue search a
            metal Region takes over. Set it to the same value on both to
            compare them.
        frequencies_hz: RF frequencies to solve at (Hz), kept ascending.
        n_strips: Number of Strips the Carrier map is reduced to; more
            strips approximate the continuous profile more closely at the
            cost of mesh size.
        bias_v: Bias point the Staircase is built from; the last point of
            the Bias sweep when unset.
        strip_span: ``(min, max)`` extent the Strips tile along the
            junction axis (um); the Junction extent — the rib — when
            unset, which leaves the doped pads out of the Staircase and
            so leaves out the series resistance they carry, the
            electrodes standing directly against the rib.
            :func:`~gsim.modulator.preset.pn_phase_shifter` therefore
            widens it to the doped slab
            (:attr:`~gsim.modulator.layout.DeviceLayout.doped_span`).
            The Strips are of equal width, so on a device whose pads are
            much wider than the rib, widening dilutes the resolution
            around the Junction; the Stage says so when the rib stops
            spanning :data:`MIN_STRIPS_ACROSS_RIB` Strips, and the answer
            is to raise ``n_strips`` with the span.
        window: In-plane RF Window (um); the full Cross-section extent
            when unset.
        window_z: Vertical RF Window (um); the full extent when unset.
        electrodes: The drawn conductors of the Traveling-wave
            electrode, flanking the Strips.
        signal_contact: Contact the RF drive is applied to, naming the
            signal conductor; the charge Stage's swept Contact when
            unset.
        num_modes: Number of Modes solved at each frequency, out of which
            the physical line Mode is selected.
        rule: Candidate rule replacing the default one of
            :func:`gsim.common.modes.select_line_mode`.
        min_index: Lower bound on ``Re(n_eff)`` for the default rule.
        max_loss_ratio: Upper bound on ``|Im(n_eff)| / Re(n_eff)`` for
            the default rule. Tighter here than the bound
            :func:`gsim.common.modes.select_line_mode` falls back to,
            because a Traveling-wave electrode is a transmission line
            rather than an arbitrary waveguide: it advances several
            radians of phase per radian of loss, and the Modes sitting
            just inside a bound of one are the discretization's, not the
            line's.
        degeneracy_rtol: Relative spread within which two candidate Modes
            count as ambiguous, and the selection warns.
        strip_permittivity: Relative permittivity of the Strip lattice,
            which the carrier conductivity loads.
        substrate_thickness_um: Substrate below the Staircase (um).
        boundary_field_tol: Warn above this boundary-field ratio, the
            sign of a Window squeezing the Mode (ADR 0002). Far looser
            than an optical Window's: the outer boundary is metallic and
            carries the line's return field, so some field there is the
            structure rather than a clipped tail.
        metallic_boundaries: Enforce PEC on the outer boundary — the
            usual condition for a shielded line solve. A femwell-Route
            setting: nothing in the Palace pipeline expresses it, and
            Palace's own default for the outer boundary is the opposite
            condition, so the Palace Route says so rather than pretending
            the Window is shielded.
        order: Finite-element order of the mode solve. Under the
            ``"pec"`` conductor model the impedance is read off the field
            around the electrode rather than out of it, and femwell
            derives that field from the curl of its solution — so the
            femwell Route wants order 2 there, and says so at order 1.
        n_guess: Effective-index guess centering the eigenvalue search;
            RF materials have large conductive ``|Im(eps)|``, so the
            default guess is the slow-wave index rather than the solver's.
        track_modes: Follow the line Mode across the sweep, guessing each
            frequency from the index solved at the one before it, so a
            shift-and-invert search anchored to one number cannot settle
            on a different branch as the materials move with frequency.
            Turn it off to aim every frequency at ``n_guess`` instead.
        jump_rtol: How far ``Re(n_eff)`` may step between neighbouring
            frequencies, per unit of relative frequency step, before the
            sweep is reported as having changed branch rather than
            dispersed. Scaling by the frequency step is what lets one
            number serve a sparse sweep and a dense one: a doubling in
            frequency is allowed ``jump_rtol`` of index, a 1% step a
            hundredth of it.
    """

    stage_name: ClassVar[str] = "rf"
    stack_kind: ClassVar[Literal["rf", "optical"]] = "rf"

    #: Bias the cached result was solved at, read through
    #: :attr:`solved_bias_v`.
    _solved_bias_v: float | None = PrivateAttr(default=None)

    frequencies_hz: list[float] = Field(
        default_factory=lambda: [10e9, 40e9], min_length=1
    )
    n_strips: int = Field(default=5, ge=1)
    bias_v: float | None = None
    conductor_model: ConductorModel | None = None
    electrodes: ElectrodeSpec = Field(default=DEFAULT_ELECTRODES)
    signal_contact: str | None = None
    num_modes: int = Field(default=4, ge=1)
    rule: LineModeRule | None = None
    min_index: float = Field(default=1.0, ge=0.0)
    max_loss_ratio: float = Field(default=0.5, gt=0.0)
    degeneracy_rtol: float = Field(default=0.03, gt=0.0)
    strip_permittivity: float = Field(default=11.9, gt=0.0)
    boundary_field_tol: float = Field(default=0.2, gt=0.0)
    metallic_boundaries: bool = True
    order: int = Field(default=1, ge=1)
    n_guess: float | None = 3.0
    track_modes: bool = True
    jump_rtol: float = Field(default=0.25, gt=0.0)

    @field_validator("frequencies_hz")
    @classmethod
    def _ascending_and_positive(cls, value: list[float]) -> list[float]:
        """Frequencies are positive, and travel in ascending order."""
        if any(freq <= 0.0 for freq in value):
            raise ValueError("RF frequencies must be positive.")
        return sorted(float(freq) for freq in value)

    # ------------------------------------------------------------------
    # Derivation
    # ------------------------------------------------------------------

    def bias_point(self) -> CarrierResponse:
        """The Bias point the Staircase is built from.

        Runs the carriers Stage (and, through it, the charge Stage) if
        they have not run.

        Returns:
            The chosen Bias point's carrier response.

        Raises:
            ValueError: When the sweep is empty, or has no point at the
                configured bias.
        """
        responses: CarrierResponseSweep = self._require_study().carriers.run()
        if not responses.points:
            raise ValueError(
                "The bias sweep has no points, so there is no carrier map to "
                "staircase. Configure study.charge(biases=[...])."
            )
        if self.bias_v is None:
            return responses.points[-1]
        for point in responses.points:
            if abs(point.bias_v - self.bias_v) <= BIAS_TOL_V:
                return point
        visited = ", ".join(f"{point.bias_v:g}" for point in responses.points)
        raise ValueError(
            f"The bias sweep has no point at V = {self.bias_v:g}; it visited "
            f"{visited} V. Solve that bias with study.charge(biases=[...]) or "
            f"pick one of them with study.{self.stage_name}(bias_v=...)."
        )

    def signal_contact_name(self) -> str:
        """Name of the Contact the RF drive is applied to.

        Returns:
            The configured Contact, or the one the charge sweep drives.
        """
        if self.signal_contact is not None:
            return self.signal_contact
        return str(self._require_study().charge.swept_contact())

    def signal_electrode(self) -> str:
        """Region name of the Staircase electrode carrying the signal.

        The Contact the drive is applied to sits on one side of the
        Junction; the electrode flanking that side is the signal
        conductor, and the other one is the return.

        Returns:
            The electrode Region name.

        Raises:
            ValueError: When no Contact of the device has that name.
        """
        layout = self._require_study().layout
        wanted = self.signal_contact_name()
        matches = [contact for contact in layout.contacts if contact.name == wanted]
        if not matches:
            raise ValueError(
                f"The device has no contact named '{wanted}'; its contacts are "
                f"{[contact.name for contact in layout.contacts]}. Name the "
                f"driven one with study.{self.stage_name}(signal_contact=...)."
            )
        low_side = layout.is_below_junction(matches[0].region)
        return self.electrodes.names[0 if low_side else 1]

    def effective_conductor_model(self) -> ConductorModel:
        """How this run expresses the electrode metal (ADR 0003).

        Returns:
            The configured model, or the one the selected Route can
            express: ``"volume"`` on femwell, which carries the metal's
            own conductivity, and ``"pec"`` on Palace, whose eigenvalue
            search returns a metal Region's own modes rather than the
            line's.
        """
        if self.conductor_model is not None:
            return self.conductor_model
        return "pec" if self.route == "palace" else "volume"

    def staircase(self) -> StaircaseCrossSection:
        """Reduce the Bias point's Carrier map to a meshable Staircase.

        The Strips tile the Junction extent — the two Regions the
        metallurgical boundary separates — unless ``strip_span`` widens
        them, and take the plasma-dispersion coefficients and mobilities
        of the carriers Stage, so the RF conductivity is the same
        coupling the optical Stage reads.

        The Strip materials are valid up to the highest requested
        frequency, which is what the solve asks of them, and the
        electrodes flanking them take this run's conductor model.

        Returns:
            The Staircase Cross-section, drawn on its own component.
        """
        self._check_strip_resolution()
        return self.build_staircase(
            self.bias_point().carriers,
            n_strips=self.n_strips,
            electrodes=replace(
                self.electrodes, conductor_model=self.effective_conductor_model()
            ),
            permittivity=self.strip_permittivity,
            fmax=max(self.frequencies_hz),
        )

    def _check_strip_resolution(self) -> None:
        """Warn when the Strips are too wide to resolve the Junction.

        Strips are of equal width, so tiling an extent wider than the rib
        — the doped slab, which is what carries the pads' series
        resistance into the line — buys that resistance at the cost of
        resolution where the carriers actually move. Below two Strips
        across the rib the depletion edge is inside a single Strip and the
        Staircase has stopped resolving what it exists for.
        """
        span = self._require_study().layout.junction_span
        extent = self.strip_span if self.strip_span is not None else span.h
        strip_width = (extent[1] - extent[0]) / self.n_strips
        rib_width = span.h[1] - span.h[0]
        if strip_width * MIN_STRIPS_ACROSS_RIB <= rib_width:
            return
        wanted = int(
            np.ceil(MIN_STRIPS_ACROSS_RIB * (extent[1] - extent[0]) / rib_width)
        )
        warnings.warn(
            f"The {self.stage_name} stage tiles {extent[1] - extent[0]:.3g} um "
            f"with {self.n_strips} strips of {strip_width:.3g} um, so the "
            f"{rib_width:.3g} um junction extent falls inside fewer than "
            f"{MIN_STRIPS_ACROSS_RIB} of them and the carrier profile across "
            "it is not resolved. Raise the count with "
            f"study.{self.stage_name}(n_strips={wanted}) or narrow the span "
            f"with study.{self.stage_name}(strip_span=...).",
            stacklevel=2,
        )

    def simulation(
        self, staircase: StaircaseCrossSection | None = None
    ) -> BoundaryModeSim:
        """Assemble the cross-section this Stage meshes.

        The Staircase is a component of its own — Strips and electrodes,
        drawn in the Cross-section's transverse coordinates and extruded
        along the propagation direction — so the plane cuts through the
        middle of it rather than through the drawn device.

        Args:
            staircase: The Staircase to mesh; built from the Bias point
                when omitted.

        Returns:
            The configured (unmeshed) ``BoundaryModeSim``.
        """
        study = self._require_study()
        stair = staircase if staircase is not None else self.staircase()
        # One mesh serves the whole frequency sweep: the femwell Route
        # re-solves it per frequency without reading the boundary-mode
        # block, which records the first frequency so the sim is a
        # complete description.
        return self.build_staircase_simulation(
            stair,
            output_dir=study.stage_dir(self.stage_name),
            freq_hz=float(self.frequencies_hz[0]),
            num_modes=self.num_modes,
        )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _check_containment(self, ratio: float, freq_hz: float) -> None:
        """Warn when a solved Mode is squeezed by its Window (ADR 0002)."""
        if ratio > self.boundary_field_tol:
            warnings.warn(
                f"The {self.stage_name} stage's mode at f = {freq_hz / 1e9:g} "
                f"GHz carries {ratio:.1%} of its peak field at the window "
                f"boundary (tolerance {self.boundary_field_tol:.1%}); the "
                "window is squeezing the line mode rather than shielding "
                f"it. Widen it with study.{self.stage_name}(window=..., "
                "window_z=...) or move the electrodes further apart.",
                stacklevel=2,
            )

    @property
    def solved_bias_v(self) -> float | None:
        """Bias the cached line parameters were solved at, or None."""
        return self._solved_bias_v if self.has_run else None

    def _pick_line_mode(self, modes: Sequence[Any], freq_hz: float) -> Any:
        """Select the physical line Mode of one frequency, and check it.

        Args:
            modes: Every Mode the Route solved at this frequency.
            freq_hz: The frequency, named in the containment warning.

        Returns:
            The selected Mode.
        """
        from gsim.common.modes import select_line_mode
        from gsim.modulator.route import mode_boundary_ratio

        mode = select_line_mode(
            modes,
            rule=self.rule,
            min_index=self.min_index,
            max_loss_ratio=self.max_loss_ratio,
            degeneracy_rtol=self.degeneracy_rtol,
        )
        self._check_containment(mode_boundary_ratio(mode), freq_hz)
        return mode

    def _guess_for(self, solved: Sequence[complex]) -> float | None:
        """Effective-index guess for the next frequency of the sweep.

        Args:
            solved: The line Modes' indices solved so far, in sweep order.

        Returns:
            The previous frequency's index when the Mode is tracked, and
            the configured guess otherwise.
        """
        if self.track_modes and solved:
            return float(solved[-1].real)
        return self.n_guess

    def _check_continuity(self, n_eff: Sequence[complex]) -> None:
        """Warn when ``n_rf(f)`` steps rather than disperses.

        A shift-and-invert search can settle on a different branch from
        one frequency to the next, and the sweep then reports an RF index
        that jumps. Tracking makes that unlikely rather than impossible,
        so the sweep is checked either way.

        Args:
            n_eff: The solved indices, in frequency order.
        """
        indices = np.asarray([value.real for value in n_eff], dtype=np.float64)
        frequencies = np.asarray(self.frequencies_hz, dtype=np.float64)
        if indices.size < 2:
            return
        steps = np.abs(np.diff(indices)) / np.maximum(np.abs(indices[:-1]), 1e-12)
        # Against the frequency step, not against a fixed number: the
        # sweep's spacing is the user's, and a line disperses by about as
        # much as its frequency moves.
        spacing = np.diff(frequencies) / frequencies[:-1]
        jumped = np.nonzero(steps > self.jump_rtol * spacing)[0]
        if jumped.size == 0:
            return
        where = ", ".join(
            f"{self.frequencies_hz[i] / 1e9:g} -> "
            f"{self.frequencies_hz[i + 1] / 1e9:g} GHz "
            f"({indices[i]:.3g} -> {indices[i + 1]:.3g})"
            for i in jumped
        )
        tracked = (
            " Every frequency after it was aimed at the index solved before "
            "it, so the branch it changed to is the one the rest of the "
            "sweep followed, smoothly and wrongly."
            if self.track_modes
            else ""
        )
        warnings.warn(
            f"The {self.stage_name} stage's rf index jumps across the sweep "
            f"at {where}, faster than the frequency step: the eigenvalue "
            "search has most likely settled on a different mode branch "
            f"rather than the line mode dispersing.{tracked} Raise "
            f"num_modes, tighten the selection with study.{self.stage_name}"
            "(min_index=...), or solve the frequencies from the jump onwards "
            "on their own.",
            stacklevel=2,
        )

    def _require_metallic_boundaries(self) -> None:
        """Refuse a perfect electrode that femwell would leave as a hole.

        femwell has one perfect-conductor condition and applies it to
        every facet of the domain boundary at once, so a ``"pec"``
        electrode — which is a hole in that boundary — is a conductor
        only while ``metallic_boundaries`` is on. Off, the same hole
        takes the natural condition and the Stage would quietly solve a
        cross-section with open slots where its electrodes should be.

        Raises:
            ValueError: When the two settings contradict each other.
        """
        if self.metallic_boundaries:
            return
        raise ValueError(
            f"The {self.stage_name} stage's femwell route cannot leave its "
            "electrodes perfect while metallic_boundaries is off: femwell "
            "applies that one condition to the whole domain boundary, and a "
            "perfect electrode is a hole in it, so the electrodes would come "
            "out as open slots. Turn the wall back on with "
            f"study.{self.stage_name}(metallic_boundaries=True), or mesh the "
            "electrodes as lossy volumes with "
            f"study.{self.stage_name}(conductor_model='volume')."
        )

    def _check_contour_order(self) -> None:
        """Warn when the contour current is read off a first-order field.

        femwell solves for ``E`` and derives ``H`` from its curl, one
        order lower, so a first-order solve leaves ``H`` piecewise
        constant exactly where a ``"pec"`` conductor's contour integral
        reads it. On the shipped coax benchmark that is 8-29% of the
        impedance depending on the mesh, and it does not converge with
        refinement — only with order.
        """
        if self.order >= 2:
            return
        warnings.warn(
            f"The {self.stage_name} stage's femwell route reads its "
            "characteristic impedance off the field around a perfect "
            f"conductor at order {self.order}, where femwell's h field is "
            "piecewise constant: the impedance is biased high by tens of "
            f"percent. Solve with study.{self.stage_name}(order=2), or mesh "
            "the electrodes as lossy volumes with "
            f"study.{self.stage_name}(conductor_model='volume').",
            stacklevel=2,
        )

    def _solve_femwell(
        self, sim: BoundaryModeSim, staircase: StaircaseCrossSection
    ) -> tuple[list[complex], list[complex]]:
        """Solve the Staircase with femwell, per frequency.

        The femwell Route reads the Mode's fields as it solves, so it is
        the Route that also checks the Window while it selects. The
        signal current it divides the power by follows the conductor
        model: a ``"volume"`` electrode carries a conduction current over
        its own elements, and a ``"pec"`` one carries Ampere's contour
        integral around the hole it left in the mesh.

        Args:
            sim: The meshed Staircase simulation.
            staircase: The Staircase it was built from.

        Returns:
            ``(n_eff, z0_ohm)``, one entry per configured frequency.
        """
        import meshio
        from scipy.constants import speed_of_light as c0

        from gsim.femwell.adapter import (
            boundary_facets_on_rect,
            epsilon_by_region,
            region_elements,
            solve_modes,
            z0_power_current,
        )

        stack = staircase.stack("rf")
        mesh_path = sim.mesh_path
        mesh = meshio.read(str(mesh_path))
        signal = self.signal_electrode()
        on_contour = self.effective_conductor_model() == "pec"
        if on_contour:
            self._require_metallic_boundaries()
            self._check_contour_order()
            h_span, v_span = staircase.electrode_extent(signal)
            signal_elements = None
        else:
            signal_elements = region_elements(mesh, signal)

        n_eff: list[complex] = []
        z0_ohm: list[complex] = []
        for freq in self.frequencies_hz:
            modes = solve_modes(
                mesh_path,
                epsilon=epsilon_by_region(mesh, stack, frequency_hz=freq),
                wavelength_um=c0 / freq * 1e6,
                num_modes=self.num_modes,
                order=self.order,
                metallic_boundaries=self.metallic_boundaries,
                n_guess=self._guess_for(n_eff),
            )
            mode = self._pick_line_mode(modes, freq)
            n_eff.append(complex(mode.n_eff))
            current_facets = (
                boundary_facets_on_rect(mode.basis.mesh, h_span=h_span, v_span=v_span)
                if on_contour
                else None
            )
            z0_ohm.append(
                z0_power_current(
                    mode,
                    frequency_hz=freq,
                    current_elements=signal_elements,
                    current_facets=current_facets,
                )
            )
        return n_eff, z0_ohm

    def _solve_palace(
        self,
        sim: BoundaryModeSim,
        staircase: StaircaseCrossSection,
        binary: Path | None,
    ) -> tuple[list[complex], list[complex]]:
        """Solve the same Staircase on the same mesh with Palace.

        Palace's *text* results are effective indices without fields, so
        the Window-containment ratio — measured while the Mode is being
        chosen — cannot be had here. The impedance can: the solve saves
        every Mode it might select, and the selected one's fields are
        read back and put through the same Marks-Williams integral the
        femwell Route runs.

        Args:
            sim: The meshed Staircase simulation.
            staircase: The Staircase it was built from, holding the
                signal conductor's outline.
            binary: Palace executable, as ``require_route`` resolved
                it; resolved again when ``None``.

        Returns:
            ``(n_eff, z0_ohm)``, one entry per configured frequency.
        """
        from gsim.modulator.route import (
            containment_unmeasurable,
            metallic_boundary_unexpressed,
            palace_binary,
            palace_line_impedance,
            solve_palace_modes,
        )

        warnings.warn(containment_unmeasurable(self.stage_name), stacklevel=2)
        if self.metallic_boundaries:
            warnings.warn(metallic_boundary_unexpressed(self.stage_name), stacklevel=2)
        executable = palace_binary(binary, stage_name=self.stage_name)
        verbose = self._is_verbose()
        h_span, v_span = staircase.electrode_extent(self.signal_electrode())

        n_eff: list[complex] = []
        z0_ohm: list[complex] = []
        for freq in self.frequencies_hz:
            guess = self._guess_for(n_eff)
            modes = solve_palace_modes(
                sim,
                freq_hz=freq,
                num_modes=self.num_modes,
                binary=executable,
                target=guess if guess is not None else 0.0,
                # Which Mode is the line Mode is not known until they are
                # all solved, so every one of them is saved.
                save=self.num_modes,
                verbose=verbose,
            )
            mode = self._pick_line_mode(modes, freq)
            n_eff.append(complex(mode.n_eff))
            z0_ohm.append(
                palace_line_impedance(
                    sim,
                    mode,
                    h_span=h_span,
                    v_span=v_span,
                    stage_name=self.stage_name,
                )
            )
        return n_eff, z0_ohm

    def _solve(self) -> RFLineParams:
        """Mesh the Staircase and solve the line Mode at every frequency."""
        from gsim.common.twmzm_report import line_params_from_neff

        # Before the charge solve and before meshing: a user whose Route
        # cannot run should pay nothing to find that out.
        binary = require_route(self.route, stage_name=self.stage_name)

        point = self.bias_point()
        staircase = self.staircase()
        sim = self.simulation(staircase)
        sim.mesh(**self.mesh)

        if self.route == "femwell":
            n_eff, z0_ohm = self._solve_femwell(sim, staircase)
        else:
            n_eff, z0_ohm = self._solve_palace(sim, staircase, binary)
        self._check_continuity(n_eff)

        self._solved_bias_v = point.bias_v
        return line_params_from_neff(
            np.asarray(self.frequencies_hz, dtype=np.float64),
            n_eff,
            z0_ohm=z0_ohm,
        )
