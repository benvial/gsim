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

Either Backend can solve the Staircase. The femwell Route is the
default and the only one that reads the Mode's fields, so it is the one
that extracts the characteristic impedance and checks the Window; the
Palace Route solves the same Staircase on the same mesh and reports the
effective index, leaving the impedance NaN rather than fabricating it.

The Palace Route is the harder one here. A Staircase carries its
electrodes as volumes of conducting material, whose permittivity has a
huge imaginary part at RF, and Palace's shift-and-invert search will
return those metal-dominated modes rather than the quasi-TEM one unless
it is aimed carefully: expect to raise ``num_modes`` and move ``n_guess``
onto the expected line index, and to be told so by
:func:`~gsim.common.modes.select_line_mode` when neither is enough.

The selected Route's runtime is checked before the Stage meshes, so a
missing extra or a missing binary costs nothing but the error message.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pydantic import Field, PrivateAttr, field_validator

from gsim.common.modes import LineModeRule
from gsim.common.stack.staircase import DEFAULT_ELECTRODES, ElectrodeSpec
from gsim.modulator.route import require_route
from gsim.modulator.staircase import StaircaseStage

if TYPE_CHECKING:
    from pathlib import Path

    from gsim.common.stack.staircase import StaircaseCrossSection
    from gsim.common.twmzm_report import RFLineParams
    from gsim.modulator.carriers import CarrierResponse, CarrierResponseSweep
    from gsim.palace import BoundaryModeSim

__all__ = ["RFStage"]

#: Biases this far apart (V) count as the same Bias point.
BIAS_TOL_V: float = 1e-9


class RFStage(StaircaseStage):
    """The line parameters of the Traveling-wave electrode, versus frequency.

    Attributes:
        route: Backend answering this Stage — ``"femwell"`` (the default)
            or ``"palace"``. Both solve the same Staircase on the same
            mesh; only femwell reads the Mode's fields, so the Palace
            Route reports NaN for the characteristic impedance and for
            the Window-containment ratio, and needs ``num_modes`` and
            ``n_guess`` aimed at the line Mode to find it past the
            electrodes' metal-dominated ones.
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
            around the Junction: raise ``n_strips`` with it.
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
            usual condition for a shielded line solve.
        order: Finite-element order of the mode solve.
        n_guess: Effective-index guess centering the eigenvalue search;
            RF materials have large conductive ``|Im(eps)|``, so the
            default guess is the slow-wave index rather than the solver's.
        track_modes: Follow the line Mode across the sweep, guessing each
            frequency from the index solved at the one before it, so a
            shift-and-invert search anchored to one number cannot settle
            on a different branch as the materials move with frequency.
            Turn it off to aim every frequency at ``n_guess`` instead.
        jump_rtol: Relative step in ``Re(n_eff)`` between neighbouring
            frequencies above which the sweep is reported as having
            jumped branch rather than dispersed.
        mesh: Keyword arguments forwarded to the mesh pipeline.
        airbox: Background region around the Staircase.
    """

    stage_name: ClassVar[str] = "rf"

    #: Bias the cached result was solved at, read through
    #: :attr:`solved_bias_v`.
    _solved_bias_v: float | None = PrivateAttr(default=None)

    frequencies_hz: list[float] = Field(
        default_factory=lambda: [10e9, 40e9], min_length=1
    )
    n_strips: int = Field(default=5, ge=1)
    bias_v: float | None = None
    electrodes: ElectrodeSpec = Field(default=DEFAULT_ELECTRODES)
    signal_contact: str | None = None
    num_modes: int = Field(default=4, ge=1)
    rule: LineModeRule | None = None
    min_index: float = Field(default=1.0, ge=0.0)
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

    def staircase(self) -> StaircaseCrossSection:
        """Reduce the Bias point's Carrier map to a meshable Staircase.

        The Strips tile the Junction extent — the two Regions the
        metallurgical boundary separates — unless ``strip_span`` widens
        them, and take the plasma-dispersion coefficients and mobilities
        of the carriers Stage, so the RF conductivity is the same
        coupling the optical Stage reads.

        The Strip materials are valid up to the highest requested
        frequency, which is what the solve asks of them.

        Returns:
            The Staircase Cross-section, drawn on its own component.
        """
        return self.build_staircase(
            self.bias_point().carriers,
            n_strips=self.n_strips,
            electrodes=self.electrodes,
            permittivity=self.strip_permittivity,
            fmax=max(self.frequencies_hz),
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
            kind="rf",
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
        if indices.size < 2:
            return
        previous = indices[:-1]
        steps = np.abs(np.diff(indices)) / np.maximum(np.abs(previous), 1e-12)
        jumped = np.nonzero(steps > self.jump_rtol)[0]
        if jumped.size == 0:
            return
        where = ", ".join(
            f"{self.frequencies_hz[i] / 1e9:g} -> "
            f"{self.frequencies_hz[i + 1] / 1e9:g} GHz "
            f"({indices[i]:.3g} -> {indices[i + 1]:.3g})"
            for i in jumped
        )
        warnings.warn(
            f"The {self.stage_name} stage's rf index jumps across the sweep "
            f"at {where}, by more than {self.jump_rtol:.0%}: the eigenvalue "
            "search has most likely settled on a different mode branch "
            "rather than the line mode dispersing. Raise num_modes, tighten "
            f"the selection with study.{self.stage_name}(min_index=...) or "
            "solve the frequencies that jumped on their own.",
            stacklevel=2,
        )

    def _solve_femwell(
        self, sim: BoundaryModeSim, staircase: StaircaseCrossSection
    ) -> tuple[list[complex], list[complex]]:
        """Solve the Staircase with femwell, per frequency.

        The femwell Route reads the Mode's fields, so it is the Route that
        extracts the characteristic impedance over the signal conductor.

        Args:
            sim: The meshed Staircase simulation.
            staircase: The Staircase it was built from.

        Returns:
            ``(n_eff, z0_ohm)``, one entry per configured frequency.
        """
        import meshio
        from scipy.constants import speed_of_light as c0

        from gsim.femwell.adapter import (
            epsilon_by_region,
            region_elements,
            solve_modes,
            z0_power_current,
        )

        stack = staircase.stack("rf")
        mesh_path = sim.mesh_path
        mesh = meshio.read(str(mesh_path))
        signal_elements = region_elements(mesh, self.signal_electrode())

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
            z0_ohm.append(
                z0_power_current(
                    mode, frequency_hz=freq, current_elements=signal_elements
                )
            )
        return n_eff, z0_ohm

    def _solve_palace(
        self, sim: BoundaryModeSim, binary: Path | None
    ) -> tuple[list[complex], list[complex]]:
        """Solve the same Staircase on the same mesh with Palace.

        Palace's ``BoundaryMode`` results are effective indices without
        fields, so the Marks-Williams impedance integral has nothing to
        integrate: the impedance comes back NaN, once warned about,
        rather than silently wrong.

        Args:
            sim: The meshed Staircase simulation.
            binary: Palace executable, as ``require_route`` resolved
                it; resolved again when ``None``.

        Returns:
            ``(n_eff, z0_ohm)``, one entry per configured frequency.
        """
        from gsim.modulator.route import (
            containment_unmeasurable,
            palace_binary,
            solve_palace_modes,
        )

        warnings.warn(
            f"The {self.stage_name} stage's palace route reports no "
            "characteristic impedance: Palace's boundary-mode results carry "
            "no mode fields, so the Marks-Williams extraction cannot run and "
            "z0_ohm comes back NaN. Solve with "
            f"study.{self.stage_name}(route='femwell') for the impedance.",
            stacklevel=2,
        )
        warnings.warn(containment_unmeasurable(self.stage_name), stacklevel=2)
        executable = palace_binary(binary, stage_name=self.stage_name)
        verbose = self._is_verbose()

        n_eff: list[complex] = []
        for freq in self.frequencies_hz:
            guess = self._guess_for(n_eff)
            modes = solve_palace_modes(
                sim,
                freq_hz=freq,
                num_modes=self.num_modes,
                binary=executable,
                target=guess if guess is not None else 0.0,
                verbose=verbose,
            )
            n_eff.append(complex(self._pick_line_mode(modes, freq).n_eff))
        return n_eff, [complex(float("nan"), float("nan"))] * len(n_eff)

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
            n_eff, z0_ohm = self._solve_palace(sim, binary)
        self._check_continuity(n_eff)

        self._solved_bias_v = point.bias_v
        return line_params_from_neff(
            np.asarray(self.frequencies_hz, dtype=np.float64),
            n_eff,
            z0_ohm=z0_ohm,
        )
