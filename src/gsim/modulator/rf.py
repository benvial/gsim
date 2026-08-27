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

femwell is optional: the Stage checks for it before it meshes, so a
missing extra costs nothing but the error message.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pydantic import Field, PrivateAttr, field_validator

from gsim.common.modes import LineModeRule
from gsim.common.stack.staircase import DEFAULT_ELECTRODES, ElectrodeSpec
from gsim.modulator.stage import Stage

if TYPE_CHECKING:
    from gsim.common.stack.staircase import StaircaseCrossSection
    from gsim.common.twmzm_report import RFLineParams
    from gsim.modulator.carriers import CarrierResponse, CarrierResponseSweep
    from gsim.palace import BoundaryModeSim

__all__ = ["RFStage"]

#: Drawn length of the Staircase along the propagation direction (um).
#: The Cross-section is invariant along it, so it is not a setting; the
#: plane is taken through the middle of the drawn rectangles.
STRIP_LENGTH_UM: float = 10.0

#: Biases this far apart (V) count as the same Bias point.
BIAS_TOL_V: float = 1e-9


class RFStage(Stage):
    """The line parameters of the Traveling-wave electrode, versus frequency.

    Attributes:
        frequencies_hz: RF frequencies to solve at (Hz), kept ascending.
        n_strips: Number of Strips the Carrier map is reduced to; more
            strips approximate the continuous profile more closely at the
            cost of mesh size.
        bias_v: Bias point the Staircase is built from; the last point of
            the Bias sweep when unset.
        strip_span: ``(min, max)`` extent the Strips tile along the
            junction axis (um); the Junction extent — the rib — when
            unset. Widen it to the doped slab to carry the pads, and
            their series resistance, into the RF solve.
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
    strip_span: tuple[float, float] | None = None
    window: tuple[float, float] | None = None
    window_z: tuple[float, float] | None = None
    electrodes: ElectrodeSpec = Field(default=DEFAULT_ELECTRODES)
    signal_contact: str | None = None
    num_modes: int = Field(default=4, ge=1)
    rule: LineModeRule | None = None
    min_index: float = Field(default=1.0, ge=0.0)
    degeneracy_rtol: float = Field(default=0.03, gt=0.0)
    strip_permittivity: float = Field(default=11.9, gt=0.0)
    substrate_thickness_um: float = Field(default=2.0, gt=0.0)
    boundary_field_tol: float = Field(default=0.2, gt=0.0)
    metallic_boundaries: bool = True
    order: int = Field(default=1, ge=1)
    n_guess: float | None = 3.0
    mesh: dict[str, Any] = Field(
        default_factory=lambda: {
            "preset": "coarse",
            "refined_mesh_size": 0.05,
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
        from gsim.common.stack.staircase import build_staircase_cross_section

        study = self._require_study()
        span = study.layout.junction_span
        point = self.bias_point()
        return build_staircase_cross_section(
            point.carriers,
            n_strips=self.n_strips,
            junction=self.strip_span if self.strip_span is not None else span.h,
            zmin=span.z[0],
            zmax=span.z[1],
            length=STRIP_LENGTH_UM,
            electrodes=self.electrodes,
            dispersion=study.carriers.dispersion,
            permittivity=self.strip_permittivity,
            mu_n_cm2=study.carriers.mu_n_cm2,
            mu_p_cm2=study.carriers.mu_p_cm2,
            fmax=max(self.frequencies_hz),
            axis="x",
            value=STRIP_LENGTH_UM / 2.0,
            substrate_thickness=self.substrate_thickness_um,
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
        from gsim.palace import BoundaryModeSim

        study = self._require_study()
        stair = staircase if staircase is not None else self.staircase()

        sim = BoundaryModeSim()
        sim.set_output_dir(study.stage_dir(self.stage_name))
        sim.set_stack(stair.stack("rf"))
        sim.set_geometry(stair.component)
        sim.set_airbox(**self.airbox)
        sim.set_cross_section(
            f"x={STRIP_LENGTH_UM / 2.0}",
            window=self.window,
            window_z=self.window_z,
        )
        # One mesh serves the whole frequency sweep: the femwell Route
        # re-solves it per frequency without reading this block, which
        # records the first one so the sim is a complete description.
        sim.set_boundary_mode(
            freq=float(self.frequencies_hz[0]), num_modes=self.num_modes
        )
        return sim

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

    def _solve(self) -> RFLineParams:
        """Mesh the Staircase and solve the line Mode at every frequency."""
        import meshio
        from scipy.constants import speed_of_light as c0

        from gsim.common.modes import select_line_mode
        from gsim.common.twmzm_report import line_params_from_neff
        from gsim.femwell.adapter import (
            boundary_field_ratio,
            epsilon_by_region,
            region_elements,
            solve_modes,
            z0_power_current,
        )
        from gsim.femwell.runtime import require_femwell, require_skfem

        # Before the charge solve and before meshing: a user without the
        # extra should pay nothing to find that out.
        require_femwell()
        require_skfem()

        point = self.bias_point()
        staircase = self.staircase()
        stack = staircase.stack("rf")

        sim = self.simulation(staircase)
        sim.mesh(**self.mesh)
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
                n_guess=self.n_guess,
            )
            mode = select_line_mode(
                modes,
                rule=self.rule,
                min_index=self.min_index,
                degeneracy_rtol=self.degeneracy_rtol,
            )
            self._check_containment(boundary_field_ratio(mode), freq)
            n_eff.append(complex(mode.n_eff))
            z0_ohm.append(
                z0_power_current(
                    mode, frequency_hz=freq, current_elements=signal_elements
                )
            )

        self._solved_bias_v = point.bias_v
        return line_params_from_neff(
            np.asarray(self.frequencies_hz, dtype=np.float64),
            n_eff,
            z0_ohm=z0_ohm,
        )
