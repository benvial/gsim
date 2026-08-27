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

femwell is optional: the Stage checks for it before it meshes, so a
missing extra costs nothing but the error message.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from gsim.modulator.stage import Stage

if TYPE_CHECKING:
    import meshio

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


class OpticalStage(Stage):
    """The carrier-perturbed optical Mode, bias point by bias point.

    Attributes:
        wavelength_um: Vacuum wavelength of the solve (um).
        num_modes: Number of Modes to solve at each Bias point; the
            slowest propagating one is reported.
        window: In-plane optical Window (um); derived as a box around the
            rib when unset.
        window_z: Vertical Window (um); derived from the guiding layer
            when unset.
        mode_margin_um: Half-width of the derived Window either side of
            the Junction (um).
        z_above_um: Margin above the guiding layer in the derived vertical
            Window (um).
        z_below_um: Margin below it (um).
        perturbed_regions: Regions whose permittivity the Carrier maps
            perturb; defaults to the device's doped Regions.
        mesh: Keyword arguments forwarded to the mesh pipeline.
        airbox: Background region around the clipped domain.
        min_index: Lower bound on ``Re(n_eff)`` for a Mode to count as
            guided; raise it to the cladding index to reject radiation
            Modes.
        boundary_field_tol: Warn above this boundary-field ratio, the sign
            of a Window too small for the Mode.
        order: Finite-element order of the mode solve.
        n_guess: Effective-index guess centering the eigenvalue search.
    """

    stage_name: ClassVar[str] = "optical"

    wavelength_um: float = Field(default=1.55, gt=0.0)
    num_modes: int = Field(default=1, ge=1)
    window: tuple[float, float] | None = None
    window_z: tuple[float, float] | None = None
    mode_margin_um: float = Field(default=2.0, gt=0.0)
    z_above_um: float = Field(default=1.0, ge=0.0)
    z_below_um: float = Field(default=1.0, ge=0.0)
    perturbed_regions: list[str] | None = None
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
    min_index: float = Field(default=1.0, ge=0.0)
    boundary_field_tol: float = Field(default=0.01, gt=0.0)
    order: int = Field(default=1, ge=1)
    n_guess: float | None = None

    # ------------------------------------------------------------------
    # Derivation
    # ------------------------------------------------------------------

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
        layout = study.layout

        sim = BoundaryModeSim()
        sim.set_output_dir(study.stage_dir(self.stage_name))
        sim.set_stack(study.stack)
        sim.set_geometry(study.component)
        sim.set_airbox(**self.airbox)
        sim.set_cross_section(
            study.plane,
            window=(
                self.window
                if self.window is not None
                else layout.window_around_junction(margin_um=self.mode_margin_um)
            ),
            window_z=(
                self.window_z
                if self.window_z is not None
                else layout.window_z_around_guide(
                    above_um=self.z_above_um, below_um=self.z_below_um
                )
            ),
        )
        sim.set_boundary_mode(
            freq=c0 / (self.wavelength_um * 1e-6), num_modes=self.num_modes
        )
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

    def _solve(self) -> OpticalSweep:
        """Mesh the optical Window and solve the Mode at every Bias point."""
        import meshio

        from gsim.common.modes import select_line_mode
        from gsim.femwell.adapter import (
            boundary_field_ratio,
            epsilon_by_region,
            solve_modes,
        )
        from gsim.femwell.runtime import require_femwell, require_skfem

        # Before the charge solve and before meshing: a user without the
        # extra should pay nothing to find that out.
        require_femwell()
        require_skfem()

        study = self._require_study()
        responses = study.carriers.run()
        if not responses.points:
            raise ValueError(
                "The bias sweep has no points, so there is no mode to solve. "
                "Configure study.charge(biases=[...])."
            )

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
                n_guess=self.n_guess,
            )
            mode = select_line_mode(modes, min_index=self.min_index)
            ratio = boundary_field_ratio(mode)
            self._check_containment(ratio, point.bias_v)
            solved.append((point.bias_v, complex(mode.n_eff), ratio))

        reference_bias, reference_index, _ = next(
            (entry for entry in solved if entry[0] == 0.0), solved[0]
        )
        wavelength_cm = self.wavelength_um * 1e-4
        return OpticalSweep(
            contact=responses.contact,
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
