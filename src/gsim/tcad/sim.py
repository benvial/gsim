"""DEVSIM charge-transport simulation on the shared native-2D mesh.

``ChargeTransportSim`` follows the established gsim backend idiom
(``set_geometry`` / ``set_stack`` / ``set_cross_section``, then solve or
sweep). Meshing is delegated to the existing ``BoundaryModeSim`` native-2D
pipeline — the exact mesh Palace BoundaryMode and the femwell adapter see —
so there is no second meshing path. Contacts declared with ``add_contact``
become named dim-1 physical groups that DEVSIM binds boundary conditions to
via ``add_gmsh_contact``.

The solve uses DEVSIM's prebuilt Scharfetter-Gummel drift-diffusion physics
(``devsim.python_packages.simple_physics``). Doping enters as ``Donors`` /
``Acceptors`` node solutions evaluated from the analytic profiles in
:mod:`gsim.tcad.doping`, combined into the ``NetDoping`` node model.
"""

from __future__ import annotations

import itertools
import logging
import math
from contextlib import suppress
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.tcad.doping import (
    DopingProfile,
    acceptor_donor_concentrations,
)
from gsim.tcad.mesh import UM_TO_CM, write_scaled_msh
from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap
from gsim.tcad.runtime import import_simple_physics, require_devsim

logger = logging.getLogger(__name__)

# DEVSIM keeps meshes and devices in one process-wide namespace, so every
# sim (and every re-mesh) claims its own names: two Studies, or one Stage
# re-run after a configuration change, would otherwise collide on the
# second setup with "a mesh already exists with name ...".
_DEVSIM_NAME_IDS = itertools.count()


def _unique_devsim_name(kind: str) -> str:
    """Return a process-unique DEVSIM mesh or device name."""
    return f"gsim_tcad_{kind}_{next(_DEVSIM_NAME_IDS)}"


#: DEVSIM devices this process created and has not deleted. DEVSIM solves
#: every registered device in one Newton, so a device left behind by a
#: finished sim would take part in the next sim's solve.
_LIVE_DEVICES: set[str] = set()


def _release_gsim_devices(devsim: Any) -> None:
    """Delete the DEVSIM devices gsim created and still holds."""
    for name in list(_LIVE_DEVICES):
        with suppress(Exception):
            devsim.delete_device(device=name)
        _LIVE_DEVICES.discard(name)


class ChargeTransportSim(BaseModel):
    """Poisson + drift-diffusion simulation of a waveguide cross-section.

    Example:
        >>> sim = ChargeTransportSim()
        >>> sim.set_output_dir("./tcad-sim")
        >>> sim.set_geometry(component)
        >>> sim.set_stack(stack)
        >>> sim.set_cross_section("x=0", window=(-25.0, -15.0))
        >>> sim.add_contact(name="anode", layer_a="metal1", layer_b="p_rib")
        >>> sim.add_contact(name="cathode", layer_a="metal1", layer_b="n_rib")
        >>> sim.add_doping(
        ...     StepDoping(
        ...         region="p_rib", dopant_type="acceptor", concentration_cm3=1e18
        ...     )
        ... )
        >>> sim.mesh(preset="coarse")
        >>> result = sim.sweep([0.0, -0.5, -1.0], contact="cathode")
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    simulation_type: Literal["charge"] = "charge"

    #: Lattice temperature (K) for the silicon parameter set.
    temperature: float = Field(default=300.0, gt=0.0)
    #: DEVSIM material name assigned to the semiconductor regions.
    material: str = "Silicon"
    #: Analytic doping profiles (each names the mesh region it applies to).
    doping: list[DopingProfile] = Field(default_factory=list)
    #: Voltage step used when ramping the swept contact between biases (V).
    bias_step_v: float = Field(default=0.1, gt=0.0)
    #: Frequency (Hz) of the small-signal AC solve extracting C(V); low
    #: enough to be quasi-static.
    small_signal_freq_hz: float = Field(default=1.0, gt=0.0)
    #: Newton absolute error target passed to ``devsim.solve``.
    absolute_error: float = Field(default=1e10, gt=0.0)
    #: Newton relative error target passed to ``devsim.solve``. The 2D
    #: drift-diffusion Newton typically floors around 1e-8 on coarse
    #: meshes; tighten per solve when the mesh supports it.
    relative_error: float = Field(default=1e-6, gt=0.0)
    #: Newton iteration cap passed to ``devsim.solve``.
    max_iterations: int = Field(default=100, gt=0)
    #: Use DEVSIM's 128-bit extended-precision assembly (recommended).
    extended_precision: bool = True
    #: Claim DEVSIM's process-global namespace when this sim sets up its
    #: device: drop the circuit (DEVSIM refuses new circuit nodes once an
    #: analysis has run) and delete the devices gsim created earlier
    #: (DEVSIM solves every registered device in one Newton). Foreign
    #: DEVSIM devices are never touched. Turn this off only when an
    #: earlier charge-transport device in the process must stay alive.
    claim_devsim_namespace: bool = True

    # Meshing is delegated to the shared native-2D BoundaryMode pipeline.
    _bsim: Any = PrivateAttr(default=None)
    _devsim_mesh_path: Path | None = PrivateAttr(default=None)
    _devsim_mesh_name: str | None = PrivateAttr(default=None)
    _device: str | None = PrivateAttr(default=None)
    _contact_regions: dict[str, str] = PrivateAttr(default_factory=dict)
    _interface_names: list[str] = PrivateAttr(default_factory=list)
    _interface_regions: dict[str, tuple[str, str]] = PrivateAttr(default_factory=dict)
    _dd_initialized: bool = PrivateAttr(default=False)
    _current_bias: dict[str, float] = PrivateAttr(default_factory=dict)

    # ------------------------------------------------------------------
    # Delegated configuration (shared geometry seam)
    # ------------------------------------------------------------------

    def _boundary_sim(self) -> Any:
        """Lazily create the internal BoundaryModeSim used for meshing."""
        if self._bsim is None:
            from gsim.palace import BoundaryModeSim

            self._bsim = BoundaryModeSim()
        return self._bsim

    def set_geometry(self, component: Any) -> None:
        """Set the gdsfactory component (delegates to the mesh pipeline)."""
        self._boundary_sim().set_geometry(component)

    def set_stack(self, *args: Any, **kwargs: Any) -> None:
        """Configure the layer stack (same arguments as the palace backends)."""
        self._boundary_sim().set_stack(*args, **kwargs)

    def set_airbox(self, **kwargs: Any) -> None:
        """Configure the background region around the clipped domain."""
        self._boundary_sim().set_airbox(**kwargs)

    def set_output_dir(self, path: str | Path) -> None:
        """Set the output directory for mesh files."""
        self._boundary_sim().set_output_dir(path)

    def set_cross_section(
        self,
        plane: Any,
        *,
        window: tuple[float, float] | None = None,
        window_z: tuple[float, float] | None = None,
    ) -> None:
        """Set the cross-section plane and charge-transport window.

        Args:
            plane: Plane spec (``"x=<value>"`` / ``"y=<value>"``) or a
                prebuilt ``CrossSectionPlaneConfig``.
            window: In-plane ``(min, max)`` clip in um — typically the doped
                slab between the contacts.
            window_z: Vertical ``(min, max)`` clip in um.
        """
        self._boundary_sim().set_cross_section(plane, window=window, window_z=window_z)

    def add_contact(self, *, name: str, layer_a: str, layer_b: str) -> None:
        """Declare a named contact at the interface between two layers."""
        self._boundary_sim().add_contact(name=name, layer_a=layer_a, layer_b=layer_b)

    def add_interface(self, *, name: str, layer_a: str, layer_b: str) -> None:
        """Declare a semiconductor-semiconductor interface between regions.

        Adjacent doped mesh regions (e.g. the P and N halves of a rib) load
        into DEVSIM as separate regions; without an interface they are
        electrically disconnected and no junction forms. The shared curves
        between the two layers are tagged as a dim-1 physical group (the
        same mechanism as contacts) and bound with
        ``devsim.add_gmsh_interface`` plus potential/carrier continuity, so
        the drift-diffusion solution is continuous across the junction.

        Args:
            name: Interface / physical-group name (e.g. ``"junction"``).
            layer_a: First doped region layer name.
            layer_b: Second doped region layer name.
        """
        self._boundary_sim().add_contact(name=name, layer_a=layer_a, layer_b=layer_b)
        self._interface_names = [*self._interface_names, name]

    def add_doping(self, profile: DopingProfile) -> None:
        """Add an analytic doping profile (see :mod:`gsim.tcad.doping`)."""
        self.doping = [*self.doping, profile]

    @property
    def geometry(self) -> Any:
        """The delegated geometry object (or None)."""
        return self._bsim.geometry if self._bsim is not None else None

    @property
    def stack(self) -> Any:
        """The delegated layer stack (or None)."""
        return self._bsim.stack if self._bsim is not None else None

    @property
    def cross_section(self) -> Any:
        """The delegated cross-section plane and window (or None)."""
        return self._bsim.cross_section if self._bsim is not None else None

    @property
    def contact_specs(self) -> list[Any]:
        """Declared contact specs."""
        return list(self._bsim.contact_specs) if self._bsim is not None else []

    @property
    def output_dir(self) -> Path | None:
        """Output directory (or None)."""
        return self._bsim.output_dir if self._bsim is not None else None

    @property
    def has_mesh(self) -> bool:
        """Whether ``mesh()`` has run and left a mesh to read."""
        return self._bsim is not None and bool(self._bsim.has_mesh)

    def _meshed_boundary_sim(self) -> Any:
        """Return the delegated mesh pipeline, or explain it has not meshed."""
        if not self.has_mesh:
            raise ValueError(
                "No mesh generated for this ChargeTransportSim. Call mesh() first."
            )
        return self._bsim

    @property
    def mesh_path(self) -> Path:
        """Path of the shared native-2D mesh (um), once meshed.

        Raises:
            ValueError: When read before ``mesh()`` has run.
        """
        return Path(self._meshed_boundary_sim().mesh_path)

    @property
    def devsim_mesh_path(self) -> Path:
        """Path of the cm-scaled mesh copy loaded by DEVSIM, once meshed.

        Raises:
            ValueError: When read before ``mesh()`` has run.
        """
        if self._devsim_mesh_path is None:
            raise ValueError(
                "No DEVSIM mesh generated for this ChargeTransportSim. "
                "Call mesh() first."
            )
        return self._devsim_mesh_path

    @property
    def mesh_groups(self) -> dict[str, Any]:
        """Physical groups of the generated mesh, once meshed.

        Raises:
            ValueError: When read before ``mesh()`` has run.
        """
        return dict(self._meshed_boundary_sim().mesh_groups)

    # ------------------------------------------------------------------
    # Meshing
    # ------------------------------------------------------------------

    def mesh(self, **kwargs: Any) -> Any:
        """Generate the shared native-2D mesh and its cm-scaled DEVSIM copy.

        All keyword arguments are forwarded to ``BoundaryModeSim.mesh``
        (presets, mesh sizes, ...). Requires geometry, stack, cross-section
        and at least one contact to be configured.

        Returns:
            The mesh-generation result of the shared pipeline.
        """
        bsim = self._boundary_sim()
        if not bsim.contact_specs:
            raise ValueError(
                "Charge transport requires at least one contact. "
                "Call add_contact(name=..., layer_a=..., layer_b=...) first."
            )
        result = bsim.mesh(**kwargs)

        output_dir = bsim.output_dir
        if output_dir is None:  # pragma: no cover - mesh() enforces this
            raise ValueError("Output directory not set.")
        self._devsim_mesh_path = write_scaled_msh(
            result.mesh_path, Path(output_dir) / "devsim.msh", scale=UM_TO_CM
        )
        # A new mesh invalidates any existing DEVSIM device.
        self.reset_device()
        self._devsim_mesh_name = _unique_devsim_name("mesh")
        return result

    # ------------------------------------------------------------------
    # DEVSIM device setup
    # ------------------------------------------------------------------

    def _release_devsim_device(self) -> None:
        """Best-effort removal of this sim's DEVSIM device and mesh."""
        if self._device is None:
            return
        try:
            import importlib

            devsim: Any = importlib.import_module("devsim")
        except ImportError:  # pragma: no cover - nothing to release
            return
        with suppress(Exception):
            devsim.delete_device(device=self._device)
        _LIVE_DEVICES.discard(self._device)
        if self._devsim_mesh_name is not None:
            with suppress(Exception):
                devsim.delete_mesh(mesh=self._devsim_mesh_name)

    def reset_device(self) -> None:
        """Forget the DEVSIM device; setup runs again on the next solve."""
        self._release_devsim_device()
        self._devsim_mesh_name = None
        self._device = None
        self._dd_initialized = False
        self._contact_regions = {}
        self._interface_regions = {}
        self._current_bias = {}

    def _device_regions(self) -> list[str]:
        """Ordered unique mesh regions named by the doping profiles."""
        regions: list[str] = []
        for profile in self.doping:
            if profile.region not in regions:
                regions.append(profile.region)
        return regions

    @property
    def _electrical_contacts(self) -> list[Any]:
        """Contact specs bound as DEVSIM contacts (interfaces excluded)."""
        return [s for s in self.contact_specs if s.name not in self._interface_names]

    @property
    def _interfaces(self) -> list[Any]:
        """Contact specs declared as region-region interfaces."""
        return [s for s in self.contact_specs if s.name in self._interface_names]

    def _validate_setup(self) -> list[str]:
        """Check mesh/doping/contact consistency; return the device regions."""
        if self._devsim_mesh_path is None:
            raise ValueError("No mesh generated. Call mesh() first.")
        if not self.doping:
            raise ValueError(
                "No doping profiles declared. Call add_doping() with at "
                "least one profile."
            )

        groups = self.mesh_groups
        volumes = set(groups.get("volumes", {}))
        contact_lines = set(groups.get("contact_lines", {}))

        regions = self._device_regions()
        unknown = [r for r in regions if r not in volumes]
        if unknown:
            raise ValueError(
                f"Doping regions {unknown} are not volume groups on the "
                f"mesh. Available regions: {sorted(volumes)}"
            )

        self._contact_regions = {}
        for spec in self._electrical_contacts:
            if spec.name not in contact_lines:
                raise ValueError(
                    f"Contact '{spec.name}' has no line group on the mesh. "
                    "Re-run mesh() after add_contact()."
                )
            sides = [s for s in (spec.layer_a, spec.layer_b) if s in regions]
            if not sides:
                raise ValueError(
                    f"Contact '{spec.name}' touches neither of the doped "
                    f"device regions {regions}: it connects "
                    f"'{spec.layer_a}' and '{spec.layer_b}'. Add a doping "
                    "profile for the semiconductor side of the contact."
                )
            self._contact_regions[spec.name] = sides[0]
        if not self._contact_regions:
            raise ValueError(
                "Charge transport requires at least one contact. "
                "Call add_contact() before mesh()."
            )

        self._interface_regions = {}
        for spec in self._interfaces:
            if spec.name not in contact_lines:
                raise ValueError(
                    f"Interface '{spec.name}' has no line group on the mesh. "
                    "Re-run mesh() after add_interface()."
                )
            if spec.layer_a not in regions or spec.layer_b not in regions:
                raise ValueError(
                    f"Interface '{spec.name}' must join two doped device "
                    f"regions; got '{spec.layer_a}' and '{spec.layer_b}' "
                    f"with device regions {regions}."
                )
            self._interface_regions[spec.name] = (spec.layer_a, spec.layer_b)
        return regions

    def _apply_doping(self, devsim: Any, device: str, region: str) -> None:
        """Evaluate the region's profiles onto DEVSIM node solutions."""
        x_cm = np.asarray(
            devsim.get_node_model_values(device=device, region=region, name="x"),
            dtype=np.float64,
        )
        y_cm = np.asarray(
            devsim.get_node_model_values(device=device, region=region, name="y"),
            dtype=np.float64,
        )
        profiles = [p for p in self.doping if p.region == region]
        acceptors, donors = acceptor_donor_concentrations(
            profiles, x_cm / UM_TO_CM, y_cm / UM_TO_CM
        )
        for name, values in (("Acceptors", acceptors), ("Donors", donors)):
            devsim.node_solution(device=device, region=region, name=name)
            devsim.set_node_values(
                device=device, region=region, name=name, values=list(values)
            )
        devsim.node_model(
            device=device,
            region=region,
            name="NetDoping",
            equation="Donors - Acceptors",
        )

    def setup_device(self, device: str | None = None) -> str:
        """Create the DEVSIM device from the shared mesh.

        Loads the cm-scaled mesh, registers one DEVSIM region per doped
        mesh region and one contact per declared contact, applies the
        doping node models, and sets up the potential-only physics.

        DEVSIM's meshes, devices and circuits live in one process-global
        namespace, and it refuses new circuit nodes once an analysis has
        run. This device therefore claims the circuit by default
        (``claim_devsim_circuit``), which drops the contact sources of any
        DEVSIM device built earlier in the process — including this sim's
        own previous device. One charge-transport device is live at a
        time; meshes and devices themselves get process-unique names.

        Args:
            device: DEVSIM device name; a process-unique one is generated
                when omitted.

        Returns:
            The device name.
        """
        regions = self._validate_setup()
        device = device if device is not None else _unique_devsim_name("device")
        devsim = require_devsim()
        sp = import_simple_physics()

        if self.claim_devsim_namespace:
            # The circuit is process-global and refuses new nodes after an
            # analysis, and every registered device joins the next solve;
            # a previous charge device's circuit and device have to go.
            with suppress(Exception):
                devsim.delete_circuit()
            _release_gsim_devices(devsim)

        if self.extended_precision:
            # 128-bit float assembly: the 2D drift-diffusion Newton floors
            # around 1e-5 relative error in double precision on these
            # meshes; extended precision restores full convergence.
            devsim.set_parameter(name="extended_solver", value=True)
            devsim.set_parameter(name="extended_model", value=True)
            devsim.set_parameter(name="extended_equation", value=True)

        mesh_name = self._devsim_mesh_name or _unique_devsim_name("mesh")
        self._devsim_mesh_name = mesh_name
        devsim.create_gmsh_mesh(mesh=mesh_name, file=str(self._devsim_mesh_path))
        for region in regions:
            devsim.add_gmsh_region(
                mesh=mesh_name,
                gmsh_name=region,
                region=region,
                material=self.material,
            )
        for spec in self._electrical_contacts:
            devsim.add_gmsh_contact(
                mesh=mesh_name,
                gmsh_name=spec.name,
                region=self._contact_regions[spec.name],
                material="metal",
                name=spec.name,
            )
        for name, (region_a, region_b) in self._interface_regions.items():
            devsim.add_gmsh_interface(
                mesh=mesh_name,
                gmsh_name=name,
                region0=region_a,
                region1=region_b,
                name=name,
            )
        devsim.finalize_mesh(mesh=mesh_name)
        devsim.create_device(mesh=mesh_name, device=device)

        for region in regions:
            self._apply_doping(devsim, device, region)
            sp.SetSiliconParameters(device, region, self.temperature)
            sp.CreateSiliconPotentialOnly(device, region)
        for spec in self._electrical_contacts:
            # Each contact is driven through a circuit voltage source so the
            # small-signal AC solve can read the terminal admittance (the
            # C(V) extraction) and the DC current from the circuit node.
            devsim.circuit_element(
                name=self._source_name(spec.name),
                n1=sp.GetContactBiasName(spec.name),
                n2=0,
                value=0.0,
                acreal=0.0,
                acimag=0.0,
            )
            sp.CreateSiliconPotentialOnlyContact(
                device, self._contact_regions[spec.name], spec.name, True
            )
            self._current_bias[spec.name] = 0.0
        for name in self._interface_regions:
            self._interface_continuity(devsim, sp, device, name, "Potential")

        self._device = device
        _LIVE_DEVICES.add(device)
        self._dd_initialized = False
        return device

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def _solve_dc(self, devsim: Any) -> None:
        """Run one DC Newton solve with the configured tolerances."""
        devsim.solve(
            type="dc",
            absolute_error=self.absolute_error,
            relative_error=self.relative_error,
            maximum_iterations=self.max_iterations,
        )

    def _create_solution(self, sp: Any, device: str, region: str, name: str) -> None:
        """simple_physics re-exports CreateSolution in most DEVSIM versions."""
        create = getattr(sp, "CreateSolution", None)
        if create is None:  # pragma: no cover - version-dependent fallback
            import importlib

            create = importlib.import_module(
                "devsim.python_packages.model_create"
            ).CreateSolution
        create(device, region, name)

    _CONTINUITY_EQUATIONS: ClassVar[dict[str, str]] = {
        "Potential": "PotentialEquation",
        "Electrons": "ElectronContinuityEquation",
        "Holes": "HoleContinuityEquation",
    }

    def _interface_continuity(
        self, devsim: Any, sp: Any, device: str, interface: str, variable: str
    ) -> None:
        """Enforce continuity of *variable* across a region-region interface."""
        equation = self._CONTINUITY_EQUATIONS[variable]
        create = getattr(sp, "CreateContinuousInterfaceModel", None)
        if create is not None:
            model_name = create(device, interface, variable)
        else:  # pragma: no cover - version-dependent fallback
            model_name = f"continuous{variable}"
            devsim.interface_model(
                device=device,
                interface=interface,
                name=model_name,
                equation=f"{variable}@r0 - {variable}@r1",
            )
        devsim.interface_equation(
            device=device,
            interface=interface,
            name=equation,
            interface_model=model_name,
            type="continuous",
        )

    def _initialize_drift_diffusion(self) -> None:
        """Initial potential-only solve, then switch on drift-diffusion."""
        if self._dd_initialized:
            return
        if self._device is None:
            self.setup_device()
        devsim = require_devsim()
        sp = import_simple_physics()
        device = self._device
        if device is None:  # pragma: no cover - setup_device() above set it
            raise RuntimeError("DEVSIM device setup did not complete.")
        regions = self._device_regions()

        self._solve_dc(devsim)

        for region in regions:
            for carrier, intrinsic in (
                ("Electrons", "IntrinsicElectrons"),
                ("Holes", "IntrinsicHoles"),
            ):
                self._create_solution(sp, device, region, carrier)
                devsim.set_node_values(
                    device=device,
                    region=region,
                    name=carrier,
                    init_from=intrinsic,
                )
            sp.CreateSiliconDriftDiffusion(device, region)
        for spec in self._electrical_contacts:
            sp.CreateSiliconDriftDiffusionAtContact(
                device, self._contact_regions[spec.name], spec.name, True
            )
        for name in self._interface_regions:
            self._interface_continuity(devsim, sp, device, name, "Electrons")
            self._interface_continuity(devsim, sp, device, name, "Holes")
        self._solve_dc(devsim)
        self._dd_initialized = True

    @staticmethod
    def _source_name(contact: str) -> str:
        """Circuit voltage-source name attached to a contact."""
        return f"V_{contact}"

    def _set_bias(self, contact: str, bias: float) -> None:
        """Ramp the contact bias to the target in ``bias_step_v`` steps."""
        devsim = require_devsim()
        start = self._current_bias.get(contact, 0.0)
        delta = bias - start
        n_steps = max(1, math.ceil(abs(delta) / self.bias_step_v))
        for i in range(1, n_steps + 1):
            value = start + delta * i / n_steps
            devsim.circuit_alter(name=self._source_name(contact), value=value)
            self._solve_dc(devsim)
        self._current_bias[contact] = bias

    def _contact_current(self, devsim: Any, contact: str) -> float:
        """DC terminal current from the contact's circuit source (A/cm)."""
        return float(
            devsim.get_circuit_node_value(
                node=f"{self._source_name(contact)}.I", solution="dcop"
            )
        )

    def _contact_charge(self, devsim: Any, contact: str) -> float:
        """Contact charge from the potential equation in C per cm of depth."""
        return float(
            devsim.get_contact_charge(
                device=self._device, contact=contact, equation="PotentialEquation"
            )
        )

    def _collect_carriers(self, devsim: Any) -> CarrierMap:
        """Concatenate node fields across the device regions (coords in um)."""
        device = self._device
        columns: dict[str, list[NDArray[np.float64]]] = {
            "x": [],
            "y": [],
            "Electrons": [],
            "Holes": [],
            "Potential": [],
            "NetDoping": [],
        }
        region_names: list[str] = []
        for region in self._device_regions():
            n_nodes = 0
            for name, column in columns.items():
                values = np.asarray(
                    devsim.get_node_model_values(
                        device=device, region=region, name=name
                    ),
                    dtype=np.float64,
                )
                n_nodes = values.size
                column.append(values)
            region_names.extend([region] * n_nodes)
        return CarrierMap(
            x_um=np.asarray(np.concatenate(columns["x"]) / UM_TO_CM, dtype=np.float64),
            y_um=np.asarray(np.concatenate(columns["y"]) / UM_TO_CM, dtype=np.float64),
            region=region_names,
            electrons_cm3=np.concatenate(columns["Electrons"]),
            holes_cm3=np.concatenate(columns["Holes"]),
            potential_v=np.concatenate(columns["Potential"]),
            net_doping_cm3=np.concatenate(columns["NetDoping"]),
        )

    def _resolve_sweep_contact(self, contact: str | None) -> str:
        """Default to the first declared contact; reject unknown names."""
        specs = self._electrical_contacts
        if not specs:
            raise ValueError("No contacts declared. Call add_contact() first.")
        if contact is None:
            return str(specs[0].name)
        names = [s.name for s in specs]
        if contact not in names:
            raise ValueError(f"Unknown contact '{contact}'. Declared contacts: {names}")
        return contact

    def solve(self, bias: float = 0.0, *, contact: str | None = None) -> BiasPoint:
        """Solve the drift-diffusion system at one bias point.

        Args:
            bias: Bias voltage applied to the swept contact (V); the other
                contacts stay at 0 V.
            contact: Swept contact name (defaults to the first declared).

        Returns:
            The solved :class:`BiasPoint` including carrier maps, terminal
            currents, and the small-signal capacitance ``Im(I)/omega`` from
            the quasi-static AC solve.
        """
        contact = self._resolve_sweep_contact(contact)
        self._initialize_drift_diffusion()
        devsim = require_devsim()

        self._set_bias(contact, bias)
        carriers = self._collect_carriers(devsim)
        currents = {
            spec.name: self._contact_current(devsim, spec.name)
            for spec in self._electrical_contacts
        }
        charge = self._contact_charge(devsim, contact)

        # Small-signal AC solve at a quasi-static frequency: the terminal
        # admittance of the swept contact's unit-amplitude source gives
        # C = |Im(I)| / omega (the small-signal charge per volt). Only the
        # swept source carries AC amplitude; the others stay AC-grounded.
        for spec in self._electrical_contacts:
            devsim.circuit_alter(
                name=self._source_name(spec.name),
                param="acreal",
                value=1.0 if spec.name == contact else 0.0,
            )
        freq = self.small_signal_freq_hz
        devsim.solve(type="ac", frequency=freq)
        current_imag = float(
            devsim.get_circuit_node_value(
                node=f"{self._source_name(contact)}.I", solution="ssac_imag"
            )
        )
        capacitance = abs(current_imag) / (2.0 * math.pi * freq)

        return BiasPoint(
            bias_v=bias,
            carriers=carriers,
            currents_a_per_cm=currents,
            charge_c_per_cm=charge,
            capacitance_f_per_cm=capacitance,
        )

    def sweep(
        self, biases: list[float] | Any, *, contact: str | None = None
    ) -> BiasSweepResult:
        """Solve a bias sweep and return carrier maps and C(V) per point.

        Args:
            biases: Bias voltages (V) applied in order to the swept contact.
            contact: Swept contact name (defaults to the first declared).

        Returns:
            :class:`BiasSweepResult` with one :class:`BiasPoint` per bias.
        """
        contact = self._resolve_sweep_contact(contact)
        points = [self.solve(float(bias), contact=contact) for bias in biases]
        return BiasSweepResult(contact=contact, points=points)


__all__ = ["ChargeTransportSim"]
