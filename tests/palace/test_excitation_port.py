"""``set_driven(excitation_port=...)`` excites exactly the named port.

With ``excitation_port`` set, a driven config writes ``Excitation`` on that
port only, whatever its kind (lumped, CPW, wave, two-terminal). With
``None`` each port keeps its own ``excited`` flag. Non-driven sims write no
excitation and leave the flags the user declared unchanged.

The sim-level tests replace ``generate_mesh`` by a fake that writes a small
``.msh`` and returns a real ``MeshResult``, so they run without Gmsh.
"""

from __future__ import annotations

import json
from pathlib import Path

import gdsfactory as gf
import pytest

from gsim.common.stack import LayerStack
from gsim.palace import DrivenSim, EigenmodeSim
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.mesh.generator import MeshResult
from gsim.palace.mesh.metadata import update_field_dofs, write_metadata
from gsim.palace.models import DrivenConfig, EigenmodeConfig
from gsim.palace.ports.config import PalacePort, PortType
from tests._electrode_layouts import stack as generic_stack

# Config-level: o1 of the given kind, o2 always a lumped port on m1.
KINDS = ("lumped", "cpw", "wave", "two_terminal")


def _first_port(kind: str) -> tuple[PalacePort, dict]:
    """Port ``o1`` of ``kind`` (declared excited) and its port surface group."""
    if kind == "lumped":
        return PalacePort(name="o1", layer="m1", length=1.0), {
            "phys_group": 10,
            "type": "lumped",
        }
    if kind == "wave":
        return PalacePort(name="o1", port_type=PortType.WAVEPORT, layer="m1"), {
            "phys_group": 10,
        }
    port = PalacePort(
        name="o1",
        layer="m1",
        multi_element=True,
        centers=[(0.0, 1.0), (0.0, -1.0)],
        directions=["+Y", "-Y"],
    )
    elements = [
        {"phys_group": 10, "direction": "+Y"},
        {"phys_group": 12, "direction": "-Y"},
    ]
    return port, {"type": kind, "elements": elements}


def _generate(
    tmp_path: Path,
    kind: str,
    excitation_port: str | None,
    simulation_type: str = "driven",
) -> tuple[dict, list[PalacePort]]:
    """Write the config of ports o1 (``kind``) and o2 (lumped), both excited."""
    first, first_group = _first_port(kind)
    ports = [first, PalacePort(name="o2", layer="m1", length=1.0)]
    groups = {
        "volumes": {"airbox": {"phys_group": 1}},
        "conductor_surfaces": {"m1_xy": {"phys_group": 3}},
        "pec_surfaces": {},
        "port_surfaces": {
            "P1": first_group,
            "P2": {"phys_group": 11, "type": "lumped"},
        },
        "boundary_surfaces": {"absorbing": {"phys_group": 4}},
    }
    config_path = generate_palace_config(
        groups=groups,
        ports=ports,
        port_info=[],
        stack=generic_stack(),
        output_path=tmp_path,
        model_name="palace",
        fmax=10e9,
        simulation_type=simulation_type,
        driven_config=DrivenConfig(fmax=10e9, excitation_port=excitation_port),
        eigenmode_config=(
            EigenmodeConfig(num_modes=2, target=4e9)
            if simulation_type == "eigenmode"
            else None
        ),
    )
    return json.loads(config_path.read_text()), ports


def _excited_indices(config: dict) -> list[int]:
    """Indices of the port entries that carry an ``Excitation``."""
    boundaries = config["Boundaries"]
    entries = [*boundaries.get("LumpedPort", []), *boundaries.get("WavePort", [])]
    return sorted(e["Index"] for e in entries if e.get("Excitation"))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    ("excitation_port", "expected"),
    [("o1", [1]), ("o2", [2]), (None, [1, 2])],
)
def test_excitation_port_excites_only_the_named_port(
    tmp_path, kind, excitation_port, expected
):
    """Only the named port is excited; with None, every declared flag holds."""
    config, ports = _generate(tmp_path, kind, excitation_port)
    assert _excited_indices(config) == expected
    # The ports passed in keep the flags they were declared with.
    assert [p.excited for p in ports] == [True, True]


def test_eigenmode_config_writes_no_excitation(tmp_path):
    """A non-driven config excites nothing and leaves the ports' flags."""
    config, ports = _generate(tmp_path, "lumped", None, simulation_type="eigenmode")
    assert _excited_indices(config) == []
    assert [p.excited for p in ports] == [True, True]


# Sim-level: lumped ports o1 and o2 on m1, meshed by a fake generate_mesh.


def _fake_generate_mesh(**kwargs) -> MeshResult:
    """Write a stand-in ``.msh`` and return the ``MeshResult`` Gmsh would."""
    output_dir = Path(kwargs["output_dir"])
    model_name = kwargs["model_name"]
    ports = kwargs["ports"]
    mesh_path = output_dir / f"{model_name}.msh"
    mesh_path.write_bytes(b"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
    groups = {
        "volumes": {"airbox": {"phys_group": 1}, "sio2": {"phys_group": 2}},
        "conductor_surfaces": {"m1_xy": {"phys_group": 3}},
        "port_surfaces": {
            f"P{i}": {"phys_group": 10 + i, "type": port.port_type.value}
            for i, port in enumerate(ports, start=1)
        },
        "boundary_surfaces": {"absorbing": {"phys_group": 4}},
    }
    port_info = [
        {
            "portnumber": i,
            "name": port.name,
            "Z0": port.impedance,
            "type": port.port_type.value,
        }
        for i, port in enumerate(ports, start=1)
    ]
    stats = {"nodes": 1234, "elements": 5000, "tetrahedra": 4321}
    numerical = kwargs.get("numerical_config")
    update_field_dofs(
        stats,
        field_order=numerical.order if numerical else 2,
        problem_type=kwargs["simulation_type"],
    )
    return MeshResult(
        mesh_path=mesh_path,
        port_info=port_info,
        mesh_stats=stats,
        metadata=write_metadata(stats, output_dir, None),
        groups=groups,
        output_dir=output_dir,
        model_name=model_name,
        fmax=kwargs["fmax"],
    )


@pytest.fixture
def fake_gmsh(monkeypatch):
    """Mesh with ``_fake_generate_mesh`` instead of Gmsh."""
    monkeypatch.setattr("gsim.palace.mesh.generator.generate_mesh", _fake_generate_mesh)


@pytest.fixture
def stack() -> LayerStack:
    """Conductors m1 and m2, a via v1, and an oxide."""
    return generic_stack()


@pytest.fixture
def component() -> gf.Component:
    """A strip on m1 with ports o1 and o2 at its ends."""
    gf.gpdk.PDK.activate()
    c = gf.Component()
    c.add_polygon([(0, 0), (10, 0), (10, 1), (0, 1)], layer=(1, 0))
    for name, x, orientation in (("o1", 0, 180), ("o2", 10, 0)):
        c.add_port(
            name=name,
            center=(x, 0.5),
            width=1.0,
            orientation=orientation,
            layer=(1, 0),
            port_type="electrical",
        )
    return c


def _with_two_ports(sim, component, stack, tmp_path):
    """Give ``sim`` the strip, the stack, a folder and ports o1, o2 (excited)."""
    sim.set_output_dir(tmp_path)
    sim.set_geometry(component)
    sim.set_stack(stack)
    sim.add_port("o1", layer="m1", length=1.0, excited=True)
    sim.add_port("o2", layer="m1", length=1.0, excited=True)
    return sim


def _config(sim) -> dict:
    return json.loads(sim.write_config(validate_mesh=False).read_text())


@pytest.mark.usefixtures("fake_gmsh")
def test_driven_sim_excites_the_excitation_port(tmp_path, component, stack):
    """``excitation_port`` picks the port; going back to None restores both."""
    sim = _with_two_ports(DrivenSim(), component, stack, tmp_path)
    sim.set_driven(fmin=1e9, fmax=10e9, excitation_port="o1")
    sim.mesh(verbose=False)
    assert _excited_indices(_config(sim)) == [1]

    sim.set_driven(fmin=1e9, fmax=10e9, excitation_port=None)
    assert _excited_indices(_config(sim)) == [1, 2]
    assert [p.excited for p in sim.ports] == [True, True]


@pytest.mark.usefixtures("fake_gmsh")
def test_eigenmode_sim_keeps_declared_excited_flags(tmp_path, component, stack):
    """Meshing an eigenmode sim leaves ``PortConfig.excited`` as declared."""
    sim = _with_two_ports(EigenmodeSim(), component, stack, tmp_path)
    sim.set_eigenmode(num_modes=2, target=5e9)
    sim.mesh(verbose=False)
    assert [p.excited for p in sim.ports] == [True, True]
    assert _excited_indices(_config(sim)) == []


def _excitation_errors(sim) -> list[str]:
    return [e for e in sim.validate_config().errors if "Excitation port" in e]


@pytest.mark.parametrize("name", ["w1", "o1"])
def test_wave_and_two_terminal_names_pass_validation(name):
    """A wave port name and a two-terminal ``plus_port`` are valid names."""
    sim = DrivenSim()
    sim.add_wave_port("w1", layer="m1")
    sim.add_two_terminal_port("o1", "o2", layer="m1")
    sim.set_driven(excitation_port=name)
    assert _excitation_errors(sim) == []


def test_unknown_excitation_port_fails_validation():
    """A name that is no port of the sim is still refused."""
    sim = DrivenSim()
    sim.add_wave_port("w1", layer="m1")
    sim.add_two_terminal_port("o1", "o2", layer="m1")
    sim.set_driven(excitation_port="nope")
    (error,) = _excitation_errors(sim)
    assert "Excitation port 'nope' not found" in error
    assert "'w1'" in error
    assert "'o1'" in error
