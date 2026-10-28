"""Reusing the mesh of one simulation in another (gsim#389), without Gmsh.

The source is meshed by the real ``mesh()``: real ``validate_config``, real
port extraction and the fingerprint ``mesh()`` records. Only
``generate_mesh`` is replaced, by a fake that writes a small ``.msh`` and
returns a real ``MeshResult``. Once the source has meshed, the fake can be
told to fail, which shows that ``reuse_mesh`` never meshes.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import gdsfactory as gf
import pytest

from gsim.common.stack import LayerStack
from gsim.hashing import compute_dir_digest, compute_input_hash
from gsim.palace import BoundaryModeSim, DrivenSim, EigenmodeSim, ElectrostaticSim
from gsim.palace.mesh.generator import MeshResult
from gsim.palace.mesh.metadata import update_field_dofs, write_metadata
from gsim.palace.models.results import SimulationResult
from tests._electrode_layouts import stack as generic_stack

# o1 and o2: lumped ports on m1 at either end of the m1 strip
TWO_PORTS = {
    "o1": {"layer": "m1", "length": 1.0},
    "o2": {"layer": "m1", "length": 1.0},
}
NODES, TETS = 1234, 4321


class FakeGmsh:
    """Stands in for ``generate_mesh``; counts its calls.

    Writes ``<model_name>.msh`` (bytes that change with every call) and
    ``metadata.json``, and returns a real ``MeshResult`` with one lumped
    ``P{i}`` port surface per extracted port.
    """

    def __init__(self) -> None:
        """Start with no calls, meshing allowed."""
        self.calls = 0
        self.allowed = True

    def __call__(self, **kwargs) -> MeshResult:
        """Write the fake mesh files and return what Gmsh would."""
        if not self.allowed:
            raise AssertionError("generate_mesh was called")
        self.calls += 1
        output_dir = Path(kwargs["output_dir"])
        model_name = kwargs["model_name"]
        ports = kwargs["ports"]

        mesh_path = output_dir / f"{model_name}.msh"
        mesh_path.write_bytes(
            f"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n% fake {self.calls}\n".encode()
        )
        groups: dict = {
            "volumes": {"airbox": {"phys_group": 1}, "sio2": {"phys_group": 2}},
            "conductor_surfaces": {"m1_xy": {"phys_group": 3}},
            "port_surfaces": {
                f"P{i}": {"phys_group": 10 + i, "type": "lumped"}
                for i, _ in enumerate(ports, start=1)
            },
            "boundary_surfaces": {"absorbing": {"phys_group": 4}},
        }
        symmetry = None
        if (plane := kwargs.get("symmetry_plane")) is not None:
            symmetry = {
                "phys_group": 5,
                "axis": plane.axis,
                "position": plane.position,
                "kind": plane.kind,
                "keep": plane.keep,
            }
            groups["boundary_surfaces"]["symmetry"] = dict(symmetry)
        port_info = [
            {"portnumber": i, "name": port.name, "Z0": port.impedance, "type": "lumped"}
            for i, port in enumerate(ports, start=1)
        ]
        stats = {"nodes": NODES, "elements": 5000, "tetrahedra": TETS}
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
            periodic_axis=kwargs.get("periodic_axis"),
            symmetry_plane=symmetry,
        )


@pytest.fixture
def stack() -> LayerStack:
    """Conductors m1 and m2, a conductive via v1, and an oxide at eps 3.9."""
    return generic_stack()


@pytest.fixture
def component() -> gf.Component:
    """A strip on m1 and one on m2, with ports o1 and o2 at the ends of m1."""
    gf.gpdk.PDK.activate()
    c = gf.Component()
    c.add_polygon([(0, 0), (10, 0), (10, 1), (0, 1)], layer=(1, 0))
    c.add_polygon([(0, 3), (10, 3), (10, 4), (0, 4)], layer=(2, 0))
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


@pytest.fixture
def fake_gmsh(monkeypatch) -> FakeGmsh:
    """Replace Gmsh meshing with :class:`FakeGmsh`."""
    fake = FakeGmsh()
    monkeypatch.setattr("gsim.palace.mesh.generator.generate_mesh", fake)
    return fake


@pytest.fixture
def make_sim(stack, component):
    """Build a driven or eigenmode sim of the layout, ready to mesh.

    Each sim gets its own copy of the component and the stack. ``ports`` maps
    each declared lumped port to its ``add_port`` settings.
    """

    def _make(cls, folder: Path | None = None, ports=TWO_PORTS):
        sim = cls(absorbing_boundary=False)
        if folder is not None:
            sim.set_output_dir(folder)
        sim.set_geometry(component.copy())
        sim.set_stack(stack.model_copy(deep=True))
        for name, settings in ports.items():
            sim.add_port(name, **settings)
        if cls is EigenmodeSim:
            sim.set_eigenmode(target=5e9)
        else:
            sim.set_driven(fmin=7e9, fmax=10e9)
        return sim

    return _make


@pytest.fixture
def meshed_eigen(make_sim, fake_gmsh, tmp_path) -> EigenmodeSim:
    """An eigenmode sim with two ports, meshed by the real ``mesh()``."""
    sim = make_sim(EigenmodeSim, tmp_path / "eigen")
    sim.mesh(verbose=False)
    assert fake_gmsh.calls == 1
    return sim


def _without_dofs(stats: dict) -> dict:
    """Mesh stats without the DOF estimate, which depends on the problem type."""
    return {k: v for k, v in stats.items() if k != "field_dofs"}


def _listing(folder: Path) -> list[str]:
    """Every path under a folder, relative to it."""
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*"))


_STATE = (
    "geometry",
    "stack",
    "materials",
    "mesh_config",
    "ports",
    "cpw_ports",
    "wave_ports",
    "two_terminal_ports",
    "_stack_kwargs",
    "_airbox_config",
    "_pec_blocks",
    "_symmetry_planes",
    "_last_mesh_result",
    "_last_ports",
    "_mesh_fingerprint",
    "_output_dir",
)


def _state(sim) -> dict:
    """The objects a sim holds, and the values that could change in place."""
    return {
        "objects": {name: id(getattr(sim, name)) for name in _STATE},
        "ports": [p.model_dump() for p in sim.ports],
        "last_ports": copy.deepcopy(sim._last_ports),
        "mesh_stats": copy.deepcopy(sim._last_mesh_result.mesh_stats),
        "fingerprint": dict(sim._mesh_fingerprint or {}),
        "flags": (sim._mesh_reused, sim._configured_ports),
    }


# ----- case 2: the four allowed pairs ------------------------------------------------


@pytest.mark.parametrize("source_cls", [DrivenSim, EigenmodeSim])
@pytest.mark.parametrize("receiver_cls", [DrivenSim, EigenmodeSim])
def test_all_four_pairs_return_the_receivers_mesh(
    source_cls, receiver_cls, make_sim, fake_gmsh, tmp_path, caplog
) -> None:
    source = make_sim(source_cls, tmp_path / "source")
    source.mesh(verbose=False)
    receiver = make_sim(receiver_cls, tmp_path / "receiver")
    fake_gmsh.allowed = False
    caplog.set_level(logging.INFO)
    caplog.clear()

    result = receiver.reuse_mesh(source)

    src = source._last_mesh_result
    own = receiver._last_mesh_result
    assert isinstance(result, SimulationResult)
    assert result.mesh_path == tmp_path / "receiver" / "palace.msh"
    assert result.output_dir == tmp_path / "receiver"
    assert result.config_path is None
    assert receiver.has_mesh
    assert receiver.mesh_path == result.mesh_path
    assert receiver.mesh_groups == src.groups
    assert own is not src
    assert own.config_path is None
    # equal to the source's, but copies
    assert result.port_info == src.port_info
    assert own.port_info is not src.port_info
    assert own.port_info[0] is not src.port_info[0]
    assert _without_dofs(result.mesh_stats) == _without_dofs(src.mesh_stats)
    assert own.mesh_stats is not src.mesh_stats
    assert result.mesh_stats["field_dofs"]["problem_type"] == receiver.simulation_type
    assert result.metadata == {
        "schema_version": 1,
        "solver": "palace",
        "mesh": result.mesh_stats,
    }
    assert _without_dofs(result.metadata["mesh"]) == _without_dofs(src.metadata["mesh"])
    assert own.metadata is not src.metadata
    assert receiver._mesh_reused is True
    assert receiver._mesh_fingerprint == source._mesh_fingerprint
    assert receiver._mesh_fingerprint is not source._mesh_fingerprint
    reuse_lines = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and "Reusing mesh" in r.getMessage()
    ]
    assert reuse_lines == [
        f"Reusing mesh of {source_cls.__name__} ({src.mesh_path}): "
        f"{NODES:,} nodes \u00b7 {TETS:,} tets"
    ]


# ----- case 3: refusals --------------------------------------------------------------


@pytest.mark.parametrize("cls", [ElectrostaticSim, BoundaryModeSim])
def test_reuse_refuses_a_receiver_of_another_class(cls, meshed_eigen, tmp_path) -> None:
    receiver = cls()
    receiver.set_output_dir(tmp_path / "other")

    with pytest.raises(ValueError, match=rf"{cls.__name__}.*mesh\(\)"):
        receiver.reuse_mesh(meshed_eigen)


@pytest.mark.parametrize("cls", [ElectrostaticSim, BoundaryModeSim])
def test_reuse_refuses_a_source_of_another_class(cls, make_sim, tmp_path) -> None:
    receiver = make_sim(DrivenSim, tmp_path / "driven")

    with pytest.raises(ValueError, match=rf"{cls.__name__}.*mesh\(\)"):
        receiver.reuse_mesh(cls())


@pytest.mark.parametrize("kind", ["str", "path", "mesh_result"])
def test_reuse_refuses_a_file_or_a_mesh_result(
    kind, meshed_eigen, make_sim, tmp_path
) -> None:
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    mesh_result = meshed_eigen._last_mesh_result
    source = {
        "str": str(mesh_result.mesh_path),
        "path": Path(mesh_result.mesh_path),
        "mesh_result": mesh_result,
    }[kind]

    with pytest.raises(TypeError, match="physical-group or port mappings"):
        receiver.reuse_mesh(source)


def test_reuse_refuses_an_unmeshed_source(make_sim, tmp_path) -> None:
    source = make_sim(EigenmodeSim, tmp_path / "eigen")
    receiver = make_sim(DrivenSim, tmp_path / "driven")

    with pytest.raises(
        ValueError,
        match=r"The source EigenmodeSim has no mesh\. Call mesh\(\) on it first\.",
    ):
        receiver.reuse_mesh(source)


def test_reuse_refuses_its_own_mesh(meshed_eigen) -> None:
    with pytest.raises(ValueError, match="its own mesh"):
        meshed_eigen.reuse_mesh(meshed_eigen)


def test_reuse_refuses_a_receiver_without_output_dir(meshed_eigen, make_sim) -> None:
    receiver = make_sim(DrivenSim)

    with pytest.raises(
        ValueError,
        match=r"Output directory not set\. Call set_output_dir\(\) first\.",
    ):
        receiver.reuse_mesh(meshed_eigen)


def test_reuse_refuses_a_driven_receiver_of_a_mesh_without_ports(
    make_sim, fake_gmsh, tmp_path
) -> None:
    """R1: a driven sim can never gain a port surface on a reused mesh."""
    source = make_sim(EigenmodeSim, tmp_path / "eigen", ports={})
    source.mesh(verbose=False)
    fake_gmsh.allowed = False

    with pytest.raises(ValueError, match="no port surfaces"):
        make_sim(DrivenSim, tmp_path / "driven", ports={}).reuse_mesh(source)

    # an eigenmode receiver needs no port
    receiver = make_sim(EigenmodeSim, tmp_path / "eigen2", ports={})
    receiver.reuse_mesh(source)
    assert receiver.has_mesh


# ----- case 4: a refused call changes nothing ----------------------------------------


def _same_folder(source, receiver, _tmp_path) -> None:
    receiver.set_output_dir(source.output_dir)


def _same_folder_once_resolved(_source, receiver, tmp_path) -> None:
    receiver.set_output_dir(tmp_path / "driven" / ".." / "eigen")


def _folder_inside_the_source_folder(source, receiver, _tmp_path) -> None:
    receiver.set_output_dir(source.output_dir / "driven")


def _folder_containing_the_source_folder(_source, receiver, tmp_path) -> None:
    receiver.set_output_dir(tmp_path)


def _missing_source_mesh_file(source, _receiver, _tmp_path) -> None:
    source.mesh_path.unlink()


def _receiver_only_port(_source, receiver, _tmp_path) -> None:
    receiver.add_port("o3", layer="m1", length=1.0)


def _port_of_another_kind(_source, receiver, _tmp_path) -> None:
    receiver.ports = [p for p in receiver.ports if p.name != "o1"]
    receiver.add_cpw_port("o1", layer="m1", s_width=1.0, gap_width=1.0)


def _port_of_another_geometry(_source, receiver, _tmp_path) -> None:
    receiver.add_port("o1", layer="m2", length=1.0)


def _source_edited_after_mesh(source, _receiver, _tmp_path) -> None:
    source.component.add_polygon([(0, 6), (2, 6), (2, 7), (0, 7)], layer=(1, 0))


def _source_without_fingerprint(source, _receiver, _tmp_path) -> None:
    source._mesh_fingerprint = None


def _mesh_without_port_surfaces(source, _receiver, _tmp_path) -> None:
    source._last_mesh_result.groups["port_surfaces"] = {}


@pytest.mark.parametrize(
    ("cause", "error", "match"),
    [
        (_same_folder, ValueError, "is the output directory"),
        (_same_folder_once_resolved, ValueError, "is the output directory"),
        (_folder_inside_the_source_folder, ValueError, "is inside the output"),
        (_folder_containing_the_source_folder, ValueError, "contains the output"),
        (_missing_source_mesh_file, FileNotFoundError, "does not exist"),
        (_receiver_only_port, ValueError, "'o3' has no surface in the reused mesh"),
        (_port_of_another_kind, ValueError, "Port 'o1' .*does not match"),
        (_port_of_another_geometry, ValueError, "Port 'o1' .*does not match"),
        (
            _source_edited_after_mesh,
            ValueError,
            r"changed after its mesh was made \(geometry \(layout polygons\)\)",
        ),
        (_source_without_fingerprint, ValueError, "has no record of the inputs"),
        (_mesh_without_port_surfaces, ValueError, "no port surfaces"),
    ],
    ids=lambda value: getattr(value, "__name__", None),
)
def test_refused_call_leaves_the_receiver_unchanged(
    cause, error, match, make_sim, fake_gmsh, tmp_path
) -> None:
    source = make_sim(EigenmodeSim, tmp_path / "eigen")
    source.mesh(verbose=False)
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.mesh(verbose=False)
    cause(source, receiver, tmp_path)
    fake_gmsh.allowed = False
    before = _state(receiver)
    listing = _listing(receiver.output_dir)

    with pytest.raises(error, match=match):
        receiver.reuse_mesh(source)

    assert _state(receiver) == before
    # no file written, and no staging folder left behind
    assert _listing(receiver.output_dir) == listing


# ----- case 5: last call wins, chaining ----------------------------------------------


def test_last_call_wins(make_sim, fake_gmsh, tmp_path) -> None:
    first = make_sim(EigenmodeSim, tmp_path / "first")
    first.mesh(verbose=False)
    second = make_sim(DrivenSim, tmp_path / "second")
    second.mesh(verbose=False)
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.mesh(verbose=False)
    own_bytes = receiver.mesh_path.read_bytes()
    fake_gmsh.allowed = False

    receiver.reuse_mesh(first)
    assert receiver.mesh_path.read_bytes() == first.mesh_path.read_bytes()
    assert receiver.mesh_path.read_bytes() != own_bytes
    assert receiver._mesh_reused is True

    receiver.reuse_mesh(second)
    assert receiver.mesh_path.read_bytes() == second.mesh_path.read_bytes()
    assert receiver._last_mesh_result.mesh_stats["field_dofs"]["problem_type"] == (
        "driven"
    )


def test_chain_eigen_to_driven_to_driven(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    fake_gmsh.allowed = False
    driven = make_sim(DrivenSim, tmp_path / "driven")
    driven.reuse_mesh(meshed_eigen)
    driven2 = make_sim(DrivenSim, tmp_path / "driven2")

    driven2.reuse_mesh(driven)
    config_path = driven2.write_config(validate_mesh=False)

    assert config_path == tmp_path / "driven2" / "config.json"
    config = json.loads(config_path.read_text())
    assert config["Problem"]["Type"] == "Driven"
    assert [p["Excitation"] for p in config["Boundaries"]["LumpedPort"]] == [1, 2]
    assert driven2.mesh_path.read_bytes() == meshed_eigen.mesh_path.read_bytes()
    # the chain carries the fingerprint of the original mesh
    assert driven2._mesh_fingerprint == meshed_eigen._mesh_fingerprint


# ----- case 6: mesh() after reuse, source re-mesh ------------------------------------


def test_mesh_after_reuse_remeshes_the_receiver(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.reuse_mesh(meshed_eigen)
    calls = fake_gmsh.calls

    receiver.mesh(verbose=False)

    assert fake_gmsh.calls == calls + 1
    assert receiver._mesh_reused is False
    assert receiver._mesh_fingerprint == receiver._current_mesh_fingerprint(
        receiver.stack
    )
    assert receiver.mesh_path.read_bytes() != meshed_eigen.mesh_path.read_bytes()


def test_source_remesh_leaves_the_receiver_intact(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    receiver.reuse_mesh(meshed_eigen)
    snapshot = receiver.mesh_path.read_bytes()
    groups = copy.deepcopy(receiver.mesh_groups)
    calls = fake_gmsh.calls

    meshed_eigen.mesh(verbose=False)

    assert fake_gmsh.calls == calls + 1
    assert meshed_eigen.mesh_path.read_bytes() != snapshot
    assert receiver.mesh_path.read_bytes() == snapshot
    assert receiver.mesh_groups == groups
    assert receiver._mesh_reused is True
    assert receiver.write_config(validate_mesh=False).parent == tmp_path / "driven"


# ----- case 7: nothing is shared -----------------------------------------------------


def test_nothing_is_shared_after_reuse(make_sim, fake_gmsh, tmp_path) -> None:
    source = make_sim(EigenmodeSim, tmp_path / "eigen")
    source.set_material("sio2", permittivity=3.9)
    source.set_airbox(margin_x=5, margin_y=5, z_above=5, z_below=0)
    source.add_pec(gds_layer=(9, 0), from_layer="m1", to_layer="m2")
    source.add_symmetry_plane(axis="x", position=5.0, verify_symmetry=False)
    source.mesh(verbose=False)
    receiver = make_sim(DrivenSim, tmp_path / "driven", ports={})
    fake_gmsh.allowed = False

    receiver.reuse_mesh(source)

    pairs = [
        (receiver.geometry, source.geometry),
        (receiver.component, source.component),
        (receiver.stack, source.stack),
        (receiver.stack.layers["m1"], source.stack.layers["m1"]),
        (receiver.stack.materials, source.stack.materials),
        (receiver.materials, source.materials),
        (receiver.materials["sio2"], source.materials["sio2"]),
        (receiver._stack_kwargs, source._stack_kwargs),
        (receiver._airbox_config, source._airbox_config),
        (receiver.mesh_config, source.mesh_config),
        (receiver._pec_blocks[0], source._pec_blocks[0]),
        (receiver._symmetry_planes[0], source._symmetry_planes[0]),
        *zip(receiver.ports, source.ports, strict=True),
        *zip(receiver._last_ports, source._last_ports, strict=True),
    ]
    own, src = receiver._last_mesh_result, source._last_mesh_result
    pairs += [
        (own, src),
        (own.port_info, src.port_info),
        (own.port_info[0], src.port_info[0]),
        (own.mesh_stats, src.mesh_stats),
        (own.metadata, src.metadata),
        (own.groups, src.groups),
        (own.groups["port_surfaces"], src.groups["port_surfaces"]),
        (own.symmetry_plane, src.symmetry_plane),
    ]
    for mine, theirs in pairs:
        assert mine is not theirs
    assert own.symmetry_plane == src.symmetry_plane
    assert receiver._airbox_config == source._airbox_config

    # the receiver changes; the source does not see it
    source_stats = copy.deepcopy(src.mesh_stats)
    source_fingerprint = dict(source._mesh_fingerprint)
    receiver.set_material("sio2", permittivity=4.1)
    receiver.ports[0].impedance = 25.0
    receiver.ports[0].excited = True
    receiver.write_config(validate_mesh=False)
    receiver.component.add_polygon([(0, 6), (2, 6), (2, 7), (0, 7)], layer=(1, 0))
    assert source.materials["sio2"].permittivity == 3.9
    assert source.ports[0].excited is False
    assert source.stack.materials["sio2"]["permittivity"] == 3.9
    assert source.ports[0].impedance == 50.0
    assert src.mesh_stats == source_stats
    assert source._current_mesh_fingerprint(source.stack) == source_fingerprint

    # the source changes; the receiver does not see it
    source.ports[1].impedance = 75.0
    source.stack.layers["m1"].zmax = 1.5
    source._airbox_config["margin_x"] = 9.0
    src.groups["volumes"]["extra"] = {"phys_group": 99}
    assert receiver.ports[1].impedance == 50.0
    assert receiver.stack.layers["m1"].zmax == 1.0
    assert receiver._airbox_config["margin_x"] == 5
    assert "extra" not in receiver.mesh_groups["volumes"]


# ----- case 8: port merge ------------------------------------------------------------


def test_port_merge_copies_undeclared_ports_and_keeps_declared_ones(
    meshed_eigen, make_sim, tmp_path
) -> None:
    receiver = make_sim(
        DrivenSim,
        tmp_path / "driven",
        ports={"o2": {"layer": "m1", "length": 1.0, "impedance": 25.0}},
    )
    own_o2 = receiver.ports[0]

    receiver.reuse_mesh(meshed_eigen)

    # in the source's order; o1 is a copy of the source's, o2 the receiver's own
    assert [p.name for p in receiver.ports] == ["o1", "o2"]
    assert receiver.ports[1] is own_o2
    assert own_o2.impedance == 25.0
    assert own_o2.excited is True
    copied = receiver.ports[0]
    assert copied is not meshed_eigen.ports[0]
    assert copied.model_dump() == meshed_eigen.ports[0].model_dump()
    assert copied.excited is False  # the eigenmode source's flag
    # the port settings reach the receiver's own extracted ports and Z0
    by_name = {p.name: p for p in receiver._last_ports}
    assert (by_name["o1"].excited, by_name["o1"].impedance) == (False, 50.0)
    assert (by_name["o2"].excited, by_name["o2"].impedance) == (True, 25.0)
    z0 = {e["name"]: e["Z0"] for e in receiver._last_mesh_result.port_info}
    assert z0 == {"o1": 50.0, "o2": 25.0}
    # the source keeps its own
    assert [(p.excited, p.impedance) for p in meshed_eigen._last_ports] == [
        (False, 50.0),
        (False, 50.0),
    ]
    assert [e["Z0"] for e in meshed_eigen._last_mesh_result.port_info] == [50.0, 50.0]


def test_port_merge_gives_an_empty_receiver_the_source_ports(
    meshed_eigen, make_sim, tmp_path
) -> None:
    receiver = make_sim(DrivenSim, tmp_path / "driven", ports={})

    receiver.reuse_mesh(meshed_eigen)

    assert [p.model_dump() for p in receiver.ports] == [
        p.model_dump() for p in meshed_eigen.ports
    ]
    assert not any(p.excited for p in receiver.ports)
    # equal declarations: the settings sync changes nothing
    assert receiver._last_ports == meshed_eigen._last_ports
    assert (
        receiver._last_mesh_result.port_info == meshed_eigen._last_mesh_result.port_info
    )


# ----- case 10: files ----------------------------------------------------------------


def test_files_copy_the_mesh_and_regenerate_metadata(
    make_sim, fake_gmsh, tmp_path
) -> None:
    source = make_sim(EigenmodeSim, tmp_path / "eigen")
    source.mesh(verbose=False, model_name="chip")
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    fake_gmsh.allowed = False

    receiver.reuse_mesh(source)

    folder = tmp_path / "driven"
    assert _listing(folder) == ["chip.msh", "metadata.json"]
    assert receiver.mesh_path == folder / "chip.msh"
    assert receiver._last_mesh_result.model_name == "chip"
    assert (folder / "chip.msh").read_bytes() == (
        tmp_path / "eigen" / "chip.msh"
    ).read_bytes()
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata == receiver._last_mesh_result.metadata
    assert "simulation" not in metadata
    assert metadata["mesh"]["field_dofs"]["problem_type"] == "driven"
    source_metadata = json.loads((tmp_path / "eigen" / "metadata.json").read_text())
    assert source_metadata["mesh"]["field_dofs"]["problem_type"] == "eigenmode"


def test_files_of_the_source_stay_unchanged_when_the_receiver_writes(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    digest = compute_dir_digest(tmp_path / "eigen")
    receiver = make_sim(DrivenSim, tmp_path / "driven")
    fake_gmsh.allowed = False

    receiver.reuse_mesh(meshed_eigen)
    receiver.write_config(validate_mesh=False)

    assert (tmp_path / "driven" / "config.json").is_file()
    assert compute_dir_digest(tmp_path / "eigen") == digest


def test_files_of_identical_receivers_hash_equal(
    meshed_eigen, make_sim, fake_gmsh, tmp_path
) -> None:
    fake_gmsh.allowed = False
    for name in ("a", "b"):
        receiver = make_sim(DrivenSim, tmp_path / name)
        receiver.reuse_mesh(meshed_eigen)
        receiver.write_config(validate_mesh=False)

    assert compute_input_hash(tmp_path / "a", "palace") == compute_input_hash(
        tmp_path / "b", "palace"
    )
