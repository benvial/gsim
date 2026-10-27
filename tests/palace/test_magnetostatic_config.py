"""Tests for the magnetostatic Palace config (``SurfaceCurrent`` and ``PEC``)."""

from __future__ import annotations

import json

import pytest

from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.models import MagnetostaticConfig, MagnetostaticSolverConfig
from gsim.palace.ports.config import PalacePort, PortGeometry, PortType
from tests._electrode_layouts import stack

# Group numbers: m1 shell 10 11, m2 shell 12 13, PEC block 14, via sides 15 16,
# source sheets 20 21, outer wall 30 31, volumes 1 2.
OUTER = [30, 31]


def _groups() -> dict:
    """A two-layer layout with a via, a PEC block and two source sheets."""
    return {
        "volumes": {
            "airbox": {"phys_group": 1, "tags": [1]},
            "v1": {"phys_group": 2, "tags": [2], "is_via": True},
        },
        "conductor_surfaces": {
            "m1_xy": {"phys_group": 10, "tags": []},
            "m1_z": {"phys_group": 11, "tags": []},
            "m2_xy": {"phys_group": 12, "tags": []},
            "m2_z": {"phys_group": 13, "tags": []},
        },
        "pec_surfaces": {"shield_xy": {"phys_group": 14, "tags": []}},
        "via_boundary_surfaces": {"v1": [15, 16]},
        "port_surfaces": {
            "P1": {"phys_group": 20, "tags": [], "type": "surface_current"},
            "P2": {"phys_group": 21, "tags": [], "type": "surface_current"},
        },
        "boundary_surfaces": {"absorbing": {"phys_group": OUTER, "tags": []}},
    }


def _sheet(name: str, orientation: float, layer: str = "m1") -> PalacePort:
    """A source sheet as ``ResolvedCurrentSource.to_palace_port`` makes it."""
    return PalacePort(
        name=name,
        port_type=PortType.SURFACE_CURRENT,
        geometry=PortGeometry.GAP,
        center=(0.0, 0.0),
        width=2.0,
        orientation=orientation,
        zmin=0.0,
        zmax=1.0,
        layer=layer,
    )


def _config(tmp_path, ports=None, **kwargs) -> dict:
    """Write the magnetostatic config of the layout and read it back."""
    ports = ports if ports is not None else [_sheet("A", 180.0), _sheet("B", 90.0)]
    kwargs.setdefault("hints", {"_metallic_boundaries": True})
    kwargs.setdefault("absorbing_boundary", False)
    path = generate_palace_config(
        groups=kwargs.pop("groups", _groups()),
        ports=ports,
        port_info=[],
        stack=stack(),
        output_path=tmp_path,
        model_name="palace",
        fmax=1e9,
        simulation_type="magnetostatic",
        numerical_config=MagnetostaticSolverConfig(),
        **kwargs,
    )
    return json.loads(path.read_text())


def test_the_problem_is_magnetostatic_with_ams_cg(tmp_path) -> None:
    config = _config(tmp_path)
    assert config["Problem"]["Type"] == "Magnetostatic"
    assert config["Solver"]["Magnetostatic"] == {"Save": 0}
    assert config["Solver"]["Linear"]["Type"] == "AMS"
    assert config["Solver"]["Linear"]["KSPType"] == "CG"


def test_problem_settings_are_written(tmp_path) -> None:
    config = _config(tmp_path, magnetostatic_config=MagnetostaticConfig(save_fields=2))
    assert config["Solver"]["Magnetostatic"] == {"Save": 2}


def test_without_a_solver_config_the_magnetostatic_defaults_apply(tmp_path) -> None:
    path = generate_palace_config(
        groups=_groups(),
        ports=[_sheet("A", 180.0)],
        port_info=[],
        stack=stack(),
        output_path=tmp_path,
        model_name="palace",
        fmax=1e9,
        simulation_type="magnetostatic",
        absorbing_boundary=False,
    )
    linear = json.loads(path.read_text())["Solver"]["Linear"]
    assert (linear["Type"], linear["KSPType"], linear["Tol"]) == ("AMS", "CG", 1e-8)


def test_pec_holds_conductors_pec_blocks_via_sides_and_the_outer_wall(
    tmp_path,
) -> None:
    pec = _config(tmp_path)["Boundaries"]["PEC"]["Attributes"]
    assert pec == [10, 11, 12, 13, 14, 15, 16, *OUTER]


def test_without_metallic_boundaries_the_outer_wall_is_left_pmc(tmp_path) -> None:
    pec = _config(tmp_path, hints={})["Boundaries"]["PEC"]["Attributes"]
    assert pec == [10, 11, 12, 13, 14, 15, 16]
    assert not set(OUTER) & set(pec)


def test_surface_currents_follow_the_source_order_and_direction(tmp_path) -> None:
    sources = _config(tmp_path)["Boundaries"]["SurfaceCurrent"]
    assert sources == [
        {"Index": 1, "Attributes": [20], "Direction": "-X"},
        {"Index": 2, "Attributes": [21], "Direction": "+Y"},
    ]


@pytest.mark.parametrize(
    ("orientation", "direction"),
    [(0.0, "+X"), (90.0, "+Y"), (180.0, "-X"), (270.0, "-Y")],
)
def test_the_direction_is_the_signed_cardinal_keyword(
    tmp_path, orientation, direction
) -> None:
    config = _config(tmp_path, ports=[_sheet("A", orientation)])
    assert config["Boundaries"]["SurfaceCurrent"][0]["Direction"] == direction


def test_only_pec_and_surface_currents_are_written(tmp_path) -> None:
    boundaries = _config(tmp_path)["Boundaries"]
    assert set(boundaries) == {"PEC", "SurfaceCurrent"}
    for entry in boundaries["SurfaceCurrent"]:
        assert set(entry) == {"Index", "Attributes", "Direction"}


def test_no_source_sheet_is_a_pec_attribute(tmp_path) -> None:
    boundaries = _config(tmp_path)["Boundaries"]
    sheets = {a for s in boundaries["SurfaceCurrent"] for a in s["Attributes"]}
    assert not sheets & set(boundaries["PEC"]["Attributes"])


def test_a_source_sheet_in_the_pec_set_is_refused(tmp_path) -> None:
    groups = _groups()
    groups["pec_surfaces"]["sheet"] = {"phys_group": 20, "tags": []}
    with pytest.raises(ValueError, match=r"source .*'A'.*PEC"):
        _config(tmp_path, groups=groups)


def test_a_port_that_is_not_a_surface_current_is_refused(tmp_path) -> None:
    lumped = PalacePort(name="o1", geometry=PortGeometry.GAP, width=2.0, zmax=1.0)
    with pytest.raises(ValueError, match=r"'o1'.*not a current source"):
        _config(tmp_path, ports=[lumped])


def test_a_source_without_a_mesh_sheet_is_refused(tmp_path) -> None:
    groups = _groups()
    del groups["port_surfaces"]["P2"]
    with pytest.raises(ValueError, match=r"'B'.*no sheet in the mesh"):
        _config(tmp_path, groups=groups)


def test_an_absorbing_wall_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"Absorbing.*magnetostatic"):
        _config(tmp_path, hints={}, absorbing_boundary=True)
