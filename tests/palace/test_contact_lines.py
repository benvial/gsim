"""Tests for named contact line groups in the native-2D BoundaryMode mesh.

Contacts are declared as layer pairs; the shared interface curves between
the two layers' meshed regions become a dim-1 physical group carrying the
contact name, so DEVSIM's ``add_gmsh_contact`` can bind to it.
"""

from __future__ import annotations

import gdsfactory as gf
import meshio
import numpy as np
import pytest
from pydantic import ValidationError

from gsim.common.cross_section import build_doped_cross_section
from gsim.common.stack.doping import make_pn_junction_profile
from gsim.common.stack.junction import PNJunctionConfig
from gsim.palace import BoundaryModeSim
from gsim.palace.models import ContactSpec


def _build_pn_device():
    """Rib with adjacent P/N doped regions (capacitance mode: touching)."""
    gf.gpdk.PDK.activate()
    comp = gf.Component()
    wg = comp << gf.c.rectangle((10.0, 0.4), centered=True, layer=(1, 0))
    wg.y = -20.0
    slab = comp << gf.c.rectangle((10.0, 100.0), centered=True, layer=(3, 0))
    slab.y = -5.0
    pn = make_pn_junction_profile(
        comp,
        length=10.0,
        center_y=-20.0,
        rib_width=0.4,
        junction=PNJunctionConfig(na_cm3=1e19, nd_cm3=1e19),
        p_region=("p_rib", (21, 0), 1.6e3),
        n_region=("n_rib", (20, 0), 1.6e3),
        junction_region=("junction", (22, 0)),
        zmin=0.0,
        zmax=0.22,
        mode="capacitance",
    )
    stack, _section = build_doped_cross_section(
        comp,
        axis="x",
        value=0.0,
        substrate_thickness=2.0,
        doping=pn,
        verbose=False,
    )
    return comp, stack


def _make_sim(tmp_path, contacts):
    comp, stack = _build_pn_device()
    sim = BoundaryModeSim()
    sim.set_output_dir(str(tmp_path))
    sim.set_stack(stack)
    sim.set_airbox(margin_x=3.0, margin_y=3.0, z_above=2.0, z_below=2.0)
    sim.set_geometry(comp)
    sim.set_cross_section("x=0")
    sim.set_boundary_mode(freq=50e9, num_modes=1)
    for contact in contacts:
        sim.add_contact(**contact)
    sim.mesh(preset="coarse", refined_mesh_size=0.05, max_mesh_size=40.0, verbose=False)
    return sim


class TestContactSpecModel:
    def test_fields(self):
        spec = ContactSpec(name="anode", layer_a="metal1", layer_b="p_rib")
        assert spec.name == "anode"

    def test_rejects_same_layer(self):
        with pytest.raises(ValidationError):
            ContactSpec(name="bad", layer_a="p_rib", layer_b="p_rib")

    def test_rejects_empty_name(self):
        with pytest.raises(ValidationError):
            ContactSpec(name="", layer_a="a", layer_b="b")


class TestContactLineGroups:
    def test_contact_group_in_mesh(self, tmp_path):
        sim = _make_sim(
            tmp_path,
            [{"name": "anode", "layer_a": "p_rib", "layer_b": "n_rib"}],
        )
        contact_lines = sim._last_mesh_result.groups["contact_lines"]
        assert "anode" in contact_lines
        assert contact_lines["anode"]["tags"]

        mesh = meshio.read(sim._last_mesh_result.mesh_path)
        field_data = mesh.field_data
        assert "anode" in field_data
        dim = int(np.asarray(field_data["anode"])[1])
        assert dim == 1
        # Lines with that physical tag actually exist in the mesh.
        tag = int(np.asarray(field_data["anode"])[0])
        line_tags = np.concatenate(
            [
                arr
                for cell_block, arr in zip(
                    mesh.cells, mesh.cell_data["gmsh:physical"], strict=True
                )
                if cell_block.type == "line"
            ]
        )
        assert (line_tags == tag).sum() > 0

    def test_multiple_contacts(self, tmp_path):
        sim = _make_sim(
            tmp_path,
            [
                {"name": "anode", "layer_a": "p_rib", "layer_b": "sio2"},
                {"name": "cathode", "layer_a": "n_rib", "layer_b": "sio2"},
            ],
        )
        contact_lines = sim._last_mesh_result.groups["contact_lines"]
        assert {"anode", "cathode"} <= set(contact_lines)

    def test_nontouching_pair_raises(self, tmp_path):
        with pytest.raises(ValueError, match="anode"):
            _make_sim(
                tmp_path,
                [{"name": "anode", "layer_a": "p_rib", "layer_b": "substrate"}],
            )

    def test_unknown_layer_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no_such_layer"):
            _make_sim(
                tmp_path,
                [{"name": "anode", "layer_a": "p_rib", "layer_b": "no_such_layer"}],
            )
