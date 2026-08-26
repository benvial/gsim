"""Shared fixtures for the gsim.tcad tests.

The hermetic tests never import the real DEVSIM: a recording fake is
injected into ``sys.modules`` so device/mesh/doping setup can be asserted
without the solver runtime, mirroring how palace-binary tests avoid the
Palace binary.
"""

from __future__ import annotations

import sys
import types

import gdsfactory as gf
import pytest

from gsim.common.cross_section import build_doped_cross_section
from gsim.common.stack.doping import make_pn_junction_profile
from gsim.common.stack.junction import PNJunctionConfig


class FakeDevsim(types.ModuleType):
    """Recording stand-in for the ``devsim`` module.

    Records every call as ``(name, kwargs)`` and returns canned values:
    node coordinates come from ``node_coords`` (cm), and the contact charge
    is ``charge_per_volt * bias`` of the contact's bias parameter, so the
    small-signal capacitance of the fake device is exactly
    ``charge_per_volt``.
    """

    def __init__(self):
        super().__init__("devsim")
        self.calls: list[tuple[str, dict]] = []
        self.node_coords = {
            "x": [0.0, 1e-4, 2e-4],
            "y": [0.0, 0.5e-4, 1e-4],
        }
        self.node_values: dict[tuple[str, str], list[float]] = {}
        self.parameters: dict[str, float] = {}
        self.charge_per_volt = 2.5e-12

    def _record(self, _call_name, **kwargs):
        self.calls.append((_call_name, kwargs))

    def called(self, call_name):
        return [kwargs for cname, kwargs in self.calls if cname == call_name]

    # -- mesh / device -------------------------------------------------
    def create_gmsh_mesh(self, **kwargs):
        self._record("create_gmsh_mesh", **kwargs)

    def add_gmsh_region(self, **kwargs):
        self._record("add_gmsh_region", **kwargs)

    def add_gmsh_contact(self, **kwargs):
        self._record("add_gmsh_contact", **kwargs)

    def finalize_mesh(self, **kwargs):
        self._record("finalize_mesh", **kwargs)

    def create_device(self, **kwargs):
        self._record("create_device", **kwargs)

    # -- models / solutions --------------------------------------------
    def node_solution(self, **kwargs):
        self._record("node_solution", **kwargs)

    def set_node_values(self, **kwargs):
        self._record("set_node_values", **kwargs)
        if "values" in kwargs:
            key = (kwargs["region"], kwargs["name"])
            self.node_values[key] = list(kwargs["values"])

    def node_model(self, **kwargs):
        self._record("node_model", **kwargs)

    def set_parameter(self, **kwargs):
        self._record("set_parameter", **kwargs)
        self.parameters[kwargs["name"]] = kwargs["value"]

    def solve(self, **kwargs):
        self._record("solve", **kwargs)

    # -- readback ------------------------------------------------------
    def get_node_model_values(self, **kwargs):
        self._record("get_node_model_values", **kwargs)
        name = kwargs["name"]
        if name in self.node_coords:
            return list(self.node_coords[name])
        key = (kwargs["region"], name)
        if key in self.node_values:
            return list(self.node_values[key])
        return [0.0] * len(self.node_coords["x"])

    def get_contact_current(self, **kwargs):
        self._record("get_contact_current", **kwargs)
        return 0.0

    def get_contact_charge(self, **kwargs):
        self._record("get_contact_charge", **kwargs)
        bias = self.parameters.get(f"{kwargs['contact']}_bias", 0.0)
        return self.charge_per_volt * bias


class FakeSimplePhysics(types.ModuleType):
    """Recording stand-in for ``devsim.python_packages.simple_physics``."""

    def __init__(self):
        super().__init__("devsim.python_packages.simple_physics")
        self.calls: list[tuple[str, tuple]] = []

    def called(self, name):
        return [args for cname, args in self.calls if cname == name]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "GetContactBiasName":
            return lambda contact: f"{contact}_bias"

        def _record(*args):
            self.calls.append((name, args))

        return _record


@pytest.fixture
def fake_devsim(monkeypatch):
    """Install the fake devsim modules and return (devsim, simple_physics)."""
    devsim = FakeDevsim()
    packages = types.ModuleType("devsim.python_packages")
    sp = FakeSimplePhysics()
    monkeypatch.setitem(sys.modules, "devsim", devsim)
    monkeypatch.setitem(sys.modules, "devsim.python_packages", packages)
    monkeypatch.setitem(sys.modules, "devsim.python_packages.simple_physics", sp)
    return devsim, sp


def build_pn_device():
    """Rib with adjacent P/N doped regions on a doped cross-section stack."""
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
