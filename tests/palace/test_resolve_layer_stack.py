"""Tests for the pure layer-stack resolver used by mesh reuse."""

from __future__ import annotations

import pytest

from gsim.common.stack import LayerStack
from gsim.palace import DrivenSim
from gsim.palace.base import resolve_layer_stack
from gsim.palace.models import MaterialConfig


def _prebuilt_stack() -> LayerStack:
    return LayerStack(
        dielectrics=[{"name": "oxide", "zmin": -2.0, "zmax": 5.2, "material": "sio2"}],
        materials={
            "sio2": {"type": "dielectric", "permittivity": 3.9},
            "aluminum": {"type": "conductor", "conductivity": 3.77e7},
        },
    )


@pytest.fixture
def overrides() -> dict[str, MaterialConfig]:
    return {
        "sio2": MaterialConfig.dielectric(permittivity=4.1, loss_tangent=1e-3),
        "gold": MaterialConfig.conductor(conductivity=4.1e7),
    }


@pytest.fixture
def fake_get_stack(monkeypatch):
    """Patch the PDK/YAML stack builder; record the arguments it receives."""
    calls: list[dict] = []

    def _get_stack(yaml_path=None, **kwargs):
        calls.append({"yaml_path": yaml_path, **kwargs})
        return LayerStack(
            dielectrics=[
                {"name": "oxide", "zmin": -2.0, "zmax": 5.2, "material": "sio2"},
                {"name": "air_box", "zmin": 5.2, "zmax": 50.0, "material": "Air"},
            ],
            materials={"sio2": {"type": "dielectric", "permittivity": 3.9}},
        )

    monkeypatch.setattr("gsim.common.stack.get_stack", _get_stack)
    return calls


def test_prebuilt_stack_gets_material_overrides_in_place(overrides):
    """F4: a prebuilt stack comes back as the same object, with overrides written in."""
    stack = _prebuilt_stack()

    resolved = resolve_layer_stack(stack, {"_prebuilt": True}, overrides)

    assert resolved is stack
    assert stack.materials == {
        "sio2": {"type": "dielectric", "permittivity": 4.1, "loss_tangent": 1e-3},
        "aluminum": {"type": "conductor", "conductivity": 3.77e7},
        "gold": {"type": "conductor", "conductivity": 4.1e7},
    }
    assert stack.dielectrics == [
        {"name": "oxide", "zmin": -2.0, "zmax": 5.2, "material": "sio2"}
    ]


def test_sim_writes_overrides_into_prebuilt_stack():
    """F4: ``_resolve_stack`` keeps the prebuilt stack and writes overrides in."""
    stack = _prebuilt_stack()
    sim = DrivenSim()
    sim.set_stack(stack)
    sim.set_material("gold", material_type="conductor", conductivity=4.1e7)

    resolved = sim._resolve_stack()

    assert resolved is stack
    assert sim.stack is stack
    assert sim._stack_kwargs == {"_prebuilt": True}
    assert stack.materials["gold"] == {"type": "conductor", "conductivity": 4.1e7}
    assert stack.materials["sio2"] == {"type": "dielectric", "permittivity": 3.9}


def test_pdk_stack_drops_air_and_leaves_kwargs_untouched(fake_get_stack, overrides):
    """The resolver builds from kwargs without mutating them and drops air regions."""
    stack_kwargs = {
        "yaml_path": "custom.yaml",
        "include_substrate": False,
        "add_oxide_dielectric": True,
    }

    resolved = resolve_layer_stack(None, stack_kwargs, overrides)

    assert stack_kwargs == {
        "yaml_path": "custom.yaml",
        "include_substrate": False,
        "add_oxide_dielectric": True,
    }
    assert fake_get_stack == [
        {
            "yaml_path": "custom.yaml",
            "include_substrate": False,
            "add_oxide_dielectric": True,
        }
    ]
    assert [d["name"] for d in resolved.dielectrics] == ["oxide"]
    assert resolved.materials["sio2"] == {
        "type": "dielectric",
        "permittivity": 4.1,
        "loss_tangent": 1e-3,
    }
    assert resolved.materials["gold"] == {"type": "conductor", "conductivity": 4.1e7}


def test_pdk_stack_without_yaml_path_leaves_kwargs_untouched(fake_get_stack):
    """Kwargs without ``yaml_path`` gain no key and reach the builder as None."""
    stack_kwargs: dict = {}

    resolve_layer_stack(None, stack_kwargs, {})

    assert stack_kwargs == {}
    assert fake_get_stack == [{"yaml_path": None}]


def test_sim_assigns_resolved_pdk_stack(fake_get_stack):
    """``_resolve_stack`` stores the PDK/YAML stack on the sim and returns it."""
    sim = DrivenSim()
    sim.set_stack(yaml_path="custom.yaml")

    resolved = sim._resolve_stack()

    assert sim.stack is resolved
    assert [d["name"] for d in resolved.dielectrics] == ["oxide"]
    assert fake_get_stack[0]["yaml_path"] == "custom.yaml"
