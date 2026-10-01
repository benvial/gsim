"""Import-time degradation: gsim.tcad works without DEVSIM installed."""

from __future__ import annotations

import builtins
import sys

import pytest


def test_package_imports_without_devsim():
    import gsim.tcad  # noqa: F401


def test_require_devsim_names_the_extra(monkeypatch):
    from gsim.tcad.runtime import require_devsim

    # Block the import even when devsim happens to be installed.
    monkeypatch.setitem(sys.modules, "devsim", None)
    with pytest.raises(ImportError, match=r"gsim\[tcad\]"):
        require_devsim()


def test_import_simple_physics_names_the_extra(monkeypatch):
    from gsim.tcad.runtime import import_simple_physics

    monkeypatch.setitem(sys.modules, "devsim", None)
    monkeypatch.setitem(sys.modules, "devsim.python_packages", None)
    monkeypatch.setitem(sys.modules, "devsim.python_packages.simple_physics", None)
    with pytest.raises(ImportError, match=r"gsim\[tcad\]"):
        import_simple_physics()


def test_tcad_extra_declared_in_packaging():
    from importlib import metadata

    try:
        requires = metadata.requires("gsim") or []
    except metadata.PackageNotFoundError:
        pytest.skip("gsim not installed as a distribution")
    extras = {
        req.split("extra == ")[1].strip("\"'") for req in requires if "extra == " in req
    }
    assert "tcad" in extras
    assert any("devsim" in req for req in requires)


def test_reset_device_does_not_reimport_devsim(monkeypatch):
    """A DEVSIM whose import fails must not be imported again to release.

    DEVSIM declares its default derivatives in a one-shot C initialiser.
    An install without the math libraries raises ``RuntimeError`` partway
    through that initialiser and leaves no ``sys.modules`` entry, so a
    second import redeclares what the first declared and raises again —
    out of a release path that only means to clean up.
    """
    from gsim.tcad import ChargeTransportSim

    monkeypatch.delitem(sys.modules, "devsim", raising=False)
    real_import = builtins.__import__

    def _devsim_fails_to_initialise(name, *args, **kwargs):
        if name == "devsim" or name.startswith("devsim."):
            raise RuntimeError("Issues initializing DEVSIM.")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _devsim_fails_to_initialise)

    sim = ChargeTransportSim()
    sim._device = "gsim_tcad_device_0"
    sim.reset_device()
    assert sim._device is None
