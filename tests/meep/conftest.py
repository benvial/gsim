"""Activate the generic PDK for tests in gsim.meep that build components."""

from __future__ import annotations

import gdsfactory as gf
import pytest


@pytest.fixture(autouse=True)
def activate_generic_pdk():
    """Activate the generic PDK before each test to prevent IHP PDK bleed-through."""
    gf.gpdk.PDK.activate()
