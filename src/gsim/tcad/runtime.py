"""DEVSIM runtime resolution for the gsim.tcad backend.

DEVSIM is an optional dependency installed through the ``tcad`` packaging
extra. Importing :mod:`gsim.tcad` never requires it; only the methods that
actually talk to the solver call :func:`require_devsim`.
"""

from __future__ import annotations

import importlib
from types import ModuleType

_INSTALL_HINT = (
    "DEVSIM is required for the charge-transport solve but is not "
    "installed. Install the optional extra: pip install 'gsim[tcad]' "
    "(or: pip install devsim)."
)


def require_devsim() -> ModuleType:
    """Import and return the ``devsim`` module.

    Raises:
        ImportError: When DEVSIM is not installed, with a message naming
            the ``tcad`` packaging extra.
    """
    try:
        return importlib.import_module("devsim")
    except ImportError as err:
        raise ImportError(_INSTALL_HINT) from err


def import_simple_physics() -> ModuleType:
    """Import DEVSIM's prebuilt Scharfetter-Gummel physics package.

    Raises:
        ImportError: When DEVSIM is not installed, with a message naming
            the ``tcad`` packaging extra.
    """
    try:
        return importlib.import_module("devsim.python_packages.simple_physics")
    except ImportError as err:
        raise ImportError(_INSTALL_HINT) from err


__all__ = ["import_simple_physics", "require_devsim"]
