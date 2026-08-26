"""femwell/skfem runtime resolution for the gsim.femwell adapter.

femwell and scikit-fem are optional dependencies installed through the
``femwell`` packaging extra. Importing :mod:`gsim.femwell` never requires
them; only the solve path calls :func:`require_femwell`.
"""

from __future__ import annotations

import importlib
from types import ModuleType

_INSTALL_HINT = (
    "femwell/scikit-fem are required for the femwell mode-solving route "
    "but are not installed. Install the optional extra: "
    "pip install 'gsim[femwell]' (or: pip install femwell)."
)


def require_femwell() -> ModuleType:
    """Import and return the ``femwell`` module.

    Raises:
        ImportError: When femwell is not installed, with a message naming
            the ``femwell`` packaging extra.
    """
    try:
        return importlib.import_module("femwell")
    except ImportError as err:
        raise ImportError(_INSTALL_HINT) from err


def require_skfem() -> ModuleType:
    """Import and return the ``skfem`` module.

    Raises:
        ImportError: When scikit-fem is not installed, with a message
            naming the ``femwell`` packaging extra.
    """
    try:
        return importlib.import_module("skfem")
    except ImportError as err:
        raise ImportError(_INSTALL_HINT) from err


__all__ = ["require_femwell", "require_skfem"]
