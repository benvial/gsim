"""Palace-specific material resolution with frequency-dependent dispersion.

Evaluates dispersion models from the materials database at a target frequency
and produces a materials dict suitable for Palace config generation.

This implements the RFC's external frequency loop strategy: since Palace
does not natively support epsilon(f), gsim evaluates each material's
dispersion model at the target frequency and writes scalar properties
to the Palace config JSON.
"""

from __future__ import annotations

from pydantic import ValidationError
from scipy.constants import c as C0  # noqa: N812

from gsim.common.stack.materials import (
    MaterialProperties,
    get_material_properties,
    resolve_material_at_wavelength,
)


def resolve_palace_materials_at_frequency(
    materials: dict[str, dict],
    frequency_hz: float,
) -> dict[str, dict]:
    """Evaluate material dispersion at a given frequency for Palace config.

    For each material in the dict, looks up the corresponding entry in
    MATERIALS_DB and evaluates its dispersion model at the target frequency.
    The resolved scalar values (permittivity, loss_tangent) replace the
    original values in the dict.

    Materials without dispersion models or without matching database entries
    are left unchanged.

    Args:
        materials: Dict of material name -> properties dict (from LayerStack)
        frequency_hz: Target frequency in Hz

    Returns:
        New materials dict with evaluated scalar properties
    """
    wavelength_um = C0 / frequency_hz * 1e6
    resolved: dict[str, dict] = {}

    for name, props in materials.items():
        # The stack entry is authoritative: it either carries the full
        # database record (dispersion models included) or the plain scalars
        # a user supplied through set_material(). Evaluating the entry
        # itself — the same rule the femwell adapter applies — keeps user
        # overrides intact; the database is only a fallback for entries
        # that do not form a valid MaterialProperties record.
        overrides: dict[str, MaterialProperties] | None
        try:
            overrides = {name: MaterialProperties.model_validate(props)}
        except ValidationError:
            overrides = None
            if get_material_properties(name) is None:
                resolved[name] = dict(props)
                continue

        evaluated = resolve_material_at_wavelength(
            name, wavelength_um, overrides=overrides
        )
        if evaluated is None or evaluated.behavior == "conductive":
            resolved[name] = dict(props)
            continue

        new_props: dict[str, object] = dict(props)

        if evaluated.permittivity is not None:
            new_props["permittivity"] = evaluated.permittivity

        if evaluated.loss_tangent is not None:
            new_props["loss_tangent"] = evaluated.loss_tangent

        if evaluated.conductivity is not None:
            new_props["conductivity"] = evaluated.conductivity
        else:
            new_props.pop("conductivity", None)

        if evaluated.permeability is not None:
            new_props["permeability"] = evaluated.permeability

        resolved[name] = new_props

    return resolved
