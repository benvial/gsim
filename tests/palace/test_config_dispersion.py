"""Tests for Palace material resolution with frequency-dependent dispersion."""

from __future__ import annotations

import pytest

from gsim.common.stack.materials import MATERIALS_DB
from gsim.palace.materials import resolve_palace_materials_at_frequency


class TestResolvePalaceMaterialsAtFrequency:
    def test_sio2_at_optical_frequency(self):
        materials = {"SiO2": MATERIALS_DB["SiO2"].to_dict()}
        freq_hz = 3e8 / (1.55e-6)
        resolved = resolve_palace_materials_at_frequency(materials, freq_hz)
        assert "SiO2" in resolved
        assert resolved["SiO2"]["permittivity"] == pytest.approx(2.085, abs=0.01)

    def test_sio2_at_rf_frequency(self):
        materials = {"SiO2": MATERIALS_DB["SiO2"].to_dict()}
        freq_hz = 5e9
        resolved = resolve_palace_materials_at_frequency(materials, freq_hz)
        assert "SiO2" in resolved
        assert resolved["SiO2"]["permittivity"] == pytest.approx(4.1)

    def test_silicon_at_optical_frequency(self):
        materials = {"silicon": MATERIALS_DB["silicon"].to_dict()}
        freq_hz = 3e8 / (1.55e-6)
        resolved = resolve_palace_materials_at_frequency(materials, freq_hz)
        assert "silicon" in resolved
        n_sq = resolved["silicon"]["permittivity"]
        n = n_sq**0.5
        assert 3.5 < n < 4.2

    def test_conductor_unchanged(self):
        materials = {"aluminum": MATERIALS_DB["aluminum"].to_dict()}
        resolved = resolve_palace_materials_at_frequency(materials, 5e9)
        assert "aluminum" in resolved
        assert resolved["aluminum"]["type"] == "conductor"

    def test_unknown_material_preserved(self):
        materials = {"custom_mat": {"permittivity": 5.0, "type": "dielectric"}}
        resolved = resolve_palace_materials_at_frequency(materials, 5e9)
        assert "custom_mat" in resolved
        assert resolved["custom_mat"]["permittivity"] == 5.0

    def test_empty_materials(self):
        resolved = resolve_palace_materials_at_frequency({}, 5e9)
        assert resolved == {}

    def test_preserves_nonoptical_fields(self):
        materials = {"SiO2": MATERIALS_DB["SiO2"].to_dict()}
        freq_hz = 3e8 / (1.55e-6)
        resolved = resolve_palace_materials_at_frequency(materials, freq_hz)
        assert resolved["SiO2"]["type"] == "dielectric"

    def test_does_not_mutate_input(self):
        materials = {"SiO2": MATERIALS_DB["SiO2"].to_dict()}
        original_permittivity = materials["SiO2"]["permittivity"]
        resolve_palace_materials_at_frequency(materials, 5e9)
        assert materials["SiO2"]["permittivity"] == original_permittivity

    def test_sapphire_anisotropic_resolved(self):
        materials = {"sapphire": MATERIALS_DB["sapphire"].to_dict()}
        freq_hz = 3e8 / (1.55e-6)
        resolved = resolve_palace_materials_at_frequency(materials, freq_hz)
        assert "sapphire" in resolved
        assert isinstance(resolved["sapphire"]["permittivity"], list)
