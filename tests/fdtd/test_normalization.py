"""Tests for source-aware FDTD result normalization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gsim.fdtd import (
    FDTDResult,
    GaussianBeamSource,
    fiber_coupling_efficiency,
    gaussian_coupling_efficiency,
)
from gsim.fdtd.results import ComplexTrace, PlaneMonitorResult


def _gaussian_source() -> GaussianBeamSource:
    return GaussianBeamSource(
        center_um=(24.4041, 0, 1.05),
        size_um=(12.48, 12.48, 0),
        aperture_normal="-z",
        propagation_direction=(-0.17388, 0, -0.98477),
        e_polarization=(0, 1, 0),
        focal_point_um=(24.4041, 0, 1.05),
        waist_radius_um=5.2,
        refractive_index=1.44,
        wavelength_um=1.55,
        wavelength_span_um=0.1,
        num_wavelengths=101,
    )


def _write_result(
    path: Path, modal_power: list[float], *, include_modal_power: bool = True
) -> FDTDResult:
    wavelengths_nm = np.linspace(1500, 1600, len(modal_power))
    samples = []
    for power in modal_power:
        sample: dict[str, float] = {"re": 0, "im": 0, "power_fraction": 1}
        if include_modal_power:
            sample["modal_power"] = power
        samples.append(sample)
    document = {
        "schema_version": 1,
        "excitation_type": "gaussian_beam",
        "ports": ["o1"],
        "frequencies": {
            "wavelength_nm": wavelengths_nm.tolist(),
            "hz": (299792458 / (wavelengths_nm * 1e-9)).tolist(),
            "below_noise_floor": [False] * len(modal_power),
        },
        "port_outputs": {"o1": samples},
    }
    result_path = path / "sparams_o1.json"
    result_path.write_text(json.dumps(document), encoding="utf8")
    return FDTDResult.from_file(result_path)


def test_gaussian_coupling_masks_spectral_tails_and_normalizes_power(
    tmp_path: Path,
) -> None:
    result = _write_result(tmp_path, [1.0] * 101)

    coupling = gaussian_coupling_efficiency(result, _gaussian_source(), port="o1")

    assert np.count_nonzero(coupling.valid) == 38
    assert coupling.wavelength_um[coupling.valid][[0, -1]].tolist() == pytest.approx(
        [1.532, 1.569]
    )
    assert np.isnan(coupling.efficiency[0])
    assert np.all(coupling.efficiency[coupling.valid] > 0)
    figure = coupling.plot_plotly()
    assert figure.layout.yaxis.title.text == "Coupling efficiency (dB)"
    assert tuple(figure.layout.xaxis.range) == pytest.approx((1.532, 1.569))


def test_gaussian_coupling_rejects_missing_modal_power(tmp_path: Path) -> None:
    result = _write_result(tmp_path, [1.0, 1.0, 1.0], include_modal_power=False)

    with pytest.raises(ValueError, match="modal-power"):
        gaussian_coupling_efficiency(result, _gaussian_source(), port="o1")


def test_fiber_coupling_converts_normalized_amplitude_to_power() -> None:
    monitor = PlaneMonitorResult(
        name="fiber",
        normal_axis="z",
        normal_sign=1,
        u_axis="x",
        v_axis="y",
        plane_position_um=1.05,
        u_extent_um=(18.1641, 30.6441),
        v_extent_um=(-6.24, 6.24),
        shape=(2, 2),
        wavelength_um=np.asarray([1.5, 1.55, 1.6]),
        coupling_efficiency=ComplexTrace(
            values=np.asarray([0.5 + 0j, 0.25j, 0]),
            valid=np.asarray([True, True, False]),
        ),
    )

    coupling = fiber_coupling_efficiency(monitor)

    np.testing.assert_allclose(coupling.efficiency[:2], [0.25, 0.0625])
    assert np.isnan(coupling.efficiency[2])
    assert coupling.label == "fiber"
