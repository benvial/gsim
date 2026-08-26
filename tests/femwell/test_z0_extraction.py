"""Marks-Williams power-current Z_0 extraction from femwell RF modes.

Validated against the analytic impedance of a dielectric-filled coaxial
line whose inner conductor is a finite-conductivity copper disk: the
power-current definition ``Z_0 = 2P / |I|^2`` must land on
``(eta_0 / (2 pi sqrt(eps_r))) ln(b/a)`` to within the good-conductor
corrections, and it must be invariant to the arbitrary normalization of
the mode fields.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import epsilon_0 as EPS0  # noqa: N812
from scipy.constants import mu_0 as MU0  # noqa: N812
from scipy.constants import speed_of_light as C0  # noqa: N812

from gsim.femwell.adapter import solve_modes, z0_power_current

pytest.importorskip("femwell")
pytest.importorskip("skfem")

# Coax geometry (um) and analysis frequency.
R_INNER = 1.0
R_OUTER = 3.0
EPS_DIELECTRIC = 2.25
SIGMA_COPPER = 5.8e7
FREQ_HZ = 100e9
ETA0 = MU0 * C0
Z0_ANALYTIC = ETA0 / (2 * np.pi * np.sqrt(EPS_DIELECTRIC)) * np.log(R_OUTER / R_INNER)


@pytest.fixture(scope="module")
def coax_mode(tmp_path_factory):
    import gmsh

    path = tmp_path_factory.mktemp("femwell-z0") / "coax.msh"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        outer = occ.addDisk(0.0, 0.0, 0.0, R_OUTER, R_OUTER)
        inner = occ.addDisk(0.0, 0.0, 0.0, R_INNER, R_INNER)
        occ.fragment([(2, outer)], [(2, inner)])
        occ.synchronize()
        surfaces = gmsh.model.getEntities(2)
        by_area = sorted(
            (gmsh.model.occ.getMass(2, tag), tag) for _dim, tag in surfaces
        )
        pg_core = gmsh.model.addPhysicalGroup(2, [by_area[0][1]])
        gmsh.model.setPhysicalName(2, pg_core, "conductor")
        pg_diel = gmsh.model.addPhysicalGroup(2, [tag for _a, tag in by_area[1:]])
        gmsh.model.setPhysicalName(2, pg_diel, "dielectric")
        # Resolve the ~0.2 um skin depth at the conductor surface.
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.35)
        field = gmsh.model.mesh.field
        ball = field.add("Ball")
        field.setNumber(ball, "Radius", R_INNER + 0.15)
        field.setNumber(ball, "VIn", 0.07)
        field.setNumber(ball, "VOut", 0.35)
        field.setAsBackgroundMesh(ball)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()

    omega = 2 * np.pi * FREQ_HZ
    eps_conductor = 1.0 - 1j * SIGMA_COPPER / (omega * EPS0)
    modes = solve_modes(
        path,
        epsilon={
            "conductor": eps_conductor,
            "dielectric": EPS_DIELECTRIC + 0j,
        },
        wavelength_um=C0 / FREQ_HZ * 1e6,
        num_modes=1,
        metallic_boundaries=True,
    )
    return modes[0]


class TestZ0PowerCurrent:
    def test_coax_matches_analytic(self, coax_mode):
        z0 = z0_power_current(coax_mode, frequency_hz=FREQ_HZ)
        # Measured 3.2% high with a small negative reactance — the expected
        # finite-conductivity correction at 100 GHz; the margin covers mesh
        # variance across gmsh versions.
        assert abs(z0.real - Z0_ANALYTIC) < 0.08 * Z0_ANALYTIC
        assert abs(z0.imag) < 0.1 * Z0_ANALYTIC

    def test_normalization_invariant(self, coax_mode):
        from dataclasses import replace

        scaled = replace(coax_mode, E=coax_mode.E * 3.7, H=coax_mode.H * 3.7)
        z0 = z0_power_current(coax_mode, frequency_hz=FREQ_HZ)
        z0_scaled = z0_power_current(scaled, frequency_hz=FREQ_HZ)
        assert z0_scaled == pytest.approx(z0, rel=1e-12)

    def test_no_conductive_elements_raises(self, coax_mode):
        with pytest.raises(ValueError, match="conductive"):
            z0_power_current(
                coax_mode,
                frequency_hz=FREQ_HZ,
                sigma_s_per_m=np.zeros_like(np.asarray(coax_mode.epsilon_r).real),
            )

    def test_explicit_sigma_matches_default(self, coax_mode):
        omega = 2 * np.pi * FREQ_HZ
        eps = np.asarray(coax_mode.epsilon_r)
        sigma = np.where(eps.imag < 0, -eps.imag, 0.0) * omega * EPS0
        z0_default = z0_power_current(coax_mode, frequency_hz=FREQ_HZ)
        z0_explicit = z0_power_current(
            coax_mode, frequency_hz=FREQ_HZ, sigma_s_per_m=sigma
        )
        assert z0_explicit == pytest.approx(z0_default, rel=1e-12)
