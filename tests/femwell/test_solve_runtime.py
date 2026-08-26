"""femwell solve tests, gated on the femwell/skfem runtime being installed."""

from __future__ import annotations

import meshio
import numpy as np
import pytest

from gsim.femwell.adapter import solve_modes

pytest.importorskip("femwell")
pytest.importorskip("skfem")


@pytest.fixture(scope="module")
def strip_mesh(tmp_path_factory):
    """Structured triangle mesh of a strip waveguide cross-section (um)."""
    import gmsh

    path = tmp_path_factory.mktemp("femwell-runtime") / "strip.msh"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        clad = occ.addRectangle(-1.5, -1.0, 0.0, 3.0, 2.22)
        core = occ.addRectangle(-0.25, 0.0, 0.0, 0.5, 0.22)
        occ.fragment([(2, clad)], [(2, core)])
        occ.synchronize()
        surfaces = gmsh.model.getEntities(2)
        core_tags = []
        clad_tags = []
        for _dim, tag in surfaces:
            x, y, _z = gmsh.model.occ.getCenterOfMass(2, tag)
            if abs(x) < 0.25 and 0.0 < y < 0.22:
                core_tags.append(tag)
            else:
                clad_tags.append(tag)
        pg_core = gmsh.model.addPhysicalGroup(2, core_tags)
        gmsh.model.setPhysicalName(2, pg_core, "core")
        pg_clad = gmsh.model.addPhysicalGroup(2, clad_tags)
        gmsh.model.setPhysicalName(2, pg_clad, "clad")
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.08)
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


class TestPiecewiseConstantSolve:
    def test_strip_waveguide_fundamental_mode(self, strip_mesh):
        modes = solve_modes(
            strip_mesh,
            epsilon={"core": 3.48**2 + 0j, "clad": 1.444**2 + 0j},
            wavelength_um=1.55,
            num_modes=1,
        )
        n_eff = np.real(modes[0].n_eff)
        # Fundamental mode of a 500x220 nm silicon strip is well guided.
        assert 1.444 < n_eff < 3.48
        assert n_eff > 2.0

    def test_unknown_region_raises(self, strip_mesh):
        with pytest.raises(ValueError, match="no_such_region"):
            solve_modes(
                strip_mesh,
                epsilon={"no_such_region": 12.0 + 0j},
                wavelength_um=1.55,
            )


class TestContinuousSolve:
    def test_elementwise_epsilon_matches_piecewise(self, strip_mesh):
        from gsim.femwell.adapter import elementwise_epsilon

        mesh = meshio.read(str(strip_mesh))
        tris = np.vstack([b.data for b in mesh.cells if b.type == "triangle"])
        centroids = mesh.points[tris][:, :, :2].mean(axis=1)
        # Build continuous samples reproducing the piecewise structure.
        xs = centroids[:, 0]
        ys = centroids[:, 1]
        eps_samples = np.where(
            (np.abs(xs) < 0.25) & (ys > 0.0) & (ys < 0.22),
            3.48**2,
            1.444**2,
        ).astype(complex)
        eps_elements = elementwise_epsilon(strip_mesh, xs, ys, eps_samples)

        modes_pc = solve_modes(
            strip_mesh,
            epsilon={"core": 3.48**2 + 0j, "clad": 1.444**2 + 0j},
            wavelength_um=1.55,
        )
        modes_cont = solve_modes(strip_mesh, epsilon=eps_elements, wavelength_um=1.55)
        assert np.real(modes_cont[0].n_eff) == pytest.approx(
            np.real(modes_pc[0].n_eff), rel=5e-3
        )
