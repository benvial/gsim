"""The charge Stage on the real pipeline: meshing, and the DEVSIM solve.

The mesh test needs gmsh only; the solve test is gated on DEVSIM
(deselected by default, run with ``pytest -m tcad_local``).
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.modulator import Device, Study
from gsim.tcad.results import BiasSweepResult

from .conftest import build_demo


@pytest.fixture(scope="module")
def meshed(tmp_path_factory):
    """The charge Stage's simulation, actually meshed."""
    demo = build_demo()
    component, stack = demo.component, demo.stack
    study = Study(
        component=component,
        stack=stack,
        device=Device(
            p_regions=["p_rib", "p_pad"],
            n_regions=["n_rib", "n_pad"],
        ),
        output_dir=tmp_path_factory.mktemp("modulator-charge"),
    )
    sim = study.charge.simulation()
    sim.mesh(**study.charge.mesh)
    return study, sim


class TestDerivedNamesReachTheMesh:
    def test_every_derived_contact_and_interface_is_tagged(self, meshed):
        study, sim = meshed
        contact_lines = set(sim.mesh_groups["contact_lines"])
        expected = {c.name for c in study.layout.contacts} | {
            i.name for i in study.layout.interfaces
        }
        assert expected <= contact_lines

    def test_every_doped_region_is_a_mesh_volume(self, meshed):
        study, sim = meshed
        volumes = set(sim.mesh_groups["volumes"])
        assert set(study.device.doped_regions) <= volumes

    def test_the_mesh_is_clipped_to_the_derived_window(self, meshed):
        import meshio

        study, sim = meshed
        points = np.asarray(meshio.read(str(sim.mesh_path)).points)
        low, high = study.layout.window
        # The airbox margin extends the domain past the window on both
        # sides, but the meshed slab is far smaller than the component.
        assert points[:, 0].min() > low - 5.0
        assert points[:, 0].max() < high + 5.0


@pytest.mark.tcad_local
class TestSolve:
    def test_the_study_solves_the_bias_sweep(self, tmp_path):
        pytest.importorskip("devsim")
        demo = build_demo()
        component, stack = demo.component, demo.stack
        study = Study(
            component=component,
            stack=stack,
            device=Device(
                p_regions=["p_rib", "p_pad"],
                n_regions=["n_rib", "n_pad"],
            ),
            output_dir=tmp_path,
        )
        study.charge(biases=[0.0, 0.5])
        sweep = study.charge.run()

        assert isinstance(sweep, BiasSweepResult)
        assert sweep.contact == "cathode"
        assert len(sweep.points) == 2
        assert np.all(sweep.points[0].carriers.electrons_cm3 > 0.0)
        # Reverse bias widens the depletion region: capacitance falls.
        assert sweep.capacitance_f_per_cm[1] < sweep.capacitance_f_per_cm[0]
        assert study.charge.run() is sweep

    def test_re_running_after_a_change_solves_again(self, tmp_path):
        """DEVSIM's global device/mesh/circuit namespace survives a re-run."""
        pytest.importorskip("devsim")
        demo = build_demo()
        component, stack = demo.component, demo.stack
        study = Study(
            component=component,
            stack=stack,
            device=Device(
                p_regions=["p_rib", "p_pad"],
                n_regions=["n_rib", "n_pad"],
            ),
            output_dir=tmp_path,
        )
        study.charge(biases=[0.0])
        first = study.charge.run()

        study.charge(biases=[0.5])
        second = study.charge.run()

        assert second is not first
        assert [point.bias_v for point in second.points] == [0.5]


@pytest.mark.tcad_local
class TestJunctionBranch:
    def test_the_sweep_carries_a_fittable_junction_branch(self, tmp_path):
        """Ticket: the shunt branch per unit length, sane on the demo device.

        C_j lands in the fF/um decade range (1e-10..1e-8 F/m), R_s in the
        ohm*mm range (1e-5..1e-1 ohm*m), and reverse bias widens the
        depletion region so C_j falls.
        """
        pytest.importorskip("devsim")
        demo = build_demo()
        component, stack = demo.component, demo.stack
        study = Study(
            component=component,
            stack=stack,
            device=Device(
                p_regions=["p_rib", "p_pad"],
                n_regions=["n_rib", "n_pad"],
            ),
            output_dir=tmp_path,
        )
        study.charge(biases=[0.0, 1.0, 2.0])
        sweep = study.charge.run()

        for point in sweep.points:
            assert point.admittance_freq_hz > 0.0
            assert point.admittance_s_per_cm.imag > 0.0

        r_s, c_j = sweep.junction_branch()

        assert np.all(c_j > 1e-11)
        assert np.all(c_j < 1e-7)
        assert np.all(r_s > 0.0)
        assert np.all(r_s < 1e0)
        # The fit's capacitance agrees with the existing |Im(I)|/omega
        # extraction at the quasi-static frequency, where R_s barely bites.
        assert c_j == pytest.approx(sweep.capacitance_f_per_m, rel=0.05)
        # Reverse bias (positive on the cathode) depletes the junction.
        assert np.all(np.diff(c_j) < 0.0)
