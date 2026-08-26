"""Runtime-gated drift-diffusion solve tests (require DEVSIM).

Deselected by default (see pyproject addopts); run with
``pytest -m tcad_local``, mirroring the ``meep_local`` convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.tcad import ChargeTransportSim, StepDoping

from .conftest import build_pn_device

pytest.importorskip("devsim")

pytestmark = pytest.mark.tcad_local


@pytest.fixture(scope="module")
def solved_sweep(tmp_path_factory):
    comp, stack = build_pn_device()
    sim = ChargeTransportSim()
    sim.set_output_dir(str(tmp_path_factory.mktemp("tcad-runtime")))
    sim.set_stack(stack)
    sim.set_airbox(margin_x=3.0, margin_y=3.0, z_above=2.0, z_below=2.0)
    sim.set_geometry(comp)
    sim.set_cross_section("x=0")
    sim.add_contact(name="anode", layer_a="p_rib", layer_b="sio2")
    sim.add_contact(name="cathode", layer_a="n_rib", layer_b="sio2")
    sim.add_doping(
        StepDoping(region="p_rib", dopant_type="acceptor", concentration_cm3=1e18)
    )
    sim.add_doping(
        StepDoping(region="n_rib", dopant_type="donor", concentration_cm3=1e18)
    )
    sim.mesh(preset="coarse", refined_mesh_size=0.02, max_mesh_size=40.0, verbose=False)
    return sim.sweep([0.0, -0.5, -1.0], contact="cathode")


class TestDriftDiffusionSolve:
    def test_carrier_maps_physical(self, solved_sweep):
        point = solved_sweep.points[0]
        assert np.all(point.carriers.electrons_cm3 > 0.0)
        assert np.all(point.carriers.holes_cm3 > 0.0)
        # Majority carriers approach the doping levels somewhere on each side.
        assert point.carriers.holes_cm3.max() == pytest.approx(1e18, rel=0.5)
        assert point.carriers.electrons_cm3.max() == pytest.approx(1e18, rel=0.5)

    def test_capacitance_positive_and_decreasing_in_reverse_bias(self, solved_sweep):
        c = solved_sweep.capacitance_f_per_cm
        assert np.all(c > 0.0)
        # Depletion widens under reverse bias: C(V) decreases.
        assert c[-1] < c[0]
