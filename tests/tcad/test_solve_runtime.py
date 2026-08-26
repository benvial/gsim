"""Runtime-gated drift-diffusion solve tests (require DEVSIM).

Deselected by default (see pyproject addopts); run with
``pytest -m tcad_local``, mirroring the ``meep_local`` convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.common.stack.junction import PNJunctionConfig
from gsim.tcad import ChargeTransportSim, StepDoping
from gsim.tcad.validation import compare_capacitance, estimate_depletion_width_um

from .conftest import build_pn_device

pytest.importorskip("devsim")

pytestmark = pytest.mark.tcad_local

_NA_CM3 = 1e18
_ND_CM3 = 1e18
_RIB_HEIGHT_UM = 0.22
# Positive cathode (n-side) bias reverse-biases the junction.
_REVERSE_BIASES = [0.0, 0.5, 1.0]


@pytest.fixture(scope="module")
def solved_sim(tmp_path_factory):
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
        StepDoping(region="p_rib", dopant_type="acceptor", concentration_cm3=_NA_CM3)
    )
    sim.add_doping(
        StepDoping(region="n_rib", dopant_type="donor", concentration_cm3=_ND_CM3)
    )
    sim.mesh(preset="coarse", refined_mesh_size=0.02, max_mesh_size=40.0, verbose=False)
    return sim


@pytest.fixture(scope="module")
def solved_sweep(solved_sim):
    return solved_sim.sweep(_REVERSE_BIASES, contact="cathode")


class TestDriftDiffusionSolve:
    def test_carrier_maps_physical(self, solved_sweep):
        point = solved_sweep.points[0]
        assert np.all(point.carriers.electrons_cm3 > 0.0)
        assert np.all(point.carriers.holes_cm3 > 0.0)
        # Majority carriers approach the doping levels somewhere on each side.
        assert point.carriers.holes_cm3.max() == pytest.approx(_NA_CM3, rel=0.5)
        assert point.carriers.electrons_cm3.max() == pytest.approx(_ND_CM3, rel=0.5)

    def test_capacitance_positive_and_decreasing_in_reverse_bias(self, solved_sweep):
        c = solved_sweep.capacitance_f_per_cm
        assert np.all(c > 0.0)
        # Depletion widens under reverse bias: C(V) decreases.
        assert c[-1] < c[0]


class TestAnalyticCrossCheck:
    def test_capacitance_tracks_sze_model(self, solved_sweep):
        junction = PNJunctionConfig(na_cm3=_NA_CM3, nd_cm3=_ND_CM3)
        comparison = compare_capacitance(
            junction, solved_sweep, height_um=_RIB_HEIGHT_UM
        )
        # Fully depleted regime on an abrupt symmetric junction: the numeric
        # small-signal C(V) must track the depletion approximation.
        assert comparison.within(0.35)

    def test_carrier_profile_edges_match_depletion_extents(self, solved_sweep):
        junction = PNJunctionConfig(
            na_cm3=_NA_CM3, nd_cm3=_ND_CM3, v_reverse=_REVERSE_BIASES[-1]
        )
        carriers = solved_sweep.points[-1].carriers
        # 1D cut across the junction at mid rib height.
        band = np.abs(carriers.y_um - _RIB_HEIGHT_UM / 2) < 0.03
        order = np.argsort(carriers.x_um[band])
        width = estimate_depletion_width_um(
            carriers.x_um[band][order],
            carriers.electrons_cm3[band][order],
            carriers.holes_cm3[band][order],
            na_cm3=_NA_CM3,
            nd_cm3=_ND_CM3,
        )
        assert width == pytest.approx(junction.w_um, rel=0.5)
