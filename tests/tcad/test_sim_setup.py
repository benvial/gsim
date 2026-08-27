"""Hermetic tests for the ChargeTransportSim device setup and solve wiring.

DEVSIM is faked (see conftest); assertions target the external seams:
configuration in, DEVSIM setup calls and generated artifacts out.
"""

from __future__ import annotations

import meshio
import numpy as np
import pytest

from gsim.tcad import ChargeTransportSim, StepDoping
from gsim.tcad.mesh import UM_TO_CM

from .conftest import build_pn_device


def _make_meshed_sim(tmp_path):
    comp, stack = build_pn_device()
    sim = ChargeTransportSim()
    sim.set_output_dir(str(tmp_path))
    sim.set_stack(stack)
    sim.set_airbox(margin_x=3.0, margin_y=3.0, z_above=2.0, z_below=2.0)
    sim.set_geometry(comp)
    sim.set_cross_section("x=0")
    sim.add_contact(name="anode", layer_a="p_rib", layer_b="sio2")
    sim.add_contact(name="cathode", layer_a="n_rib", layer_b="sio2")
    sim.add_interface(name="junction", layer_a="p_rib", layer_b="n_rib")
    sim.add_doping(
        StepDoping(region="p_rib", dopant_type="acceptor", concentration_cm3=1e18)
    )
    sim.add_doping(
        StepDoping(region="n_rib", dopant_type="donor", concentration_cm3=1e18)
    )
    sim.mesh(preset="coarse", refined_mesh_size=0.05, max_mesh_size=40.0, verbose=False)
    return sim


@pytest.fixture(scope="module")
def meshed_sim(tmp_path_factory):
    return _make_meshed_sim(tmp_path_factory.mktemp("tcad"))


@pytest.fixture(autouse=True)
def _fresh_device(request):
    """Solver-side state must not leak between tests sharing the mesh."""
    if "meshed_sim" in request.fixturenames:
        request.getfixturevalue("meshed_sim").reset_device()


class TestMeshing:
    def test_shared_mesh_and_scaled_copy(self, meshed_sim):
        # The shared native-2D mesh is the geometry source ...
        assert meshed_sim.mesh_path is not None
        assert meshed_sim.mesh_path.exists()
        # ... and the DEVSIM copy is the same mesh with cm coordinates.
        assert meshed_sim.devsim_mesh_path is not None
        assert meshed_sim.devsim_mesh_path.exists()
        um_mesh = meshio.read(str(meshed_sim.mesh_path))
        cm_mesh = meshio.read(str(meshed_sim.devsim_mesh_path))
        np.testing.assert_allclose(
            cm_mesh.points, um_mesh.points * UM_TO_CM, atol=1e-12
        )
        assert set(um_mesh.field_data) == set(cm_mesh.field_data)
        assert {"p_rib", "n_rib", "anode", "cathode"} <= set(cm_mesh.field_data)

    def test_mesh_requires_contacts(self, tmp_path):
        comp, stack = build_pn_device()
        sim = ChargeTransportSim()
        sim.set_output_dir(str(tmp_path))
        sim.set_stack(stack)
        sim.set_geometry(comp)
        sim.set_cross_section("x=0")
        with pytest.raises(ValueError, match="contact"):
            sim.mesh(preset="coarse", verbose=False)


class TestDeviceSetup:
    def test_devsim_setup_sequence(self, meshed_sim, fake_devsim):
        devsim, sp = fake_devsim
        meshed_sim.setup_device("pn")

        [create] = devsim.called("create_gmsh_mesh")
        assert create["file"] == str(meshed_sim.devsim_mesh_path)

        regions = {c["region"] for c in devsim.called("add_gmsh_region")}
        assert regions == {"p_rib", "n_rib"}
        for call in devsim.called("add_gmsh_region"):
            assert call["gmsh_name"] == call["region"]
            assert call["material"] == "Silicon"

        contacts = {c["name"]: c["region"] for c in devsim.called("add_gmsh_contact")}
        assert contacts == {"anode": "p_rib", "cathode": "n_rib"}

        assert devsim.called("finalize_mesh")
        [created] = devsim.called("create_device")
        assert created["device"] == "pn"

        # Potential-only physics on every region, contact BCs at 0 V.
        assert {args[1] for args in sp.called("CreateSiliconPotentialOnly")} == {
            "p_rib",
            "n_rib",
        }
        assert {args[2] for args in sp.called("CreateSiliconPotentialOnlyContact")} == {
            "anode",
            "cathode",
        }
        # Each electrical contact is driven through a circuit source at 0 V.
        assert devsim.circuit["V_anode"] == 0.0
        assert devsim.circuit["V_cathode"] == 0.0
        sources = {c["name"]: c for c in devsim.called("circuit_element")}
        assert set(sources) == {"V_anode", "V_cathode"}
        assert sources["V_cathode"]["n1"] == "cathode_bias"

        # The P/N junction is a region-region interface, not a contact,
        # with potential continuity from the potential-only stage.
        [iface] = devsim.called("add_gmsh_interface")
        assert iface["name"] == "junction"
        assert {iface["region0"], iface["region1"]} == {"p_rib", "n_rib"}
        potential_continuity = [
            c
            for c in devsim.called("interface_equation")
            if c["name"] == "PotentialEquation"
        ]
        assert len(potential_continuity) == 1
        assert potential_continuity[0]["type"] == "continuous"

    def test_doping_node_models(self, meshed_sim, fake_devsim):
        devsim, _sp = fake_devsim
        meshed_sim.setup_device("pn")

        # NetDoping = Donors - Acceptors on every region.
        net_models = {
            c["region"]: c["equation"]
            for c in devsim.called("node_model")
            if c["name"] == "NetDoping"
        }
        assert net_models == {
            "p_rib": "Donors - Acceptors",
            "n_rib": "Donors - Acceptors",
        }

        # Node values match the analytic profiles evaluated at the node
        # coordinates (fake coords are in cm; profiles take um).
        n_nodes = len(devsim.node_coords["x"])
        np.testing.assert_allclose(
            devsim.node_values[("p_rib", "Acceptors")], [1e18] * n_nodes
        )
        np.testing.assert_allclose(
            devsim.node_values[("p_rib", "Donors")], [0.0] * n_nodes
        )
        np.testing.assert_allclose(
            devsim.node_values[("n_rib", "Donors")], [1e18] * n_nodes
        )

    @pytest.mark.usefixtures("fake_devsim")
    def test_unknown_doping_region_raises(self, meshed_sim):
        sim = meshed_sim.model_copy()
        sim.doping = [
            StepDoping(
                region="no_such_region",
                dopant_type="donor",
                concentration_cm3=1e18,
            )
        ]
        with pytest.raises(ValueError, match="no_such_region"):
            sim.setup_device()

    def test_setup_before_mesh_raises(self):
        sim = ChargeTransportSim()
        sim.add_doping(
            StepDoping(region="p_rib", dopant_type="acceptor", concentration_cm3=1e18)
        )
        with pytest.raises(ValueError, match="mesh"):
            sim.setup_device()

    @pytest.mark.usefixtures("fake_devsim")
    def test_no_doping_raises(self, meshed_sim):
        sim = meshed_sim.model_copy()
        sim.doping = []
        with pytest.raises(ValueError, match="doping"):
            sim.setup_device()


class TestSolveWiring:
    def test_solve_returns_bias_point_with_small_signal_capacitance(
        self, meshed_sim, fake_devsim
    ):
        devsim, _sp = fake_devsim
        point = meshed_sim.solve(-1.0, contact="cathode")

        assert point.bias_v == -1.0
        # Fake device charge is linear in bias: C = charge_per_volt exactly.
        assert point.capacitance_f_per_cm == pytest.approx(
            devsim.charge_per_volt, rel=1e-6
        )
        assert point.capacitance_f_per_m == pytest.approx(
            devsim.charge_per_volt * 1e2, rel=1e-6
        )
        assert set(point.currents_a_per_cm) == {"anode", "cathode"}
        # The swept contact's circuit source carries the requested bias.
        assert devsim.circuit["V_cathode"] == pytest.approx(-1.0)

        # Carrier maps concatenate both regions; coords are back in um.
        n_nodes = len(devsim.node_coords["x"])
        assert point.carriers.x_um.size == 2 * n_nodes
        assert set(point.carriers.region) == {"p_rib", "n_rib"}
        np.testing.assert_allclose(
            point.carriers.x_um[:n_nodes],
            np.asarray(devsim.node_coords["x"]) / UM_TO_CM,
        )

    def test_sweep_orders_points(self, meshed_sim, fake_devsim):
        devsim, _sp = fake_devsim
        biases = [0.0, -0.5, -1.0]
        result = meshed_sim.sweep(biases, contact="cathode")
        assert result.contact == "cathode"
        np.testing.assert_allclose(result.voltages, biases)
        np.testing.assert_allclose(
            result.capacitance_f_per_cm, devsim.charge_per_volt, rtol=1e-6
        )
        assert result.capacitance_f_per_m.shape == (3,)

    @pytest.mark.usefixtures("fake_devsim")
    def test_unknown_sweep_contact_raises(self, meshed_sim):
        with pytest.raises(ValueError, match="gate"):
            meshed_sim.solve(0.0, contact="gate")

    def test_drift_diffusion_initialized_once(self, meshed_sim, fake_devsim):
        _devsim, sp = fake_devsim
        meshed_sim.solve(0.0, contact="cathode")
        meshed_sim.solve(-0.2, contact="cathode")
        # DD assembly happens once per region despite two solves.
        assert len(sp.called("CreateSiliconDriftDiffusion")) == 2
        assert {args[1] for args in sp.called("CreateSiliconDriftDiffusion")} == {
            "p_rib",
            "n_rib",
        }

    def test_carrier_continuity_across_interface(self, meshed_sim, fake_devsim):
        devsim, _sp = fake_devsim
        meshed_sim.solve(0.0, contact="cathode")
        equations = {c["name"] for c in devsim.called("interface_equation")}
        assert {
            "PotentialEquation",
            "ElectronContinuityEquation",
            "HoleContinuityEquation",
        } <= equations


class TestDevsimNamespace:
    """DEVSIM's mesh/device namespace is process-wide, so names must differ."""

    def test_each_setup_claims_its_own_mesh_and_device(self, meshed_sim, fake_devsim):
        devsim, _sp = fake_devsim
        first = meshed_sim.setup_device()
        first_mesh = devsim.called("create_gmsh_mesh")[0]["mesh"]

        meshed_sim.reset_device()
        second = meshed_sim.setup_device()
        second_mesh = devsim.called("create_gmsh_mesh")[1]["mesh"]

        assert first != second
        assert first_mesh != second_mesh

    def test_reset_releases_the_device_and_its_mesh(self, meshed_sim, fake_devsim):
        devsim, _sp = fake_devsim
        device = meshed_sim.setup_device()
        mesh_name = devsim.called("create_gmsh_mesh")[0]["mesh"]

        meshed_sim.reset_device()

        assert devsim.called("delete_device") == [{"device": device}]
        assert devsim.called("delete_mesh") == [{"mesh": mesh_name}]

    def test_the_mesh_name_is_recognisable(self, meshed_sim, fake_devsim):
        devsim, _sp = fake_devsim
        meshed_sim.setup_device()
        name = devsim.called("create_gmsh_mesh")[0]["mesh"]
        assert name.startswith("gsim_tcad_mesh")
