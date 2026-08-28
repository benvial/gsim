"""The charge Stage of a Study: configuration, run, and the missing extra."""

from __future__ import annotations

import sys

import pytest

from gsim.tcad.results import BiasSweepResult


class TestConfiguration:
    def test_defaults_are_readable(self, study):
        assert study.charge.biases == [0.0]
        assert study.charge.window is None
        assert study.charge.has_run is False

    def test_the_section_is_callable(self, study):
        assert study.charge(biases=[0.0, -1.0]) is study.charge
        assert study.charge.biases == [0.0, -1.0]

    def test_unknown_setting_is_rejected(self, study):
        with pytest.raises(ValueError, match="nope"):
            study.charge(nope=1)


class TestSimulationAssembly:
    def test_the_sim_carries_the_derived_contacts_and_interfaces(self, study):
        sim = study.charge.simulation()
        declared = {spec.name for spec in sim.contact_specs}
        assert declared == {
            "anode",
            "cathode",
            "junction",
            "n_pad_n_rib",
            "p_rib_p_pad",
        }

    def test_the_sim_carries_one_doping_profile_per_doped_region(self, study):
        sim = study.charge.simulation()
        by_region = {profile.region: profile for profile in sim.doping}
        assert set(by_region) == {"p_rib", "p_pad", "n_rib", "n_pad"}
        assert by_region["p_rib"].dopant_type == "acceptor"
        assert by_region["n_rib"].dopant_type == "donor"
        assert by_region["p_rib"].concentration_cm3 == 1e18

    def test_the_derived_window_reaches_the_sim(self, study):
        sim = study.charge.simulation()
        assert sim.cross_section.window == pytest.approx(study.layout.window)

    def test_an_explicit_window_reaches_the_sim(self, study):
        study.charge(window=(-21.0, -19.0))
        sim = study.charge.simulation()
        assert sim.cross_section.window == pytest.approx((-21.0, -19.0))


class TestMissingExtra:
    def test_running_without_devsim_names_the_extra(self, study, monkeypatch):
        monkeypatch.setitem(sys.modules, "devsim", None)
        with pytest.raises(ImportError, match=r"gsim\[tcad\]"):
            study.charge.run()

    def test_the_check_happens_before_any_meshing(self, study, monkeypatch):
        monkeypatch.setitem(sys.modules, "devsim", None)
        meshed = []
        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.mesh",
            lambda self, **kwargs: meshed.append(kwargs),
        )
        with pytest.raises(ImportError):
            study.charge.run()
        assert meshed == []


class TestSweptContact:
    def test_defaults_to_the_contact_on_the_n_side(self, study, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.mesh", lambda s, **k: None
        )
        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.sweep",
            lambda self, biases, contact=None: (
                calls.append(contact)
                or BiasSweepResult(contact=str(contact), points=[])
            ),
        )
        monkeypatch.setattr("gsim.modulator.charge.require_devsim", lambda: None)

        assert study.charge.contact is None
        study.charge.run()

        assert calls == ["cathode"]


class TestDevsimRelease:
    def test_a_re_run_releases_the_previous_devsim_device(self, study, monkeypatch):
        released = []

        def record_reset(sim):
            released.append(sim)

        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.mesh", lambda s, **k: None
        )
        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.sweep",
            lambda self, biases, contact=None: BiasSweepResult(
                contact=str(contact), points=[]
            ),
        )
        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.reset_device",
            record_reset,
        )
        monkeypatch.setattr("gsim.modulator.charge.require_devsim", lambda: None)

        study.charge.run()
        assert released == []

        study.charge(biases=[0.0, 0.5])
        study.charge.run()
        assert len(released) == 1


class TestRun:
    def test_returns_the_bias_sweep_result_type(self, study, monkeypatch):
        sweep = BiasSweepResult(contact="cathode", points=[])
        calls = []

        def fake_sweep(_self, biases, *, contact=None):
            calls.append((list(biases), contact))
            return sweep

        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.mesh", lambda s, **k: None
        )
        monkeypatch.setattr("gsim.tcad.sim.ChargeTransportSim.sweep", fake_sweep)
        monkeypatch.setattr("gsim.modulator.charge.require_devsim", lambda: None)

        study.charge(biases=[0.0, -1.0], contact="cathode")
        result = study.charge.run()

        assert result is sweep
        assert calls == [([0.0, -1.0], "cathode")]
        assert study.charge.contact == "cathode"
        assert study.charge.run() is sweep
        assert len(calls) == 1

    def test_verbose_reports_the_stage(
        self, phase_shifter, device, tmp_path, monkeypatch, capsys
    ):
        from gsim.modulator import Study

        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.mesh", lambda s, **k: None
        )
        monkeypatch.setattr(
            "gsim.tcad.sim.ChargeTransportSim.sweep",
            lambda self, biases, contact=None: BiasSweepResult(contact="c", points=[]),
        )
        monkeypatch.setattr("gsim.modulator.charge.require_devsim", lambda: None)

        component, stack = phase_shifter
        study = Study(
            component=component,
            stack=stack,
            device=device,
            output_dir=tmp_path,
            verbose=True,
        )
        study.charge.run()

        out = capsys.readouterr().out.splitlines()
        assert len(out) == 2
        assert all("charge" in line for line in out)
