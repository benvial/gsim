"""The realistic rib Phase shifter, and the Staircases that follow it.

A rib on a thinner slab, a lightly doped core, moderately doped plus
Regions and heavily doped contact Regions under the metal. Nothing here
solves: the charge Stage is seeded with a synthetic Carrier map, so what
is under test is what the builder draws and describes, and that the RF
Staircase draws each Strip at the height of the silicon it stands for.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from gsim.modulator import pn_phase_shifter, rib_phase_shifter
from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap

RIB_HEIGHT = 0.22
SLAB_HEIGHT = 0.09
RIB_REGIONS = {"n_rib", "p_rib"}


@pytest.fixture(scope="module")
def device():
    """The rib Phase shifter at its published defaults, drawn once."""
    return rib_phase_shifter()


@pytest.fixture
def study(device, tmp_path):
    """The preset over the rib device, with its own electrodes."""
    return pn_phase_shifter(
        component=device.component,
        stack=device.stack,
        device=device.device,
        electrodes=device.electrodes,
        output_dir=tmp_path,
    )


def depleted_map(study, bias_v: float = 2.0) -> CarrierMap:
    """A Carrier map across the doped slab: doped, depleted at the Junction."""
    low, high = study.layout.doped_span
    centre = study.layout.junction_position
    y = np.linspace(low, high, 801)
    z = np.linspace(0.0, RIB_HEIGHT, 12)
    yy, zz = (a.ravel() for a in np.meshgrid(y, z, indexing="ij"))
    depleted = np.abs(yy - centre) < 0.07 * np.sqrt(1.0 + bias_v)
    n_side = yy < centre
    return CarrierMap(
        x_um=yy,
        y_um=zz,
        region=["n_rib" if side else "p_rib" for side in n_side],
        electrons_cm3=np.where(n_side & ~depleted, 3e17, 1e5),
        holes_cm3=np.where(~n_side & ~depleted, 5e17, 1e5),
    )


@pytest.fixture
def seeded(study):
    """The Study with a depleted 2 V Bias point in its charge Stage."""
    study.charge.seed(
        BiasSweepResult(
            contact="cathode",
            points=[BiasPoint(bias_v=2.0, carriers=depleted_map(study))],
        )
    )
    return study


class TestTheDrawnDevice:
    def test_the_rib_stands_taller_than_the_slab_beside_it(self, study):
        spans = study.layout.region_spans
        for name in study.device.doped_regions:
            expected = RIB_HEIGHT if name in RIB_REGIONS else SLAB_HEIGHT
            assert spans[name].z == pytest.approx((0.0, expected)), name

    def test_the_junction_is_at_the_centre_of_the_rib(self, study, device):
        assert set(study.layout.junction.regions) == RIB_REGIONS
        assert study.layout.junction_position == pytest.approx(device.center_um)

    def test_the_metal_lands_on_the_heavily_doped_regions(self, study):
        landed = {contact.name: contact.region for contact in study.layout.contacts}
        assert landed == {"cathode": "n_contact", "anode": "p_contact"}

    def test_each_region_is_doped_at_its_own_level(self, study, device):
        profiles = {p.region: p for p in study.charge.simulation().doping}

        assert profiles["p_rib"].concentration_cm3 == pytest.approx(5e17)
        assert profiles["n_rib"].concentration_cm3 == pytest.approx(3e17)
        assert profiles["p_contact"].concentration_cm3 == pytest.approx(1e20)
        assert profiles["n_contact"].dopant_type == "donor"
        assert device.doping_cm3 == {
            name: profile.concentration_cm3 for name, profile in profiles.items()
        }

    def test_the_rf_line_takes_the_devices_electrodes(self, study, device):
        assert study.rf.electrodes == device.electrodes


class TestTheRFStaircaseFollowsIt:
    def test_each_strip_stands_at_the_height_of_its_region(self, seeded):
        seeded.rf(n_strips=10, strips_per_region=2)
        strips = seeded.rf.staircase().strips

        centres = 0.5 * (strips.edges_um[1:] + strips.edges_um[:-1])
        rib = seeded.layout.junction_span.h
        in_rib = (centres > rib[0]) & (centres < rib[1])
        np.testing.assert_allclose(strips.zmax_um[in_rib], RIB_HEIGHT)
        np.testing.assert_allclose(strips.zmax_um[~in_rib], SLAB_HEIGHT)
        # Ten across the rib, two across each of the six slab Regions.
        assert strips.count == 10 + 6 * 2

    def test_the_default_strips_leave_one_depleted(self, seeded):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            seeded.rf.staircase()


class TestTheOpticalStaircase:
    def test_a_staircase_drawn_at_the_rib_height_says_so(self, seeded):
        """The optical Staircase still draws every Strip at the Junction's
        height, which is the slab's only when the slab is the rib."""
        seeded.optical(route="palace", n_strips=5)
        point = seeded.carriers.run().points[0]

        with pytest.warns(UserWarning, match="rib's height"):
            seeded.optical.staircase(point)
