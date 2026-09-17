"""The Staircase against the drawn device, on one charge solve.

Every other numeric gate on this branch compares two things that share a
representation. ``test_palace_route_runtime`` solves one Staircase in both
Routes, so its 1% agreement is a statement about the two solvers.
``tests/femwell/test_staircase_convergence`` varies only the materials on
a fixed mesh. Neither can see a Staircase whose *geometry* is not the
drawn device — which is what ticket 19 turned out to be: the Staircase
omitted the drawn electrodes, and no amount of strip count moved the
0.18 that cost.

The comparison here is the one a user actually chooses between: the
Staircase Route's answer against the drawn device solved with a
continuous ``eps(x, y)``. femwell spans both representations, so no
Palace binary is involved and the whole gate is one mesh per strip count.

It cannot assert tight agreement — the Staircase is a deliberately
coarser model and some gap is legitimate. What it asserts is that the gap
is bounded, that it is dominated by the strip count rather than by the
geometry, and that raising the strip count closes it. The failure ticket
19 describes shows up here as a gap that N cannot move.

The index shift carries the tighter bound of the two: a constant index
offset between representations cancels out of ``VpiL``, and an error that
moves with bias does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from gsim.modulator import Device, Study
from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap

from .conftest import CENTER_Y, HALF_WIDTH, PAD_WIDTH, RIB_HEIGHT, build_demo

pytest.importorskip("gmsh")
pytest.importorskip("femwell")
pytest.importorskip("skfem")

DOPING_CM3 = 1e18
DEPLETED_CM3 = 1e10
SLAB = (CENTER_Y - HALF_WIDTH - PAD_WIDTH, CENTER_Y + HALF_WIDTH + PAD_WIDTH)

#: The bias pair the index shift is measured across (V).
BIASES = (0.0, 4.0)

#: Strip counts the gate compares against the continuous reference.
STRIP_COUNTS = (2, 16)

#: What the coarsest staircase may differ from the drawn device by.
N_EFF_RTOL = 0.01

#: And what the finest one may — the bound the geometry has to earn.
#: Each strip count is meshed afresh, so a few parts in ten thousand of
#: this is mesh noise rather than representation error.
N_EFF_RTOL_FINE = 0.002

#: The index shift is the quantity VpiL is computed from, so its bound is
#: the tighter one: a constant index offset cancels out of VpiL, an error
#: that moves with bias does not.
INDEX_SHIFT_RTOL_FINE = 0.10


def graded_carriers(bias_v: float) -> CarrierMap:
    """A depletion region that widens smoothly with reverse bias.

    Not a charge solve — a monotone stand-in for one. The profile is
    graded rather than step-like on purpose: a step is either resolved by
    a strip edge or not, so its staircase error jumps with N instead of
    shrinking, and a gate on "the gap responds to N" would be measuring
    the sampling luck of the edges.

    Every doped region is labelled, so the continuous route perturbs the
    pads exactly as the strips do; a map that names only the rib would
    make the two representations differ in their doping as well as in
    their binning.
    """
    y = np.linspace(SLAB[0], SLAB[1], 161)
    z = np.linspace(0.0, RIB_HEIGHT, 9)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    yy, zz = yy.ravel(), zz.ravel()

    # Depleted at the junction, doped away from it, with an edge that
    # softens and widens as the reverse bias grows.
    width = 0.06 * np.sqrt(1.0 + abs(bias_v))
    depletion = 1.0 / (1.0 + np.exp(-(np.abs(yy - CENTER_Y) - width) / 0.04))
    n_side = yy < CENTER_Y

    electrons = DEPLETED_CM3 + DOPING_CM3 * depletion * n_side
    holes = DEPLETED_CM3 + DOPING_CM3 * depletion * ~n_side
    return CarrierMap(
        x_um=yy,
        y_um=zz,
        region=[
            ("n_pad" if v < CENTER_Y - HALF_WIDTH else "n_rib")
            if v < CENTER_Y
            else ("p_pad" if v > CENTER_Y + HALF_WIDTH else "p_rib")
            for v in yy
        ],
        electrons_cm3=electrons,
        holes_cm3=holes,
    )


def study_at(output_dir) -> Study:
    """A Study over the demo phase shifter with the bias pair canned."""
    demo = build_demo()
    study = Study(
        component=demo.component,
        stack=demo.stack,
        device=Device(
            p_regions=["p_rib", "p_pad"],
            n_regions=["n_rib", "n_pad"],
            p_doping_cm3=DOPING_CM3,
            n_doping_cm3=DOPING_CM3,
        ),
        plane="x=0",
        output_dir=output_dir,
    )
    study.charge.seed(
        BiasSweepResult(
            contact="cathode",
            points=[BiasPoint(bias_v=v, carriers=graded_carriers(v)) for v in BIASES],
        )
    )
    return study


def solve(output_dir, *, n_strips: int | None) -> tuple[float, float]:
    """``(Re n_eff at the first bias, index shift across the pair)``."""
    study = study_at(output_dir)
    study.optical(route="femwell", n_strips=n_strips, num_modes=1)
    sweep = study.optical.run()
    return float(sweep.n_eff[0].real), float(sweep.index_shift[-1])


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """The drawn device, and the Staircase of it at two strip counts."""
    root = tmp_path_factory.mktemp("representation")
    drawn = solve(root / "continuous", n_strips=None)
    staircase = {n: solve(root / f"strips-{n}", n_strips=n) for n in STRIP_COUNTS}
    return drawn, staircase


class TestTheStaircaseIsTheDrawnDevice:
    def test_the_coarsest_staircase_is_already_the_same_waveguide(self, measured):
        """Two strips is a crude model; it is not a different guide."""
        (n_drawn, _), staircase = measured
        for n_strips, (n_eff, _) in staircase.items():
            assert abs(n_eff - n_drawn) < N_EFF_RTOL * n_drawn, n_strips

    def test_the_finest_staircase_lands_on_the_drawn_index(self, measured):
        (n_drawn, _), staircase = measured
        n_eff, _ = staircase[max(STRIP_COUNTS)]
        assert abs(n_eff - n_drawn) < N_EFF_RTOL_FINE * n_drawn

    def test_the_finest_staircase_lands_on_the_drawn_index_shift(self, measured):
        """The bound VpiL depends on, and the tighter of the two."""
        (_, shift_drawn), staircase = measured
        _, shift = staircase[max(STRIP_COUNTS)]
        assert shift_drawn > 0.0
        assert abs(shift - shift_drawn) < INDEX_SHIFT_RTOL_FINE * shift_drawn


class TestTheGapRespondsToStripCount:
    """What separates a coarse model from a wrong one.

    Before ticket 19 the staircase sat 0.384 from the continuous answer
    while N = 2 to 16 moved it by 0.010: a gap the strip count could not
    touch, because it was the geometry and not the binning.
    """

    def test_more_strips_move_the_index_towards_the_drawn_device(self, measured):
        (n_drawn, _), staircase = measured
        coarse = abs(staircase[min(STRIP_COUNTS)][0] - n_drawn)
        fine = abs(staircase[max(STRIP_COUNTS)][0] - n_drawn)
        assert fine < coarse

    def test_more_strips_move_the_index_shift_towards_it_too(self, measured):
        (_, shift_drawn), staircase = measured
        coarse = abs(staircase[min(STRIP_COUNTS)][1] - shift_drawn)
        fine = abs(staircase[max(STRIP_COUNTS)][1] - shift_drawn)
        # And by a margin: the binning is what is left to converge, so
        # four times the strips has to more than halve the error.
        assert fine < 0.5 * coarse


@pytest.mark.tcad_local
class TestOnARealChargeSolve:
    def test_the_gate_holds_off_devsim_carrier_maps(self, tmp_path):
        """The same bound, with the charge solve in front of it."""
        pytest.importorskip("devsim")

        def solved(output_dir, *, n_strips):
            study = study_at(output_dir)
            study.charge.invalidate()
            study.charge(biases=list(BIASES))
            study.optical(route="femwell", n_strips=n_strips, num_modes=1)
            sweep = study.optical.run()
            return float(sweep.n_eff[0].real), float(sweep.index_shift[-1])

        n_drawn, shift_drawn = solved(tmp_path / "continuous", n_strips=None)
        n_eff, shift = solved(tmp_path / "strips", n_strips=max(STRIP_COUNTS))

        assert abs(n_eff - n_drawn) < N_EFF_RTOL * n_drawn
        assert abs(shift - shift_drawn) < INDEX_SHIFT_RTOL_FINE * abs(shift_drawn)
