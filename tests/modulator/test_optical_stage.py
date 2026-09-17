"""The optical Stage: its Window, its configuration, and the missing extra.

Everything here runs without femwell, skfem or DEVSIM: the Window
derivation and the Stage's configuration are pure derivation, and the
missing-extra path is exactly the one a user without the extra hits.
The real solve lives in ``test_optical_stage_runtime.py``.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

from .conftest import CENTER_Y, HALF_WIDTH, PAD_WIDTH, RIB_HEIGHT

JUNCTION_Y = CENTER_Y


class TestConfiguration:
    def test_defaults_are_readable(self, study):
        assert study.optical.wavelength_um == 1.55
        assert study.optical.num_modes == 1
        assert study.optical.window is None
        assert study.optical.has_run is False

    def test_the_section_is_callable(self, study):
        assert study.optical(wavelength_um=1.31, num_modes=2) is study.optical
        assert study.optical.wavelength_um == 1.31
        assert study.optical.num_modes == 2

    def test_unknown_setting_is_rejected(self, study):
        with pytest.raises(ValueError, match="nope"):
            study.optical(nope=1)

    def test_a_nonsense_wavelength_is_rejected(self, study):
        with pytest.raises(ValueError):
            study.optical(wavelength_um=0.0)

    def test_at_least_one_mode_must_be_asked_for(self, study):
        with pytest.raises(ValueError):
            study.optical(num_modes=0)


class TestDerivedWindow:
    def test_the_window_is_a_box_around_the_rib_not_the_doped_slab(self, study):
        window = study.optical.simulation().cross_section.window

        # Centred on the junction, and wider than the charge window, which
        # is clipped to the doped slab between the contacts (ADR 0002).
        assert window == pytest.approx((JUNCTION_Y - 2.0, JUNCTION_Y + 2.0))
        assert window != pytest.approx(study.layout.window)

    def test_the_vertical_window_clears_the_guiding_layer(self, study):
        window_z = study.optical.simulation().cross_section.window_z

        assert window_z == pytest.approx((-1.0, RIB_HEIGHT + 1.0))

    def test_the_margin_sizes_the_window(self, study):
        study.optical(mode_margin_um=3.0, z_above_um=0.5, z_below_um=0.25)

        cross_section = study.optical.simulation().cross_section

        assert cross_section.window == pytest.approx(
            (JUNCTION_Y - 3.0, JUNCTION_Y + 3.0)
        )
        assert cross_section.window_z == pytest.approx((-0.25, RIB_HEIGHT + 0.5))

    def test_an_explicit_window_overrides_the_derivation(self, study):
        study.optical(window=(-21.0, -19.0), window_z=(-0.5, 0.8))

        cross_section = study.optical.simulation().cross_section

        assert cross_section.window == pytest.approx((-21.0, -19.0))
        assert cross_section.window_z == pytest.approx((-0.5, 0.8))

    def test_the_junction_is_where_the_doped_regions_meet(self, study):
        assert study.layout.junction_position == pytest.approx(JUNCTION_Y)

    def test_the_charge_window_spans_the_whole_doped_slab(self, study):
        """The Window the optical Stage is deliberately not reusing."""
        assert study.layout.window == pytest.approx(
            (
                CENTER_Y - HALF_WIDTH - PAD_WIDTH - 0.5,
                CENTER_Y + HALF_WIDTH + PAD_WIDTH + 0.5,
            )
        )


class TestOwnMesh:
    def test_the_optical_stage_writes_into_its_own_directory(self, study):
        sim = study.optical.simulation()
        assert sim.output_dir == study.stage_dir("optical")
        assert sim.output_dir != study.stage_dir("charge")

    def test_the_perturbed_regions_default_to_the_doped_ones(self, study):
        assert study.optical.perturbed_region_names() == study.device.doped_regions

    def test_the_perturbed_regions_are_overridable(self, study):
        study.optical(perturbed_regions=["p_rib", "n_rib"])
        assert study.optical.perturbed_region_names() == ["p_rib", "n_rib"]


class TestMissingExtra:
    def test_running_without_femwell_names_the_extra(self, study, monkeypatch):
        monkeypatch.setitem(sys.modules, "femwell", None)
        with pytest.raises(ImportError, match=r"gsim\[femwell\]"):
            study.optical.run()

    def test_the_extra_is_checked_before_anything_is_meshed(self, study, monkeypatch):
        """A user without femwell pays for no mesh and no charge solve."""
        monkeypatch.setitem(sys.modules, "femwell", None)

        def fail(*_args, **_kwargs):
            raise AssertionError("the charge stage must not run")

        monkeypatch.setattr("gsim.modulator.charge.ChargeStage._solve", fail)
        with pytest.raises(ImportError):
            study.optical.run()


class TestInvalidation:
    def test_a_carriers_change_drops_the_optical_result(self, study):
        order = list(study.stages)
        assert order.index("carriers") < order.index("optical")
        assert study.optical in study.carriers._downstream
        assert study.optical in study.charge._downstream

    def test_the_optical_stage_feeds_the_line_stage(self, study):
        assert study.optical._downstream == [study.line]


class TestStripExtent:
    """What the Strips tile, and what happens when they cannot reach it."""

    def test_the_default_is_the_doped_slab_not_the_rib(self, biased):
        """Ticket 19: strips on the rib alone solve the wrong waveguide."""
        extent = biased.optical.strip_extent()

        assert extent == pytest.approx(biased.layout.doped_span)
        assert extent[0] < biased.layout.junction_span.h[0]
        assert extent[1] > biased.layout.junction_span.h[1]

    def test_a_chosen_span_is_taken_as_given(self, biased):
        biased.optical(strip_span=(CENTER_Y - 0.1, CENTER_Y + 0.1))

        assert biased.optical.strip_extent() == pytest.approx(
            (CENTER_Y - 0.1, CENTER_Y + 0.1)
        )

    def test_a_chosen_span_is_narrowed_by_the_map_too(self, biased):
        """The preset chooses the span, so the clamp has to reach it."""
        biased.optical(strip_span=biased.layout.doped_span)
        carriers = biased.charge.result.points[0].carriers
        carriers.x_um = np.clip(carriers.x_um, CENTER_Y - 0.2, CENTER_Y + 0.2)

        with pytest.warns(UserWarning, match="the extent asked for"):
            extent = biased.optical.strip_extent(carriers)

        assert extent == pytest.approx((CENTER_Y - 0.2, CENTER_Y + 0.2))

    def test_a_carrier_map_narrower_than_the_slab_narrows_the_default(self, biased):
        """Strips cannot outrun the map they average, and say so."""
        carriers = biased.charge.result.points[0].carriers
        carriers.x_um = np.clip(carriers.x_um, CENTER_Y - 0.2, CENTER_Y + 0.2)

        with pytest.warns(UserWarning, match="derived for this stage"):
            extent = biased.optical.strip_extent(carriers)

        assert extent == pytest.approx((CENTER_Y - 0.2, CENTER_Y + 0.2))

    def test_a_map_covering_the_slab_narrows_nothing_and_says_nothing(self, biased):
        carriers = biased.charge.result.points[0].carriers

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            extent = biased.optical.strip_extent(carriers)

        assert extent == pytest.approx(biased.layout.doped_span)


class TestUnperturbedIndex:
    """What the Strips are before the carriers move them."""

    def test_it_is_the_drawn_junction_material_not_a_textbook_value(self, study):
        """The continuous route perturbs this index; so must the strips."""
        from gsim.common.stack.staircase import DEFAULT_SI_INDEX

        index = study.optical.unperturbed_index()

        # The demo draws its doped silicon at eps = 11.9, which is not the
        # database's silicon at 1.55 um.
        assert index == pytest.approx(11.9**0.5)
        assert index != pytest.approx(DEFAULT_SI_INDEX)

    def test_a_chosen_index_is_taken_as_given(self, study):
        study.optical(strip_index=3.5)

        assert study.optical.unperturbed_index() == 3.5


class TestSurroundings:
    """The drawn device, redrawn around the Strips."""

    def test_the_drawn_slab_reaches_the_staircase(self, biased):
        names = {region.name for region in biased.optical.surroundings()}

        assert any(name.startswith("slab90") for name in names)

    def test_the_drawn_metal_reaches_it_as_a_conductor(self, biased):
        """The electrodes are what omitting cost 0.18 on this device."""
        regions = {region.name: region for region in biased.optical.surroundings()}

        assert regions["cathode_metal"].layer_type == "conductor"
        assert regions["anode_metal"].layer_type == "conductor"

    def test_the_doped_silicon_the_strips_replace_does_not(self, biased):
        names = {region.name for region in biased.optical.surroundings()}

        assert not names & {"n_pad", "n_rib", "p_rib", "p_pad"}

    def test_strips_on_the_rib_leave_the_pads_as_drawn_silicon(self, biased):
        """Narrow the strips and the pads come back, unperturbed."""
        biased.optical(strip_span=(CENTER_Y - HALF_WIDTH, CENTER_Y + HALF_WIDTH))

        names = {region.name for region in biased.optical.surroundings()}

        assert "n_pad" in names
        assert "p_pad" in names


class TestConductorClearance:
    """What the Palace Route cannot mesh, refused rather than crashed.

    A drawn conductor is meshed as an outline with its interior left out
    of the domain (ADR 0003). When the Window cuts one, that outline runs
    along the Window's own wall and the Palace binary aborts with no
    message at all — deterministically, on this geometry.
    """

    @staticmethod
    def _metal(h, z):
        from gsim.common.stack.staircase import SurroundingRegion

        return SurroundingRegion(
            name="pad_metal", h=h, z=z, material="aluminum", layer_type="conductor"
        )

    def test_a_conductor_inside_the_window_is_fine(self, study):
        study.optical(window=(CENTER_Y - 2.0, CENTER_Y + 2.0), window_z=(-1.0, 1.0))

        study.optical._check_conductor_clearance(
            [self._metal((-20.6, -20.3), (0.22, 0.72))]
        )

    def test_a_conductor_outside_it_is_fine_too(self, study):
        study.optical(window=(CENTER_Y - 2.0, CENTER_Y + 2.0), window_z=(-1.0, 1.0))

        study.optical._check_conductor_clearance(
            [self._metal((-20.6, -20.3), (1.1, 1.8))]
        )

    def test_a_conductor_the_vertical_window_cuts_is_refused(self, study):
        study.optical(window=(CENTER_Y - 2.0, CENTER_Y + 2.0), window_z=(-1.0, 1.0))

        with pytest.raises(ValueError, match=r"window_z.*cuts through|pad_metal"):
            study.optical._check_conductor_clearance(
                [self._metal((-20.6, -20.3), (0.5, 1.5))]
            )

    def test_a_conductor_the_in_plane_window_cuts_is_refused(self, study):
        study.optical(window=(CENTER_Y - 0.5, CENTER_Y + 0.5), window_z=(-1.0, 1.0))

        with pytest.raises(ValueError, match="pad_metal"):
            study.optical._check_conductor_clearance(
                [self._metal((-21.0, -20.3), (0.22, 0.72))]
            )

    def test_a_dielectric_the_window_cuts_is_not_its_business(self, study):
        from gsim.common.stack.staircase import SurroundingRegion

        study.optical(window=(CENTER_Y - 2.0, CENTER_Y + 2.0), window_z=(-1.0, 1.0))

        study.optical._check_conductor_clearance(
            [
                SurroundingRegion(
                    name="slab", h=(-30.0, -10.0), z=(0.0, 0.09), material="si"
                )
            ]
        )


class TestBoundaryCondition:
    """Both Routes have to read the drawn metal the same way."""

    def test_the_domain_boundary_is_metallic_by_default(self, study):
        assert study.optical.metallic_boundaries is True
        assert study.optical.simulation().metallic_boundaries is True

    def test_turning_it_off_reaches_the_simulation(self, study):
        study.optical(metallic_boundaries=False)

        assert study.optical.simulation().metallic_boundaries is False


class TestStaircaseWavelength:
    """The Staircase and the continuous profile solve the one problem.

    Both paths turn the same carriers into a complex permittivity, and
    the only wavelength either may use for that is the one the Stage is
    solving at. The plasma-dispersion model's own wavelength is where its
    coefficients were fitted; using it instead inflates every Strip's
    free-carrier absorption whenever the Stage solves somewhere else.
    """

    @staticmethod
    def _strips(study, *, wavelength_um: float):
        """The Strip response of a single-Bias Staircase at a wavelength."""
        import numpy as np

        from gsim.tcad.results import BiasPoint, BiasSweepResult, CarrierMap

        h = np.linspace(JUNCTION_Y - HALF_WIDTH, JUNCTION_Y + HALF_WIDTH, 41)
        z = np.linspace(0.0, RIB_HEIGHT, 5)
        hh, zz = (a.ravel() for a in np.meshgrid(h, z))
        n_side = hh < JUNCTION_Y
        study.charge.seed(
            BiasSweepResult(
                contact="cathode",
                points=[
                    BiasPoint(
                        bias_v=0.0,
                        carriers=CarrierMap(
                            x_um=hh,
                            y_um=zz,
                            region=["n_rib" if side else "p_rib" for side in n_side],
                            electrons_cm3=np.where(n_side, 1e18, 1e10),
                            holes_cm3=np.where(n_side, 1e10, 1e18),
                        ),
                    )
                ],
            )
        )
        study.optical(route="femwell", n_strips=4, wavelength_um=wavelength_um)
        return study.optical.staircase(study.carriers.run().points[0]).strips

    def test_the_strips_take_the_extinction_of_the_solve_wavelength(self, study):
        """What the Staircase carries is what the continuous path builds."""
        from gsim.common.carriers import permittivity_perturbation

        wavelength_um = 1.31
        assert study.carriers.dispersion.wavelength_um != wavelength_um
        strips = self._strips(study, wavelength_um=wavelength_um)

        for i, eps in enumerate(strips["eps_complex"]):
            expected = permittivity_perturbation(
                n0=study.optical.unperturbed_index(),
                dn=float(strips["dn"][i]),
                dalpha_cm=float(strips["dalpha_cm"][i]),
                wavelength_um=wavelength_um,
            )
            assert eps == pytest.approx(expected)

    def test_moving_the_solve_wavelength_moves_the_strip_loss(self, study):
        """And it is the solve wavelength that moves it, not the fit."""
        at_fit = self._strips(study, wavelength_um=1.55)["eps_complex"]
        at_solve = self._strips(study, wavelength_um=1.31)["eps_complex"]

        for fitted, solved in zip(at_fit, at_solve, strict=True):
            assert solved.imag == pytest.approx(fitted.imag * 1.31 / 1.55)
