"""The line Stage: the electrode, its terminations, and the device report.

Every Stage before this one answers about a cross-section; this one adds
the two things that are not cross-section physics — how long the
Traveling-wave electrode is, and what it is driven from and terminated
into — and combines them with the optical and RF results into the whole
device answer.

The combination itself is not re-derived here. It is
:func:`~gsim.common.twmzm_report.twmzm_figures_of_merit`, the same entry
point the hand-assembled workflow calls, so the Stage's job is to feed it
the Study's results and hand back its
:class:`~gsim.common.twmzm_report.TWMZMReport`: the EO response and its
3 dB bandwidth, the Velocity mismatch and the Walk-off bandwidth, the
RLGC line parameters, and the Modulation efficiency along the Bias sweep.

Two numbers the solvers do not produce are configured here. The optical
group index is not a single-wavelength solve's output, so it is a setting;
without one the Stage stands the phase index in and says so. And the EO
bandwidth is read off the frequency axis, which the RF Stage samples only
where it can afford to solve, so a denser response grid can be asked for
and the line parameters are interpolated onto it.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from pydantic import Field, field_validator

from gsim.modulator.stage import Stage

if TYPE_CHECKING:
    from gsim.common.twmzm_report import (
        OpticalPhaseSweep,
        RFLineParams,
        TWMZMReport,
    )
    from gsim.modulator.optical import OpticalSweep

__all__ = ["LineStage"]


class LineStage(Stage):
    """The whole-device figures of merit of the Traveling-wave modulator.

    Attributes:
        length_um: Length of the Traveling-wave electrode (um). The
            figures of merit are length-dependent — walk-off and RF loss
            both accumulate along it — so this is the setting that turns
            a Cross-section into a device.
        z_load_ohm: Termination impedance the line is loaded with (ohm);
            complex values are accepted.
        z_gen_ohm: Generator impedance the line is driven from (ohm);
            complex values are accepted.
        n_group: Optical group index of the Phase shifter, which sets the
            Velocity mismatch against the RF index. A solve at one
            wavelength cannot produce it, so it is configured; when unset
            the optical Mode's phase index at the reference bias stands
            in, and the Stage warns that it did.
        response_frequencies_hz: Frequencies the EO response is reported
            on (Hz), kept ascending. The RF Stage's own frequencies when
            unset; a denser grid interpolates the RF line parameters onto
            it, which is what makes the 3 dB bandwidth readable off a
            handful of solved frequencies. The grid is not extrapolated:
            beyond the solved range the end values hold, and the Stage
            warns.
    """

    stage_name: ClassVar[str] = "line"

    length_um: float = Field(default=3000.0, gt=0.0)
    z_load_ohm: complex = 50.0 + 0.0j
    z_gen_ohm: complex = 50.0 + 0.0j
    n_group: float | None = Field(default=None, gt=0.0)
    response_frequencies_hz: list[float] | None = Field(default=None, min_length=1)

    @field_validator("z_load_ohm", "z_gen_ohm", mode="after")
    @classmethod
    def _as_complex(cls, value: complex) -> complex:
        """Keep impedances complex, so a real one still serializes as one."""
        return complex(value)

    @field_validator("response_frequencies_hz")
    @classmethod
    def _ascending_and_positive(cls, value: list[float] | None) -> list[float] | None:
        """Response frequencies are positive, and travel ascending."""
        if value is None:
            return None
        if any(freq <= 0.0 for freq in value):
            raise ValueError("Response frequencies must be positive.")
        return sorted(float(freq) for freq in value)

    # ------------------------------------------------------------------
    # Derivation
    # ------------------------------------------------------------------

    @property
    def length_m(self) -> float:
        """Electrode length in meters, as the analysis functions take it."""
        return float(self.length_um) * 1e-6

    def group_index(self, sweep: OpticalSweep) -> float:
        """Optical group index the Velocity mismatch is measured against.

        Args:
            sweep: The optical Stage's result.

        Returns:
            The configured group index, or the phase index at the
            reference bias when none is configured.
        """
        if self.n_group is not None:
            return float(self.n_group)
        phase_index = self._reference_phase_index(sweep)
        warnings.warn(
            f"The {self.stage_name} stage has no optical group index, so it "
            f"is standing the phase index Re(n_eff) = {phase_index:.4f} at "
            "the reference bias in for it. Velocity mismatch and the "
            "walk-off bandwidth are only as good as that substitution; set "
            f"the group index with study.{self.stage_name}(n_group=...).",
            stacklevel=2,
        )
        return phase_index

    @staticmethod
    def _reference_phase_index(sweep: OpticalSweep) -> float:
        """``Re(n_eff)`` of the Mode solved at the sweep's reference bias."""
        for point in sweep.points:
            if point.bias_v == sweep.reference_bias_v:
                return float(point.n_eff.real)
        return float(sweep.points[0].n_eff.real)

    def optical_sweep(self, sweep: OpticalSweep) -> OpticalPhaseSweep:
        """The optical Stage's result, as the report's optical input.

        Modulation efficiency is the slope of the index shift against
        bias, so the Bias points are ordered by voltage here rather than
        left in the order the charge Stage happened to visit them in.

        Args:
            sweep: The optical Stage's result.

        Returns:
            The bias sweep of index shift and loss the figures of merit
            are computed from.

        Raises:
            ValueError: When the sweep holds fewer than two Bias points,
                so the index shift has no slope, or when it visits one
                bias twice, so the slope there is undefined.
        """
        from gsim.common.twmzm_report import OpticalPhaseSweep

        if len(sweep.points) < 2:
            raise ValueError(
                f"The {self.stage_name} stage needs at least two bias points "
                "to differentiate the index shift, and the sweep has "
                f"{len(sweep.points)}. Widen it with "
                "study.charge(biases=[...])."
            )
        voltages = sweep.voltages
        order = np.argsort(voltages, kind="stable")
        voltages = voltages[order]
        repeated = np.unique(voltages[:-1][np.diff(voltages) == 0.0])
        if repeated.size:
            raise ValueError(
                f"The bias sweep visits {', '.join(f'{v:g}' for v in repeated)} V "
                "twice, so the index shift has no slope there and modulation "
                "efficiency is undefined. Sweep each bias once with "
                "study.charge(biases=[...])."
            )
        return OpticalPhaseSweep(
            voltages_v=voltages,
            dn_eff=sweep.index_shift[order],
            alpha_opt_db_cm=sweep.loss_db_cm[order],
            wavelength_um=sweep.wavelength_um,
            n_group=self.group_index(sweep),
        )

    def line_params(self, solved: RFLineParams) -> RFLineParams:
        """The RF Stage's result on the frequency grid the report uses.

        Args:
            solved: The RF Stage's result, on the frequencies it solved.

        Returns:
            The same parameters when no response grid is configured, and
            their linear interpolation onto that grid otherwise.
        """
        from gsim.common.twmzm_report import RFLineParams

        if not np.all(np.isfinite(solved.z0_ohm)):
            warnings.warn(
                f"The rf stage reported no characteristic impedance, so the "
                f"{self.stage_name} stage's EO response and RLGC parameters "
                "are NaN and the 3 dB bandwidth is unreadable. Both routes "
                "extract Z0, so the rf stage will have said why it could "
                "not — re-run it and read its warnings.",
                stacklevel=2,
            )

        if self.response_frequencies_hz is None:
            return solved

        grid = np.asarray(self.response_frequencies_hz, dtype=np.float64)
        solved_freq = solved.freq_hz
        outside = grid[(grid < solved_freq[0]) | (grid > solved_freq[-1])]
        if outside.size:
            warnings.warn(
                f"The {self.stage_name} stage's response grid reaches "
                f"{outside.min() / 1e9:g}-{outside.max() / 1e9:g} GHz, outside "
                f"the {solved_freq[0] / 1e9:g}-{solved_freq[-1] / 1e9:g} GHz "
                "the rf stage solved; the line parameters are held at their "
                "end values there rather than extrapolated. Solve those "
                "frequencies with study.rf(frequencies_hz=[...]).",
                stacklevel=2,
            )
        return RFLineParams(
            freq_hz=grid,
            n_rf=np.interp(grid, solved_freq, solved.n_rf),
            alpha_rf_np_m=np.interp(grid, solved_freq, solved.alpha_rf_np_m),
            z0_ohm=np.asarray(
                np.interp(grid, solved_freq, solved.z0_ohm.real)
                + 1j * np.interp(grid, solved_freq, solved.z0_ohm.imag),
                dtype=np.complex128,
            ),
        )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def _solve(self) -> TWMZMReport:
        """Combine both EM Stages into the device report, running them first."""
        from gsim.common.twmzm_report import twmzm_figures_of_merit

        study = self._require_study()
        optical = self.optical_sweep(study.optical.run())
        rf = self.line_params(study.rf.run())
        return twmzm_figures_of_merit(
            rf,
            optical,
            length_m=self.length_m,
            z_load_ohm=self.z_load_ohm,
            z_gen_ohm=self.z_gen_ohm,
        )
