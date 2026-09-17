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
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

from gsim.modulator.stage import Stage

if TYPE_CHECKING:
    from gsim.common.circuit import SaxLineModel
    from gsim.common.twmzm_report import (
        OpticalPhaseSweep,
        RFLineParams,
        TWMZMReport,
    )
    from gsim.modulator.optical import OpticalSweep

__all__ = ["ExportRoundTrip", "LineStage"]


class ExportRoundTrip(BaseModel):
    """Proof the compact-model handoff loses nothing.

    Both exported artifacts read back from their files and replayed with
    plain network math, next to the Study's own numbers: the driven
    response of the Traveling-wave electrode — ideal generator, the
    exported two-port, a load — against the same response from the
    solved line parameters, and the junction model file's series-RC
    branch against the charge sweep's fit.

    The two responses share every input except the file: the internal
    one comes from the telegrapher ABCD of ``gamma``/``Z0``, the
    reassembled one from the Touchstone S-matrix, so what the comparison
    exercises is exactly the writers, the readers and the S-parameter
    conversion in between. The junction file carries JSON's exact float
    representation, so its columns are compared for equality, not
    tolerance.

    Attributes:
        touchstone_path: The two-port file the response was rebuilt from.
        junction_path: The junction model file that was read back.
        freq_hz: Frequencies of both responses (Hz).
        internal: ``V_load/V_gen`` from the solved line parameters.
        reassembled: ``V_load/V_gen`` from the exported two-port.
        bias_v: Biases of the junction sweep (V).
        r_s_internal_ohm_m: Series resistance from the charge Stage.
        r_s_file_ohm_m: Series resistance read back from the file.
        c_j_internal_f_per_m: Junction capacitance from the charge Stage.
        c_j_file_f_per_m: Junction capacitance read back from the file.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    touchstone_path: Path
    junction_path: Path
    freq_hz: NDArray[np.float64]
    internal: NDArray[np.complex128]
    reassembled: NDArray[np.complex128]
    bias_v: NDArray[np.float64]
    r_s_internal_ohm_m: NDArray[np.float64]
    r_s_file_ohm_m: NDArray[np.float64]
    c_j_internal_f_per_m: NDArray[np.float64]
    c_j_file_f_per_m: NDArray[np.float64]

    @property
    def response_rel_diff(self) -> NDArray[np.float64]:
        """``|reassembled - internal| / |internal|`` per frequency."""
        return np.asarray(
            np.abs(self.reassembled - self.internal) / np.abs(self.internal),
            dtype=np.float64,
        )

    def table(self) -> str:
        """The side-by-side comparison as a printable table."""
        lines = [
            "Driven response V_load/V_gen (internal vs reassembled from "
            f"{self.touchstone_path.name}):",
            f"{'f (GHz)':>10} {'|H| internal':>14} {'|H| files':>14} {'rel diff':>10}",
        ]
        lines.extend(
            f"{freq / 1e9:>10.3f} {abs(internal):>14.6e} "
            f"{abs(rebuilt):>14.6e} {diff:>10.2e}"
            for freq, internal, rebuilt, diff in zip(
                self.freq_hz,
                self.internal,
                self.reassembled,
                self.response_rel_diff,
                strict=True,
            )
        )
        lines.append(f"Junction branch (charge stage vs {self.junction_path.name}):")
        lines.append(
            f"{'bias (V)':>10} {'R_s (ohm*m)':>14} {'C_j (F/m)':>14} {'match':>10}"
        )
        lines.extend(
            f"{bias:>10.3f} {r_file:>14.6e} {c_file:>14.6e} "
            f"{'exact' if r_file == r_int and c_file == c_int else 'DIFFERS':>10}"
            for bias, r_int, r_file, c_int, c_file in zip(
                self.bias_v,
                self.r_s_internal_ohm_m,
                self.r_s_file_ohm_m,
                self.c_j_internal_f_per_m,
                self.c_j_file_f_per_m,
                strict=True,
            )
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        """The printable side-by-side table."""
        return self.table()

    def check(self, *, response_tol: float = 1e-8) -> ExportRoundTrip:
        """Fail loudly where an export does not round-trip.

        The driven responses agree to ``response_tol`` — the Touchstone
        file quantizes to 12 significant digits, so the default leaves
        three orders of margin above that floor and anything past it is
        a real defect, not formatting. The junction columns must match
        exactly.

        Args:
            response_tol: Largest relative driven-response difference
                accepted at any solved frequency.

        Returns:
            This comparison, so the call chains.

        Raises:
            ValueError: Naming the diverging quantity — and where it
                diverges — when a round-trip fails, or when the
                responses are not finite and agreement cannot be
                verified at all.
        """
        diff = self.response_rel_diff
        finite = np.isfinite(diff)
        if not np.all(finite):
            # NaN compares False against any tolerance, so a NaN response
            # (the rf stage reported no impedance, say) must fail here
            # rather than slip through as agreement.
            where = int(np.argmax(~finite))
            raise ValueError(
                "The driven response comparison is not finite at "
                f"{self.freq_hz[where] / 1e9:g} GHz (internal "
                f"{self.internal[where]}, reassembled "
                f"{self.reassembled[where]}), so the round trip cannot be "
                "verified. The rf stage will have said why its line "
                "parameters are unusable — re-run it and read its warnings."
            )
        worst = int(np.argmax(diff))
        if diff[worst] > response_tol:
            raise ValueError(
                "The driven response reassembled from "
                f"{self.touchstone_path.name} differs from the line stage's "
                f"by {diff[worst]:.3e} (relative) at "
                f"{self.freq_hz[worst] / 1e9:g} GHz, above the "
                f"{response_tol:g} tolerance."
            )
        for name, internal, read_back in (
            ("R_s", self.r_s_internal_ohm_m, self.r_s_file_ohm_m),
            ("C_j", self.c_j_internal_f_per_m, self.c_j_file_f_per_m),
        ):
            mismatch = np.nonzero(internal != read_back)[0]
            if mismatch.size:
                where = int(mismatch[0])
                raise ValueError(
                    f"The junction model's {name} read back from "
                    f"{self.junction_path.name} differs from the charge "
                    f"stage's at {self.bias_v[where]:g} V: "
                    f"{read_back[where]!r} != {internal[where]!r}."
                )
        return self


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
        return solved.resampled(grid)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _solved_line(self) -> RFLineParams:
        """The RF Stage's result, running it first when it has not run.

        Exports carry the solved frequencies rather than the response
        grid: the grid is this Stage's own interpolation, and the
        consumer of an export interpolates for itself.

        Returns:
            The RF line parameters on the solved frequencies.
        """
        rf: RFLineParams = self._require_study().rf.run()
        return rf

    @staticmethod
    def _export_provenance(rf: RFLineParams, *, length_m: float) -> list[str]:
        """Comment lines recording what the exported two-port is.

        Read off the RF result itself: the record carries the Contact
        its impedance was read over and the Bias it was solved at.
        """
        lines = [f"length_m = {length_m:g}"]
        if rf.signal_contact is not None:
            lines.append(f"signal contact: {rf.signal_contact}")
        if rf.bias_v is not None:
            lines.append(f"bias_v = {rf.bias_v:g}")
        return lines

    def export_touchstone(
        self, path: str | Path | None = None, *, z_ref_ohm: float = 50.0
    ) -> Path:
        """Write the Traveling-wave electrode as a Touchstone two-port.

        The solved ``gamma(f)`` and ``Z0(f)`` and this Stage's length
        become the uniform line's S-matrix on the RF Stage's solved
        frequencies, written as a ``.s2p`` file any circuit simulator
        reads — gsim's half of the compact-model handoff to circulax.
        Runs the RF Stage first when it holds no result.

        Args:
            path: Output file; ``electrode.s2p`` in the line Stage's
                output directory when omitted.
            z_ref_ohm: Port reference impedance of the S-parameters
                (ohm, real — the Touchstone convention).

        Returns:
            The written path.
        """
        from gsim.common.circuit import line_smatrix, write_touchstone

        rf = self._solved_line()
        target = (
            Path(path)
            if path is not None
            else self._require_study().stage_dir(self.stage_name) / "electrode.s2p"
        )
        return write_touchstone(
            target,
            freq_hz=rf.freq_hz,
            s=line_smatrix(
                rf.gamma_per_m,
                rf.z0_ohm,
                length_m=self.length_m,
                z_ref_ohm=z_ref_ohm,
            ),
            z_ref_ohm=z_ref_ohm,
            comments=self._export_provenance(rf, length_m=self.length_m),
        )

    def sax_model(self, *, z_ref_ohm: complex = 50.0) -> SaxLineModel:
        """The Traveling-wave electrode as a SAX-convention callable.

        A plain function over numpy arrays — no sax import anywhere —
        returning the dict of S-matrix entries at the frequencies it is
        called with (the solved ones by default), for circulax or any
        sdict consumer to compose into a circuit. Runs the RF Stage
        first when it holds no result.

        Args:
            z_ref_ohm: Port reference impedance of the S-parameters
                (ohm).

        Returns:
            The model callable.
        """
        from gsim.common.circuit import sax_line_model

        rf = self._solved_line()
        return sax_line_model(
            rf.freq_hz,
            rf.gamma_per_m,
            rf.z0_ohm,
            length_m=self.length_m,
            z_ref_ohm=z_ref_ohm,
        )

    def driven_response(self) -> NDArray[np.complex128]:
        """``V_load / V_gen`` of the terminated electrode, per frequency.

        The electrical response of the Traveling-wave electrode between
        this Stage's generator and load, from the solved line parameters
        via the telegrapher ABCD matrix, on the RF Stage's solved
        frequencies. Runs the RF Stage first when it holds no result.

        Returns:
            The complex voltage transfer per solved frequency.
        """
        from gsim.common.circuit import line_driven_response

        rf = self._solved_line()
        return line_driven_response(
            rf.gamma_per_m,
            rf.z0_ohm,
            length_m=self.length_m,
            z_gen_ohm=self.z_gen_ohm,
            z_load_ohm=self.z_load_ohm,
        )

    def verify_exports(
        self,
        *,
        touchstone_path: str | Path | None = None,
        junction_path: str | Path | None = None,
        z_ref_ohm: float = 50.0,
        quiet: bool = False,
    ) -> ExportRoundTrip:
        """Prove the exported compact models round-trip losslessly.

        Writes both handoff artifacts — the electrode's Touchstone
        two-port and the junction model file — reads them back with the
        plain stdlib/numpy readers a consumer would use, reassembles the
        driven line response from the files (ideal generator, the
        two-port, this Stage's load; no circulax and no sax anywhere),
        and puts it side by side with the same response from the solved
        line parameters. Prints the comparison table and returns it;
        gate on it with :meth:`ExportRoundTrip.check`.

        Args:
            touchstone_path: Where to write the two-port; the line
                Stage's directory when omitted.
            junction_path: Where to write the junction model; the charge
                Stage's directory when omitted.
            z_ref_ohm: Port reference impedance of the exported
                S-parameters (ohm, real).
            quiet: Skip printing the table.

        Returns:
            The side-by-side comparison.
        """
        from gsim.common.circuit import (
            read_junction_model,
            read_touchstone,
            terminated_response,
        )

        study = self._require_study()
        touchstone = self.export_touchstone(touchstone_path, z_ref_ohm=z_ref_ohm)
        junction = study.charge.export_junction_model(junction_path)

        two_port = read_touchstone(touchstone)
        model = read_junction_model(junction)
        # The RF Stage owns the reading of the charge sweep's admittance
        # into a shunt branch; the line Stage reads the EM Stages only.
        bias_v, branch = study.rf.junction_branches()

        comparison = ExportRoundTrip(
            touchstone_path=touchstone,
            junction_path=junction,
            freq_hz=two_port.freq_hz,
            internal=self.driven_response(),
            reassembled=terminated_response(
                two_port.s,
                z_ref_ohm=two_port.z_ref_ohm,
                z_gen_ohm=self.z_gen_ohm,
                z_load_ohm=self.z_load_ohm,
            ),
            bias_v=bias_v,
            r_s_internal_ohm_m=np.asarray(branch.r_s_ohm_m, dtype=np.float64),
            r_s_file_ohm_m=model.r_s_ohm_m,
            c_j_internal_f_per_m=np.asarray(branch.c_j_f_per_m, dtype=np.float64),
            c_j_file_f_per_m=model.c_j_f_per_m,
        )
        if not quiet:
            print(comparison.table())  # noqa: T201
        return comparison

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
