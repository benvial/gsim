# Shared components guide

Use shared geometry and layer-stack objects to describe components and their fabrication stack across simulation engines.
The links below open the API reference for each object or helper.

`gsim.common` itself is a narrow façade: it re-exports the geometry and
stack names of the two sections below and nothing else. Everything else in
the package — `modes`, `circuit`, `carriers`, `transmission_line`,
`interpolate`, `mesh_regions` — is imported by its submodule path, as
`from gsim.common.transmission_line import RFLineParams`, so importing
`gsim.common` never pulls a backend's dependencies in with it.

## Geometry

- [`Geometry`][gsim.common.Geometry]
- [`GeometryModel`][gsim.common.GeometryModel]
- [`Prism`][gsim.common.Prism]
- [`extract_geometry_model`][gsim.common.extract_geometry_model]

## Stack

- [`LayerStack`][gsim.common.LayerStack]
- [`Layer`][gsim.common.Layer]

## PN Junction

Depletion model after Sze & Ng, *Physics of Semiconductor Devices*, ch. 2:
the quick analytic path beside the charge-transport solve, and the
reference its numeric `C(V)` is validated against.

- [`PNJunctionConfig`][gsim.common.stack.PNJunctionConfig]
- [`built_in_voltage`][gsim.common.stack.built_in_voltage]
- [`depletion_width`][gsim.common.stack.depletion_width]
- [`depletion_extents`][gsim.common.stack.depletion_extents]
- [`junction_capacitance_per_area`][gsim.common.stack.junction_capacitance_per_area]

## Visualization

- [`plot_prisms_3d`][gsim.common.viz.plot_prisms_3d]
- [`plot_prism_slices`][gsim.common.viz.plot_prism_slices]
- [`create_web_export`][gsim.common.viz.create_web_export]
- [`export_3d_mesh`][gsim.common.viz.export_3d_mesh]

## Transmission line

The Traveling-wave electrode as a transmission line: what a mode solve says
about it, its per-unit-length circuit, how the junction's series-RC branch
loads it, and what a periodically loaded electrode behaves as. The
[TW-MZM notebook](../nbs/tcad_twmzm.md) turns these into device figures of
merit.

- [`RFLineParams`][gsim.common.transmission_line.RFLineParams]
- [`line_params_from_neff`][gsim.common.transmission_line.line_params_from_neff]
- [`line_params_from_gamma`][gsim.common.transmission_line.line_params_from_gamma]
- [`rlgc_from_line_params`][gsim.common.transmission_line.rlgc_from_line_params]
- [`JunctionBranch`][gsim.common.junction.JunctionBranch]
- [`series_rc_from_admittance`][gsim.common.junction.series_rc_from_admittance]
- [`loaded_line_params`][gsim.common.transmission_line.loaded_line_params]
- [`section_abcd`][gsim.common.transmission_line.section_abcd]
- [`segmented_period_abcd`][gsim.common.transmission_line.segmented_period_abcd]
- [`segmented_line_params`][gsim.common.transmission_line.segmented_line_params]
- [`segmented_line`][gsim.common.transmission_line.segmented_line]
- [`bragg_fraction`][gsim.common.transmission_line.bragg_fraction]

## Circuit export

The compact-model handoff to circuit simulators: the Traveling-wave
electrode as a two-port, the junction as a tabulated series-RC model
file, and the plain-numpy readers and driven-line responses that prove
the round trip. The junction model file format is specified in
`write_junction_model`.

- [`line_smatrix`][gsim.common.circuit.line_smatrix]
- [`write_touchstone`][gsim.common.circuit.write_touchstone]
- [`read_touchstone`][gsim.common.circuit.read_touchstone]
- [`sax_line_model`][gsim.common.circuit.sax_line_model]
- [`write_junction_model`][gsim.common.circuit.write_junction_model]
- [`read_junction_model`][gsim.common.circuit.read_junction_model]
- [`terminated_response`][gsim.common.circuit.terminated_response]
- [`line_driven_response`][gsim.common.circuit.line_driven_response]
