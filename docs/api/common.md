# Common API

## Geometry

::: gsim.common.Geometry
    options:
      show_source: false
      inherited_members: false
      members: false

::: gsim.common.GeometryModel
    options:
      show_source: false
      inherited_members: false
      members: false

::: gsim.common.Prism
    options:
      show_source: false
      inherited_members: false
      members: false

::: gsim.common.extract_geometry_model
    options:
      show_source: false

## Stack

::: gsim.common.LayerStack
    options:
      show_source: false
      inherited_members: false
      members: false

::: gsim.common.Layer
    options:
      show_source: false
      inherited_members: false
      members: false

## PN Junction

Depletion model after Sze & Ng, *Physics of Semiconductor Devices*, ch. 2,
plus the 1D free-carrier plasma-dispersion model for the complex optical
permittivity and the doping-profile geometry builders.

::: gsim.common.stack.PNJunctionConfig
    options:
      show_source: false

::: gsim.common.stack.make_pn_junction_profile
    options:
      show_source: false

::: gsim.common.stack.make_segmented_junction_profile
    options:
      show_source: false

::: gsim.common.stack.make_doping_profile
    options:
      show_source: false

::: gsim.common.stack.junction_epsilon_profile
    options:
      show_source: false

::: gsim.common.stack.carrier_profile_1d
    options:
      show_source: false

::: gsim.common.stack.epsilon_eff_relative
    options:
      show_source: false

::: gsim.common.stack.optical_params
    options:
      show_source: false

::: gsim.common.stack.refractive_index
    options:
      show_source: false

::: gsim.common.stack.drude_relaxation_times
    options:
      show_source: false

::: gsim.common.stack.PNJunctionConfig
    options:
      show_source: false

::: gsim.common.stack.make_pn_junction_profile
    options:
      show_source: false

::: gsim.common.stack.built_in_voltage
    options:
      show_source: false

::: gsim.common.stack.depletion_width
    options:
      show_source: false

::: gsim.common.stack.depletion_extents
    options:
      show_source: false

::: gsim.common.stack.junction_capacitance_per_area
    options:
      show_source: false

::: gsim.common.stack.select_junction_mode
    options:
      show_source: false

## Visualization

::: gsim.common.viz.plot_prisms_3d
    options:
      show_source: false

::: gsim.common.viz.plot_prism_slices
    options:
      show_source: false

::: gsim.common.viz.create_web_export
    options:
      show_source: false

::: gsim.common.viz.export_3d_mesh
    options:
      show_source: false

## Circuit export

The compact-model handoff to circuit simulators: the Traveling-wave
electrode as a two-port, the junction as a tabulated series-RC model
file, and the plain-numpy readers and driven-line responses that prove
the round trip. The junction model file format is specified in
`write_junction_model`.

::: gsim.common.circuit.line_smatrix
    options:
      show_source: false

::: gsim.common.circuit.write_touchstone
    options:
      show_source: false

::: gsim.common.circuit.read_touchstone
    options:
      show_source: false

::: gsim.common.circuit.sax_line_model
    options:
      show_source: false

::: gsim.common.circuit.write_junction_model
    options:
      show_source: false

::: gsim.common.circuit.read_junction_model
    options:
      show_source: false

::: gsim.common.circuit.terminated_response
    options:
      show_source: false

::: gsim.common.circuit.line_driven_response
    options:
      show_source: false
