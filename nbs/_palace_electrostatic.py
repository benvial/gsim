# ---
# jupyter:
#   jupytext:
#     jupytext_version: 1.19.2
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Electrostatic Capacitance Extraction with Palace
#
# This notebook demonstrates electrostatic simulation using `gsim.palace.ElectrostaticSim` to extract both the ground-referenced capacitance matrix (C) and the mutual capacitance matrix (Cm) between conductor terminals.
#
# We use the IHP `cmim` (MIM capacitor) cell to illustrate the two-terminal mutual capacitance approach. Instead of computing capacitance to ground (single terminal), we assign both plates as terminals and extract the direct mutual capacitance Cm[1,2] between Metal5 (bottom) and TopMetal1 (top). The cmim cell connects the plates through a 10x10 array of Vmim vias across a thin MIM dielectric (0.19 um SiO2).
#
# **Capacitance conventions:** Palace outputs two matrices:
#   - `terminal-C.csv`: Maxwell capacitance matrix C[i,j]. Self-terms C[i,i] include coupling to ground; off-diagonal C[i,j] are negative coupling between terminals.
#   - `terminal-Cm.csv`: Mutual capacitance matrix. Cm[i,j] = -C[i,j] for i != j (direct coupling), Cm[i,i] = sum_j C[i,j] (total capacitance seen from terminal i).
#
# **Limitation:** The current terminal system assigns one terminal per layer. Structures with multiple electrodes on the same layer (e.g., interdigitated capacitors) are not yet supported.
#
# **Requirements:**
# - IHP PDK: `uv pip install ihp-gdsfactory`
# - [GDSFactory+](https://gdsfactory.com) account for cloud simulation

# %% [markdown]
# ### Load IHP MIM capacitor

# %%
from ihp import PDK, cells

PDK.activate()

cap_width = 10.0  # um
cap_length = 10.0  # um

# IHP cmim: Metal5 (bottom plate, MINUS) -> MIM dielectric (0.19 um SiO2)
#   -> 10x10 Vmim vias (0.42 um, pitch 0.94 um) -> TopMetal1 (top plate, PLUS)
c = cells.cmim(width=cap_width, length=cap_length).copy()
print("Ports:", [(p.name, tuple(p.center)) for p in c.ports])

cc = c.copy()
cc.draw_ports()
cc

# %% [markdown]
# ### Configure ElectrostaticSim

# %%
from gsim.palace import ElectrostaticSim

sim = ElectrostaticSim()

sim.set_output_dir("./palace-sim-electrostatic")
sim.set_geometry(c)
sim.set_stack(substrate_thickness=2.0)
sim.set_airbox(margin_x=5, margin_y=5, z_above=5, z_below=5)

# Metal5 = bottom plate (MINUS), TopMetal1 = top plate (PLUS)
sim.add_terminal("T1", layer="metal5")
sim.add_terminal("T2", layer="topmetal1")

sim.set_electrostatic(save_fields=1)

print(sim.validate_config())

# %% [markdown]
# ### Mesh and generate config

# %%
# Vmim vias are 0.42 um with 0.52 um gaps; MIM dielectric is 0.19 um thick
sim.mesh(preset="fine", refined_mesh_size=0.1, merge_via_distance=0)

# %%
# sim.plot_mesh(show_groups=["metal", "topmetal", "via", "dielectric", "SiO2__vmim"])

# sim.plot_mesh(show_groups=["metal5", "topmetal1", "vmim", "SiO2__vmim"])

sim.plot_mesh(
    style="solid",
    transparent_groups=["air__None", "sio2__None", "air__sio2", "air__passive"],
    interactive=True,
)

# %% [markdown]
# ### Analytical estimate
#
# For the MIM capacitor: C = epsilon_0 * epsilon_r * A / d
#
# In this estimate, epsilon_r is read from the active PDK-derived stack material table for SiO2, the MIM drawing dimensions are read from the geometry `MIMdrawing` polygon bounding box, and we report two spacings:
# - d_topmetal1 = topmetal1_bottom - metal5_top
# - d_vmim = vmim_bottom - metal5_top

# %%
import scipy.constants as const

# Pull SiO2 epsilon_r and plate spacings from the active stack derived from the current PDK.
stack = sim._resolve_stack()
sio2_props = stack.materials.get("sio2") or stack.materials.get("SiO2")
if sio2_props is None or sio2_props.get("permittivity") is None:
    raise ValueError("Could not find SiO2 permittivity in active stack materials")
eps_r = float(sio2_props["permittivity"])

metal5 = stack.layers.get("metal5")
topmetal1 = stack.layers.get("topmetal1")
vmim = stack.layers.get("vmim")
if metal5 is None or topmetal1 is None or vmim is None:
    raise ValueError("Could not find metal5/topmetal1/vmim in active stack layers")

metal5_top = metal5.zmin + metal5.thickness
topmetal1_bottom = topmetal1.zmin
vmim_bottom = vmim.zmin

d_topmetal1_um = topmetal1_bottom - metal5_top
d_vmim_um = vmim_bottom - metal5_top
if d_topmetal1_um <= 0:
    raise ValueError(f"Non-physical topmetal1 spacing: {d_topmetal1_um} um")
if d_vmim_um <= 0:
    raise ValueError(f"Non-physical vmim spacing: {d_vmim_um} um")

d_topmetal1 = d_topmetal1_um * 1e-6  # m
d_vmim = d_vmim_um * 1e-6  # m

# Read MIM drawing dimensions from geometry.
geom_component = sim.geometry.component
mim_polys = geom_component.get_polygons(by="name").get("MIMdrawing", [])
if not mim_polys:
    raise ValueError("Could not find MIMdrawing polygons in geometry")

dbu = geom_component.kcl.dbu
left = min(poly.bbox().left for poly in mim_polys)
right = max(poly.bbox().right for poly in mim_polys)
bottom = min(poly.bbox().bottom for poly in mim_polys)
top = max(poly.bbox().top for poly in mim_polys)

mim_width_um = (right - left) * dbu
mim_length_um = (top - bottom) * dbu
A_mim = (mim_width_um * 1e-6) * (mim_length_um * 1e-6)  # m^2

C_analytical_topmetal1 = const.epsilon_0 * eps_r * A_mim / d_topmetal1
C_analytical_vmim = const.epsilon_0 * eps_r * A_mim / d_vmim
C_analytical = C_analytical_topmetal1

print(f"eps_r(SiO2) = {eps_r}")
print(f"MIM drawing from geometry: {mim_width_um:.3f} x {mim_length_um:.3f} um")
print(
    f"Analytical (d=topmetal1_bottom-metal5_top={d_topmetal1_um:.3f} um): {C_analytical_topmetal1 * 1e15:.1f} fF"
)
print(
    f"Analytical (d=vmim_bottom-metal5_top={d_vmim_um:.3f} um):      {C_analytical_vmim * 1e15:.1f} fF"
)

# %% [markdown]
# ### Run on cloud
#
# Uncomment to submit to GDSFactory+ cloud. The result should be a capacitance matrix CSV.

# %%
results = sim.run()

# %% [markdown]
# ### Load and analyze results

# %%
import csv
from pathlib import Path

import numpy as np

results_dir = Path(results["terminal-C.csv"]).parent


def read_palace_csv(path):
    """Read a Palace output CSV, returning header and data as numpy array."""
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        data = np.array([[float(x) for x in row] for row in reader])
    return [h.strip() for h in header], data


# Ground-referenced capacitance matrix (C): each terminal is excited in sequence
# while all others are grounded. C[i,i] = self-capacitance to ground,
# C[i,j] = coupling between terminals i and j (negative).
header, C_matrix = read_palace_csv(results_dir / "terminal-C.csv")
print("Ground-referenced capacitance matrix C (F):")
print(f"  C[1,1] = {C_matrix[0, 1]:+.4e} F  ({C_matrix[0, 1] * 1e15:+.3f} fF)")
print(f"  C[1,2] = {C_matrix[0, 2]:+.4e} F  ({C_matrix[0, 2] * 1e15:+.3f} fF)")
print(f"  C[2,1] = {C_matrix[1, 1]:+.4e} F  ({C_matrix[1, 1] * 1e15:+.3f} fF)")
print(f"  C[2,2] = {C_matrix[1, 2]:+.4e} F  ({C_matrix[1, 2] * 1e15:+.3f} fF)")

# Mutual capacitance matrix (Cm): derived from C.
# Cm[i,j] = -C[i,j] for i != j (direct coupling, positive for parallel plates).
# Cm[i,i] = sum_j C[i,j] (total capacitance from terminal i to everything else).
_, Cm = read_palace_csv(results_dir / "terminal-Cm.csv")
Cm_12 = abs(Cm[0, 2])
print(f"\nMutual capacitance Cm[1,2] = {Cm_12 * 1e15:.3f} fF")
print(
    f"  (Cm[i,j] = -C[i,j] for i != j, verified: Cm[1,2] vs -C[1,2] = {-C_matrix[0, 2] * 1e15:.3f} fF)"
)

# Domain energy
_, E = read_palace_csv(results_dir / "domain-E.csv")
print(
    f"\nStored electric energy: {E[0, 1]:.4e} J (excitation 1), {E[1, 1]:.4e} J (excitation 2)"
)

# %% [markdown]
# ### Compare mutual capacitance with analytical estimate
#
# The mutual capacitance Cm[1,2] between the two cmim plates is the primary
# quantity of interest — it represents the direct coupling between the
# Metal5 and TopMetal1 plates, without ground reference.
#
# We compare against the parallel-plate formula:
#   C = epsilon_0 * epsilon_r * A / d
# using both the topmetal1 and vmim plate spacings from the stack.

# %%
import scipy.constants as const

# Use mutual capacitance Cm[1,2] — the direct coupling between the two plates.
C_palace = Cm_12  # mutual capacitance, already computed above

stack = sim._resolve_stack()
sio2_props = stack.materials.get("sio2") or stack.materials.get("SiO2")
if sio2_props is None or sio2_props.get("permittivity") is None:
    raise ValueError("Could not find SiO2 permittivity in active stack materials")
eps_r = float(sio2_props["permittivity"])

metal5 = stack.layers.get("metal5")
topmetal1 = stack.layers.get("topmetal1")
vmim = stack.layers.get("vmim")
if metal5 is None or topmetal1 is None or vmim is None:
    raise ValueError("Could not find metal5/topmetal1/vmim in active stack layers")

metal5_top = metal5.zmin + metal5.thickness
topmetal1_bottom = topmetal1.zmin
vmim_bottom = vmim.zmin

d_topmetal1_um = topmetal1_bottom - metal5_top
d_vmim_um = vmim_bottom - metal5_top
if d_topmetal1_um <= 0:
    raise ValueError(f"Non-physical topmetal1 spacing: {d_topmetal1_um} um")
if d_vmim_um <= 0:
    raise ValueError(f"Non-physical vmim spacing: {d_vmim_um} um")

d_topmetal1 = d_topmetal1_um * 1e-6
d_vmim = d_vmim_um * 1e-6

geom_component = sim.geometry.component
mim_polys = geom_component.get_polygons(by="name").get("MIMdrawing", [])
if not mim_polys:
    raise ValueError("Could not find MIMdrawing polygons in geometry")

dbu = geom_component.kcl.dbu
left = min(poly.bbox().left for poly in mim_polys)
right = max(poly.bbox().right for poly in mim_polys)
bottom = min(poly.bbox().bottom for poly in mim_polys)
top = max(poly.bbox().top for poly in mim_polys)

mim_width_um = (right - left) * dbu
mim_length_um = (top - bottom) * dbu
A_mim = (mim_width_um * 1e-6) * (mim_length_um * 1e-6)

C_analytical_topmetal1 = const.epsilon_0 * eps_r * A_mim / d_topmetal1
C_analytical_vmim = const.epsilon_0 * eps_r * A_mim / d_vmim
C_analytical = C_analytical_topmetal1

print(f"Mutual capacitance Cm[1,2] (Palace):           {C_palace * 1e15:.3f} fF")
print(
    f"Analytical (d=topmetal1_bottom-metal5_top):   {C_analytical_topmetal1 * 1e15:.1f} fF"
)
print(
    f"Analytical (d=vmim_bottom-metal5_top):        {C_analytical_vmim * 1e15:.1f} fF"
)
print(
    f"Ratio Cm / topmetal1-based analytical:        {C_palace / C_analytical_topmetal1:.3f}"
)
print(
    f"Ratio Cm / vmim-based analytical:             {C_palace / C_analytical_vmim:.3f}"
)

# For reference: ground-referenced |C[1,2]| should approximately equal Cm[1,2].
C_ground_val = abs(C_matrix[0, 1])
print(f"\nGround-referenced |C[1,2]| (for comparison):  {C_ground_val * 1e15:.3f} fF")
print("  (|C[1,2]| ~= Cm[1,2] for symmetric two-terminal systems)")

print(f"\nUsing A_mim = {mim_width_um:.3f} x {mim_length_um:.3f} um^2 from MIMdrawing")
print(f"d_topmetal1 = {d_topmetal1_um:.3f} um, d_vmim = {d_vmim_um:.3f} um")

# %% [markdown]
# ### Visualize |E| field cross-section
#
# We slice through the capacitor center (XZ plane, normal to Y at origin)
# to visualize the electric field magnitude. This reveals:
# - Dense uniform |E| in the MIM dielectric gap -> the "ideal" parallel-plate region
# - Fringing at plate edges -> contributes extra capacitance beyond analytical
# - Non-zero |E| above TopMetal1 in passivation/air -> parasitic coupling to ground
#
# Electrostatic fields are real-valued (DC), so we read the ``E_real`` array
# from Palace's ParaView output. The plot function computes
# ``|E| = sqrt(|E_x|^2 + |E_y|^2 + |E_z|^2)`` in the slice plane.
#
# .. note::
#    Requires the simulation to be re-run with ``save_fields=1``
#    (set above in ``set_electrostatic``). If you ran without it,
#    re-execute the ``sim.set_electrostatic(save_fields=1)`` cell
#    and then ``sim.run()`` to regenerate the ParaView output.

# %%
from pathlib import Path

from gsim.palace.field_viz import plot_fields_2d

if list(
    Path(results["terminal-C.csv"]).parent.rglob("paraview/electrostatic/**/*.pvtu")
):
    plot_fields_2d(
        results,
        field="E_real",
        normal="y",
        origin=0.0,
        grid_resolution=(360, 240),
        cmap="hot",
        title="|E| magnitude — XZ cross-section at y=0 (capacitor center)",
        figsize=(10, 5),
        streamplot_linewidth=0.6,
        streamplot_color="white",
        streamplot_density=1.0,
        streamplot_minlength=0.4,
        streamplot_maxlength=3.0,
        use_targeted_gap_seeds=True,
        targeted_seed_offset=3.0,
    )
else:
    print(
        "No ParaView field output found. "
        "Re-run the simulation with save_fields=1 (set above in set_electrostatic) "
        "to generate the |E| cross-section plot."
    )

# %%
