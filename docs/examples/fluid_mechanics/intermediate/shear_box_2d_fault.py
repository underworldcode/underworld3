# %% [markdown]
r"""
# Shear Box with Embedded 2D Fault (Anisotropic Rheology)

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** intermediate
**RUNTIME:** < 1 minute

## Description

A 2D shear box with an embedded weak fault zone using transverse isotropic
(anisotropic) rheology. The fault is represented as a 1D polyline embedded
in the 2D domain, with the fault normal defining the director for the
anisotropic viscosity tensor.

## Key Concepts

- 2D Surface (polyline) creation using `uw.meshing.Surface`
- Signed distance field from embedded surfaces
- **Transverse isotropic rheology** with fault-normal director
- Influence functions for smooth rheology transitions
- Unit-aware problem setup with `uw.Params`

## Physical Setup

| Parameter | Value |
|-----------|-------|
| Domain | 20 km × 10 km |
| Fault | 10 km long at $y = 5$ km |
| $\eta_0$ (strong) | $10^{21}$ Pa·s |
| $\eta_1$ (weak) | $10^{19}$ Pa·s |
| Director | Fault normal vector |
| BCs | Top: +V, Bottom: -V |
"""

# %%
#| echo: false
import nest_asyncio
nest_asyncio.apply()

# %% [markdown]
"""
## Parameters

All physical parameters defined using `uw.Params` for easy modification
and command-line override support.
"""

# %%
import os
import numpy as np
import underworld3 as uw
import sympy

# Units and scaling
u = uw.scaling.units
ndim = uw.scaling.non_dimensionalise

# Scaling system
scaling = uw.scaling.get_coefficients()
scaling["[length]"] = 10 * u.kilometer
scaling["[time]"] = 10 * u.kilometer / (1 * u.centimeter / u.year)
scaling["[mass]"] = (1e21 * u.pascal * u.second) * (10 * u.kilometer) * scaling["[time]"]

# Define all tunable parameters (units encoded in names for CLI clarity)
params = uw.Params(
    # Domain geometry
    uw_domain_x_km = 20.0,           # Domain width
    uw_domain_y_km = 10.0,           # Domain height
    uw_cell_size_km = 0.25,          # Mesh resolution

    # Fault geometry
    uw_fault_y_km = 5.0,             # Fault y-position
    uw_fault_x_min_km = 5.0,         # Fault left end
    uw_fault_x_max_km = 15.0,        # Fault right end
    uw_fault_width_km = 0.5,         # Fault influence width

    # Material properties
    uw_log10_eta_strong = 21.0,      # log10(viscosity / Pa·s)
    uw_log10_eta_weak = 17.0,        # log10(viscosity / Pa·s)

    # Boundary conditions
    uw_v_plate_cm_per_yr = 1.0,      # Plate velocity
)

# %%
# Display parameters (renders as table in Jupyter)
params

# %% [markdown]
"""
Convert parameters to physical quantities with units:
"""

# %%
# Domain geometry with units
DOMAIN_X = params.uw_domain_x_km * u.kilometer
DOMAIN_Y = params.uw_domain_y_km * u.kilometer
CELL_SIZE = params.uw_cell_size_km * u.kilometer

# Fault geometry with units
FAULT_Y = params.uw_fault_y_km * u.kilometer
FAULT_X_MIN = params.uw_fault_x_min_km * u.kilometer
FAULT_X_MAX = params.uw_fault_x_max_km * u.kilometer
FAULT_WIDTH = params.uw_fault_width_km * u.kilometer

# Material properties with units
ETA_STRONG = 10**params.uw_log10_eta_strong * u.pascal * u.second
ETA_WEAK = 10**params.uw_log10_eta_weak * u.pascal * u.second

# Boundary velocity with units
V_PLATE = params.uw_v_plate_cm_per_yr * u.centimeter / u.year

# %% [markdown]
"""
## Setup
"""

# %%
import matplotlib
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("output", exist_ok=True)

# Visualization setup
render = True
try:
    import pyvista as pv
    import underworld3.visualisation as vis
    import trame
    if uw.mpi.size > 1:
        render = False
except ImportError:
    render = False

# %% [markdown]
"""
## Mesh

Units are passed directly to the mesh constructor.
"""

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0 * u.kilometer, 0.0 * u.kilometer),
    maxCoords=(DOMAIN_X, DOMAIN_Y),
    cellSize=CELL_SIZE,
    regular=False,
    qdegree=3,
)

# %% [markdown]
"""
## Fault Surface

The fault is a 1D polyline embedded in the 2D domain. The `Surface` class
computes normals automatically and provides a signed distance field.

The `symbol` parameter gives a clean LaTeX representation in expressions.
"""

# %%
fault_points = np.array([
    [FAULT_X_MIN.magnitude, FAULT_Y.magnitude],
    [FAULT_X_MAX.magnitude, FAULT_Y.magnitude],
]) * u.kilometer

fault = uw.meshing.Surface("main_fault", mesh, fault_points, symbol="F")
fault.discretize()

# %% [markdown]
"""
The fault normal vector defines the director for the anisotropic rheology:
"""

# %%
fault.normals[:, :2]

# %% [markdown]
"""
## Visualize Distance Field
"""

# %%
if render:
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["distance"] = vis.scalar_fn_to_pv_points(pvmesh, fault.distance.sym[0])

    pl = pv.Plotter(window_size=(800, 400))
    pl.add_mesh(pvmesh, scalars="distance", cmap="RdBu", clim=[-0.5, 0.5], show_edges=False)

    fp = fault.control_points
    pl.add_mesh(pv.Line((fp[0, 0], fp[0, 1], 0), (fp[1, 0], fp[1, 1], 0)),
                color="black", line_width=4)
    pl.add_title("Signed Distance from Fault")
    pl.view_xy()
    pl.show()

# %% [markdown]
r"""
## Stokes Solver with Transverse Isotropic Rheology

The `TransverseIsotropicFlowModel` defines an anisotropic viscosity tensor:

$$\eta_{ijkl} = \eta_0 \cdot I_{ijkl} + (\eta_0 - \eta_1) \cdot A_{ijkl}(\hat{n})$$

where $\hat{n}$ is the director (fault normal) and $A_{ijkl}$ couples shear
to the weak plane orientation.

- Far from fault: $\eta_1 = \eta_0$ → isotropic behavior
- Near fault: $\eta_1 < \eta_0$ → anisotropic weakness
"""

# %%
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel

# %% [markdown]
r"""
### Configure Rheology Parameters

The director is the fault normal. The weak viscosity $\eta_1$ varies
spatially using the fault's influence function.
"""

# %%
# Director from fault normal (row vector)
fault_normal = fault.normals[0, :2]
director = sympy.Matrix([fault_normal[0], fault_normal[1]]).T

# Influence function: 1 at fault, 0 far away
fault_influence = fault.influence_function(
    width=FAULT_WIDTH,
    value_near=1.0,
    value_far=0.0,
    profile="smoothstep",
)

# Non-dimensional viscosities for expressions
eta_0_nd = ndim(ETA_STRONG)
eta_1_nd = ndim(ETA_WEAK)

# eta_1 varies: isotropic far from fault, anisotropic near fault
eta_1_expr = eta_0_nd - (eta_0_nd - eta_1_nd) * fault_influence

# Set constitutive model parameters
stokes.constitutive_model.Parameters.eta_0 = eta_0_nd
stokes.constitutive_model.Parameters.eta_1 = eta_1_expr
stokes.constitutive_model.Parameters.director = director

stokes.penalty = 0.0
stokes.saddle_preconditioner = 1.0 / eta_0_nd

# %% [markdown]
"""
The influence function expression (renders as LaTeX in Jupyter):
"""

# %%
fault_influence

# %% [markdown]
"""
## Boundary Conditions

Simple shear: top moves right, bottom moves left.
"""

# %%
V_nd = ndim(V_PLATE)
stokes.add_dirichlet_bc((V_nd, 0.0), "Top", components=(0, 1))
stokes.add_dirichlet_bc((-V_nd, 0.0), "Bottom", components=(0, 1))

# %% [markdown]
"""
## Solve
"""

# %%
stokes.solve(verbose=False)

# %% [markdown]
"""
## Project Strain Rate
"""

# %%
strain_rate = uw.discretisation.MeshVariable("SR", mesh, 1, degree=1)
sr_projection = uw.systems.Projection(mesh, strain_rate)
sr_projection.uw_function = stokes.Unknowns.Einv2
sr_projection.smoothing = 1e-3
sr_projection.solve()

# %% [markdown]
"""
## Visualize Results
"""

# %%
if render:
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)
    pvmesh.point_data["Vmag"] = np.linalg.norm(pvmesh.point_data["V"], axis=1)

    velocity_points = vis.meshVariable_to_pv_cloud(v)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v.sym)

    pl = pv.Plotter(window_size=(800, 400))
    pl.add_mesh(pvmesh, scalars="Vmag", cmap="viridis", show_edges=False)
    arrows = velocity_points.glyph(orient="V", scale=0.01, factor=0.15)
    pl.add_mesh(arrows, color="white", opacity=0.2)

    fp = fault.control_points
    pl.add_mesh(pv.Line((fp[0, 0], fp[0, 1], 0), (fp[1, 0], fp[1, 1], 0)),
                color="red", line_width=3)
    pl.add_title("Velocity Magnitude")
    pl.view_xy()
    pl.show()

# %%
if render:
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["SR"] = vis.scalar_fn_to_pv_points(pvmesh, strain_rate.sym[0])

    pl = pv.Plotter(window_size=(800, 400))
    pl.add_mesh(pvmesh, scalars="SR", cmap="hot", show_edges=True, log_scale=False)

    fp = fault.control_points
    pl.add_mesh(pv.Line((fp[0, 0], fp[0, 1], 0), (fp[1, 0], fp[1, 1], 0)),
                color="cyan", line_width=3)
    pl.add_title("Strain Rate")
    pl.view_xy()
    pl.show()

# %% [markdown]
"""
## Strain Rate Profile

Sample strain rate along a vertical profile through the domain center.
"""

# %%
# Non-dimensional values for sampling
Lx_nd = ndim(DOMAIN_X)
Ly_nd = ndim(DOMAIN_Y)
fault_y_nd = ndim(FAULT_Y)
fault_width_nd = ndim(FAULT_WIDTH)

n_samples = 100
y_profile = np.linspace(0.1, Ly_nd - 0.1, n_samples)
profile_coords = np.column_stack([np.full(n_samples, Lx_nd / 2), y_profile])
sr_profile = uw.function.evaluate(strain_rate.sym[0], profile_coords).flatten()

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(y_profile, sr_profile, 'b-', linewidth=2, label='Strain rate')
ax.axvline(x=fault_y_nd, color='r', linestyle='--', linewidth=2, label='Fault')
ax.axvspan(fault_y_nd - fault_width_nd, fault_y_nd + fault_width_nd,
           alpha=0.2, color='red', label='Fault zone')
ax.set_xlabel('y (model units)')
ax.set_ylabel('Strain rate invariant')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/strain_rate_profile.png", dpi=150)
plt.close()

# %% [markdown]
"""
## Quantify Strain Localization
"""

# %%
x_sample = np.linspace(0.1, Lx_nd - 0.1, 50)

coords_on = np.column_stack([x_sample, np.full(50, fault_y_nd)])
sr_on = uw.function.evaluate(strain_rate.sym[0], coords_on).flatten()

coords_off = np.column_stack([x_sample, np.full(50, 0.2)])
sr_off = uw.function.evaluate(strain_rate.sym[0], coords_off).flatten()

localization = sr_on.mean() / sr_off.mean()
print(f"Localization factor: {localization:.1f}×")

# %% [markdown]
"""
## Save Output
"""

# %%
fault.save("output/fault_surface.vtk")
mesh.write_timestep("shear_box_2d", meshVars=[v, p, strain_rate], outputPath="./output", index=0)

# %% [markdown]
"""
## Summary

This example demonstrated:

1. **Parameter management** using `uw.Params` for easy modification and CLI override
2. **Embedded fault geometry** using `uw.meshing.Surface` with unit-aware points
3. **Transverse isotropic rheology** with the fault normal as director
4. **Smooth transitions** via influence functions from isotropic (far) to anisotropic (near fault)
5. **Strain localization** in the anisotropic weak zone

The anisotropic rheology produces strain localization along the fault,
with the localization factor depending on the viscosity contrast and
the orientation of shear relative to the fault normal.

### Command-line Override

Parameters can be modified from the command line (units are in the name):
```bash
python shear_box_2d_fault.py -uw_log10_eta_weak 20.0 -uw_fault_width_km 1.0
```
"""
