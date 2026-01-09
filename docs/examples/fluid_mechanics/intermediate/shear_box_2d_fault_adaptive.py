# %% [markdown]
r"""
# Adaptive Shear Box with Embedded 2D Fault

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** intermediate
**RUNTIME:** < 2 minutes (requires AMR environment)

## Description

A 2D shear box with an embedded weak fault zone, demonstrating **mesh adaptation**
to refine resolution near the fault while keeping the total element count manageable.

This example builds on `shear_box_2d_fault.py` by adding:
- Metric-based mesh adaptation using `mesh.adapt()`
- Surface-based refinement criteria via `fault.refinement_metric()`
- Strategy for ~10× refinement near the fault with constant element count

## Key Concepts

- **Mesh adaptation** using MMG via PETSc's `adaptMetric`
- **Refinement metrics** based on distance from embedded surfaces
- **Element count control**: balancing refinement and coarsening
- Variables automatically reset after adaptation (reinitialization needed)

## Adaptation Strategy

To achieve ~10× refinement near the fault while keeping element count constant:

1. **Start coarser**: Use 2× larger initial cell size than the uniform mesh
2. **Define metric**: `h_near = h_initial / 5`, `h_far = h_initial`
3. **Result**: 10× finer at fault (vs original), same density far away
4. **Element balance**: Small refined region + large coarse region ≈ original count

| Region | Element Size | Relative to Original |
|--------|--------------|---------------------|
| Far from fault | 0.5 km | 2× coarser |
| Near fault | 0.05 km | 5× finer |
| **Net effect at fault** | | **10× finer than original** |
"""

# %%
#| echo: false
import nest_asyncio
nest_asyncio.apply()

# %% [markdown]
"""
## Parameters

Same physical setup as the non-adaptive version, but with coarser initial mesh.
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

# Define all tunable parameters
params = uw.Params(
    # Domain geometry
    uw_domain_x_km = 20.0,
    uw_domain_y_km = 10.0,
    uw_cell_size_km = 0.5,           # COARSER initial mesh (2× original)

    # Fault geometry
    uw_fault_y_km = 5.0,
    uw_fault_x_min_km = 5.0,
    uw_fault_x_max_km = 15.0,
    uw_fault_width_km = 0.5,         # Rheology transition width

    # Adaptation parameters (empirically tuned for ~same element count)
    # h_near/h_far ratio determines refinement ratio
    # Larger h_far allows coarsening far from fault to compensate for refinement
    uw_adapt_h_near_km = 0.05,       # ~5× finer than initial cellSize
    uw_adapt_h_far_km = 0.9,         # ~2× coarser than initial (compensates)
    uw_adapt_width_km = 1.5,         # Metric transition width

    # Material properties
    uw_log10_eta_strong = 21.0,
    uw_log10_eta_weak = 12.0,

    # Boundary conditions
    uw_v_plate_cm_per_yr = 1.0,
)

# %%
params

# %% [markdown]
"""
## Convert Parameters to Physical Quantities
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

# Adaptation parameters with units
ADAPT_H_NEAR = params.uw_adapt_h_near_km * u.kilometer
ADAPT_H_FAR = params.uw_adapt_h_far_km * u.kilometer
ADAPT_WIDTH = params.uw_adapt_width_km * u.kilometer

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

# Visualization setup - only render in interactive/notebook mode
render = False
try:
    import pyvista as pv
    import underworld3.visualisation as vis
    # Only render if in notebook or has display
    if hasattr(pv, 'BUILDING_GALLERY'):
        render = False  # Skip during docs build
    elif os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        render = True
    # Check for Jupyter notebook
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            render = True
    except ImportError:
        pass
    if uw.mpi.size > 1:
        render = False
except ImportError:
    render = False

# %% [markdown]
"""
## Initial (Coarse) Mesh

We start with a coarser mesh than the target resolution. The adaptation will
refine near the fault and maintain coarse resolution elsewhere.
"""

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0 * u.kilometer, 0.0 * u.kilometer),
    maxCoords=(DOMAIN_X, DOMAIN_Y),
    cellSize=CELL_SIZE,
    regular=False,
    qdegree=3,
)

n_elements_initial = mesh.dm.getChart()[1] - mesh.dm.getChart()[0]
print(f"Initial mesh: ~{n_elements_initial} entities")

# %% [markdown]
"""
## Fault Surface

Create the embedded fault surface. The Surface automatically registers with
the mesh for adaptation notifications.
"""

# %%
fault_points = np.array([
    [FAULT_X_MIN.magnitude, FAULT_Y.magnitude],
    [FAULT_X_MAX.magnitude, FAULT_Y.magnitude],
]) * u.kilometer

fault = uw.meshing.Surface("main_fault", mesh, fault_points, symbol="F")
fault.discretize()

print(f"Fault registered with mesh: {fault in mesh._registered_surfaces}")

# %% [markdown]
r"""
## Mesh Adaptation

### Creating the Refinement Metric

The `fault.refinement_metric()` method creates an H-field (target edge length)
based on distance from the fault:

$$h(d) = h_{\text{near}} + (h_{\text{far}} - h_{\text{near}}) \cdot f\left(\frac{|d|}{\text{width}}\right)$$

where $f$ is a transition function (linear, smoothstep, or gaussian).
"""

# %%
# Create refinement metric based on fault distance
# h_near: element size at the fault
# h_far: element size far from the fault
# width: transition distance
metric = fault.refinement_metric(
    h_near=ndim(ADAPT_H_NEAR),  # Non-dimensional
    h_far=ndim(ADAPT_H_FAR),
    width=ndim(ADAPT_WIDTH),
    profile="smoothstep",
)

with mesh.access(metric):
    print(f"Metric range: h_min={metric.data.min():.4f}, h_max={metric.data.max():.4f}")

# %% [markdown]
"""
### Visualize the Metric Field

The metric field shows target element sizes across the domain.
"""

# %%
if render:
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["h_target"] = vis.scalar_fn_to_pv_points(pvmesh, metric.sym[0])

    pl = pv.Plotter(window_size=(800, 400))
    pl.add_mesh(pvmesh, scalars="h_target", cmap="viridis_r", show_edges=True)

    fp = fault.control_points
    pl.add_mesh(pv.Line((fp[0, 0], fp[0, 1], 0), (fp[1, 0], fp[1, 1], 0)),
                color="red", line_width=4)
    pl.add_title("Target Element Size (h-field)")
    pl.view_xy()
    pl.show()

# %% [markdown]
"""
### Perform Adaptation

The `mesh.adapt()` method:
1. Calls PETSc/MMG mesh adaptation
2. Reinitializes MeshVariables (reset to zero)
3. Notifies surfaces to recompute distance fields
4. Marks solvers for rebuild
"""

# %%
print("Adapting mesh...")
mesh.adapt(metric, verbose=True)

n_elements_adapted = mesh.dm.getChart()[1] - mesh.dm.getChart()[0]
print(f"\nAdapted mesh: ~{n_elements_adapted} entities")
print(f"Ratio: {n_elements_adapted / n_elements_initial:.2f}×")

# %% [markdown]
"""
### Visualize Adapted Mesh
"""

# %%
if render:
    pvmesh = vis.mesh_to_pv_mesh(mesh)

    pl = pv.Plotter(window_size=(800, 400))
    pl.add_mesh(pvmesh, show_edges=True, edge_color="gray", color="lightblue")

    fp = fault.control_points
    pl.add_mesh(pv.Line((fp[0, 0], fp[0, 1], 0), (fp[1, 0], fp[1, 1], 0)),
                color="red", line_width=4)
    pl.add_title("Adapted Mesh (refined near fault)")
    pl.view_xy()
    pl.show()

# %% [markdown]
"""
## Stokes Solver Setup

After adaptation, we set up the Stokes solver. Note that:
- MeshVariables were reset to zero during adaptation
- The solver will be built fresh on the adapted mesh
"""

# %%
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel

# %% [markdown]
r"""
### Configure Anisotropic Rheology

The director is the fault normal, defining the weak plane orientation.
"""

# %%
# Director from fault normal
fault_normal = fault.normals[0, :2]
director = sympy.Matrix([fault_normal[0], fault_normal[1]]).T

# Influence function for rheology (uses recomputed distance field)
fault_influence = fault.influence_function(
    width=FAULT_WIDTH,
    value_near=1.0,
    value_far=0.0,
    profile="gaussian",
)

# Non-dimensional viscosities
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
### Boundary Conditions
"""

# %%
V_nd = ndim(V_PLATE)
stokes.add_dirichlet_bc((V_nd, 0.0), "Top")
stokes.add_dirichlet_bc((-V_nd, 0.0), "Bottom")
stokes.add_dirichlet_bc((None, 0.0), "Left")
stokes.add_dirichlet_bc((None, 0.0), "Right")

# %% [markdown]
"""
## Solve on Adapted Mesh
"""

# %%
print("Solving Stokes on adapted mesh...")
stokes.solve( verbose=False)
print("Done!")

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

# %%
uw.

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
    pl.add_mesh(pvmesh, scalars="Vmag", cmap="viridis", show_edges=True,
                edge_color="gray", edge_opacity=0.3)
    arrows = velocity_points.glyph(orient="V", scale=0.01, factor=0.15)
    pl.add_mesh(arrows, color="white", opacity=0.2)

    fp = fault.control_points
    pl.add_mesh(pv.Line((fp[0, 0], fp[0, 1], 0), (fp[1, 0], fp[1, 1], 0)),
                color="red", line_width=3)
    pl.add_title("Velocity Magnitude (Adapted Mesh)")
    pl.view_xy()
    pl.show()

# %%
if render:
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["SR"] = vis.scalar_fn_to_pv_points(pvmesh, strain_rate.sym[0])

    pl = pv.Plotter(window_size=(800, 400))
    pl.add_mesh(pvmesh, scalars="SR", cmap="hot", show_edges=True,
                edge_color="gray", edge_opacity=0.3, log_scale=True)

    fp = fault.control_points
    pl.add_mesh(pv.Line((fp[0, 0], fp[0, 1], 0), (fp[1, 0], fp[1, 1], 0)),
                color="cyan", line_width=3)
    pl.add_title("Strain Rate (Adapted Mesh)")
    pl.view_xy()
    pl.show()

# %% [markdown]
"""
## Strain Rate Profile
"""

# %%
# Non-dimensional values for sampling
Lx_nd = ndim(DOMAIN_X)
Ly_nd = ndim(DOMAIN_Y)
fault_y_nd = ndim(FAULT_Y)
fault_width_nd = ndim(FAULT_WIDTH)

# Note: Function evaluation after adaptation may have issues with variable indexing
# Using try/except to handle gracefully
try:
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
    ax.set_title('Strain Rate Profile (Adapted Mesh)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/strain_rate_profile_adaptive.png", dpi=150)
    plt.close()
except Exception as e:
    print(f"Note: Strain rate profile skipped (function evaluation issue after adaptation): {e}")

# %% [markdown]
"""
## Quantify Strain Localization
"""

# %%
try:
    x_sample = np.linspace(0.1, Lx_nd - 0.1, 50)

    coords_on = np.column_stack([x_sample, np.full(50, fault_y_nd)])
    sr_on = uw.function.evaluate(strain_rate.sym[0], coords_on).flatten()

    coords_off = np.column_stack([x_sample, np.full(50, 0.2)])
    sr_off = uw.function.evaluate(strain_rate.sym[0], coords_off).flatten()

    localization = sr_on.mean() / sr_off.mean()
    print(f"Localization factor: {localization:.1f}x")
except Exception as e:
    print(f"Note: Localization calculation skipped (function evaluation issue after adaptation): {e}")

# %% [markdown]
"""
## Save Output
"""

# %%
fault.save("output/fault_surface_adaptive.vtk")
mesh.write_timestep("shear_box_2d_adaptive", meshVars=[v, p, strain_rate],
                    outputPath="./output", index=0)

# %% [markdown]
r"""
## Summary

This example demonstrated **mesh adaptation** for fault-localized problems:

1. **Coarse initial mesh**: Started 2× coarser than target uniform resolution
2. **Surface-based metric**: `fault.refinement_metric()` creates H-field from distance
3. **Adaptation**: `mesh.adapt(metric)` refines near fault, coarsens elsewhere
4. **Element count control**: ~10× finer at fault with similar total count
5. **Automatic updates**: Surfaces recompute distance fields, variables reset

### Adaptation Parameters

| Parameter | Effect |
|-----------|--------|
| `h_near` | Element size at the fault (smaller = finer) |
| `h_far` | Element size far from fault |
| `width` | Transition distance from h_near to h_far |
| `profile` | Transition shape: "linear", "smoothstep", "gaussian" |

### Running with AMR Environment

This example requires PETSc built with MMG support:
```bash
pixi run -e amr python shear_box_2d_fault_adaptive.py
```

### Command-line Override

Adaptation parameters can be modified:
```bash
python shear_box_2d_fault_adaptive.py \
    -uw_adapt_h_near_km 0.025 \
    -uw_adapt_width_km 2.0
```
"""
