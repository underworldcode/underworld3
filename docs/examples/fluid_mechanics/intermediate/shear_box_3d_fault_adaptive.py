# %% [markdown]
r"""
# 3D Adaptive Shear Box with Oblique Fault

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** intermediate
**RUNTIME:** < 5 minutes (requires AMR environment)

## Description

A 3D shear box with an **oblique fault plane** that:
- Is not axis-aligned (strikes at ~20° from the y-axis)
- Reaches the top surface but **terminates at depth** (fault tip within domain)
- Creates stress concentrations at the fault tip

This demonstrates 3D mesh adaptation with a more realistic fault geometry
that challenges the mesh adaptation algorithm.

## Physics Setup

- **Driving**: Left/right walls impose opposing velocities (simple shear)
- **Top/Bottom**: Free slip (zero normal velocity, free tangential)
- **Front/Back**: Free slip
- **Fault**: Weak anisotropic zone localized to the fault plane

## Key Differences from 2D

1. **Fault tip effects**: The fault terminates at depth, creating 3D stress
   concentration at the tip line
2. **Oblique geometry**: Tests that metric field handles non-axis-aligned features
3. **3D adaptation**: MMG3D handles tetrahedral remeshing
"""

# %%
#| echo: false
import nest_asyncio
nest_asyncio.apply()

# %% [markdown]
"""
## Parameters
"""

# %%
import os
import numpy as np
import underworld3 as uw
import sympy

# Units and scaling
u = uw.scaling.units
ndim = uw.scaling.non_dimensionalise

# Scaling system (same as 2D)
scaling = uw.scaling.get_coefficients()
scaling["[length]"] = 10 * u.kilometer
scaling["[time]"] = 10 * u.kilometer / (1 * u.centimeter / u.year)
scaling["[mass]"] = (1e21 * u.pascal * u.second) * (10 * u.kilometer) * scaling["[time]"]

# Define all tunable parameters
params = uw.Params(
    # Domain geometry (slightly elongated in y to accommodate oblique fault)
    uw_domain_x_km = 10.0,
    uw_domain_y_km = 15.0,
    uw_domain_z_km = 10.0,
    uw_cell_size_km = 1.0,           # Coarse initial mesh for 3D

    # Fault geometry
    # Fault plane defined by: center point, strike angle, and vertical extent
    uw_fault_center_x_km = 5.0,
    uw_fault_center_y_km = 7.5,
    uw_fault_strike_deg = 20.0,       # Strike angle from y-axis (non-axis-aligned!)
    uw_fault_length_km = 10.0,        # Along-strike length
    uw_fault_top_z_km = 10.0,         # Reaches top surface
    uw_fault_bottom_z_km = 4.0,       # Tips out at depth (doesn't reach bottom!)
    uw_fault_width_km = 0.5,          # Rheology transition width

    # Adaptation parameters
    uw_adapt_h_near_km = 0.2,         # Fine resolution at fault
    uw_adapt_h_far_km = 1.5,          # Coarse away from fault
    uw_adapt_width_km = 2.0,          # Transition width

    # Material properties
    uw_log10_eta_strong = 21.0,
    uw_log10_eta_weak = 18.0,         # Less extreme contrast for 3D

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
DOMAIN_Z = params.uw_domain_z_km * u.kilometer
CELL_SIZE = params.uw_cell_size_km * u.kilometer

# Fault geometry
FAULT_CENTER_X = params.uw_fault_center_x_km * u.kilometer
FAULT_CENTER_Y = params.uw_fault_center_y_km * u.kilometer
FAULT_STRIKE = np.radians(params.uw_fault_strike_deg)  # Convert to radians
FAULT_LENGTH = params.uw_fault_length_km * u.kilometer
FAULT_TOP_Z = params.uw_fault_top_z_km * u.kilometer
FAULT_BOTTOM_Z = params.uw_fault_bottom_z_km * u.kilometer
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

# Visualization setup
render = False
try:
    import pyvista as pv
    import underworld3.visualisation as vis
    if hasattr(pv, 'BUILDING_GALLERY'):
        render = False
    elif os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        render = True
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
## Build Fault Geometry

The fault is an oblique vertical plane defined by 4 corners.
Strike angle rotates it away from axis alignment.
"""

# %%
# Build fault corner points
# The fault plane is vertical, with strike rotated from y-axis
cx = FAULT_CENTER_X.magnitude
cy = FAULT_CENTER_Y.magnitude
half_len = FAULT_LENGTH.magnitude / 2
z_top = FAULT_TOP_Z.magnitude
z_bot = FAULT_BOTTOM_Z.magnitude

# Strike direction (rotated from y-axis)
strike_x = np.sin(FAULT_STRIKE)  # Component in x-direction
strike_y = np.cos(FAULT_STRIKE)  # Component in y-direction

# Four corners of the fault plane (clockwise from top-left looking at fault)
# Top edge
top_p1 = np.array([cx - half_len * strike_x, cy - half_len * strike_y, z_top])
top_p2 = np.array([cx + half_len * strike_x, cy + half_len * strike_y, z_top])
# Bottom edge (fault tips out here)
bot_p1 = np.array([cx - half_len * strike_x, cy - half_len * strike_y, z_bot])
bot_p2 = np.array([cx + half_len * strike_x, cy + half_len * strike_y, z_bot])

fault_points = np.array([top_p1, top_p2, bot_p2, bot_p1]) * u.kilometer

print(f"Fault strike: {params.uw_fault_strike_deg}° from y-axis")
print(f"Fault extends: z = {z_bot} to {z_top} km")
print(f"Fault corners:\n{fault_points.magnitude}")

# %% [markdown]
"""
## Initial (Coarse) Mesh
"""

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0 * u.kilometer, 0.0 * u.kilometer, 0.0 * u.kilometer),
    maxCoords=(DOMAIN_X, DOMAIN_Y, DOMAIN_Z),
    cellSize=CELL_SIZE,
    regular=False,
    qdegree=3,
)

print(f"Mesh dimension: {mesh.dim}")
n_elements_initial = mesh.dm.getHeightStratum(0)[1] - mesh.dm.getHeightStratum(0)[0]
print(f"Initial mesh: {n_elements_initial} elements")

# %% [markdown]
"""
## Fault Surface

Create the embedded fault as a 3D planar surface.
"""

# %%
fault = uw.meshing.Surface("oblique_fault", mesh, fault_points, symbol="F")
fault.discretize()

# Compute fault normal (should be perpendicular to strike, in x-z plane for vertical fault)
# For vertical fault with this strike, normal points in direction (-strike_y, strike_x, 0)
fault_normal = np.array([-np.cos(FAULT_STRIKE), np.sin(FAULT_STRIKE), 0.0])
fault_normal = fault_normal / np.linalg.norm(fault_normal)
print(f"Fault normal: {fault_normal}")

# %% [markdown]
r"""
## Mesh Adaptation

Create refinement metric based on distance to the oblique fault plane.
"""

# %%
metric = fault.refinement_metric(
    h_near=ndim(ADAPT_H_NEAR),
    h_far=ndim(ADAPT_H_FAR),
    width=ndim(ADAPT_WIDTH),
    profile="smoothstep",
)

with mesh.access(metric):
    print(f"Metric range: h_min={metric.data.min():.4f}, h_max={metric.data.max():.4f}")

# %% [markdown]
"""
### Perform Adaptation
"""

# %%
print("Adapting 3D mesh...")
mesh.adapt(metric, verbose=True)

n_elements_adapted = mesh.dm.getChart()[1] - mesh.dm.getChart()[0]
print(f"\nAdapted mesh: ~{n_elements_adapted} entities")
print(f"Ratio: {n_elements_adapted / n_elements_initial:.2f}×")

# %% [markdown]
"""
### Visualize Adapted Mesh

Show a slice through the mesh to see refinement near the oblique fault.
"""

# %%
if render:
    pvmesh = vis.mesh_to_pv_mesh(mesh)

    # Create a slice perpendicular to z at mid-height
    slice_z = pvmesh.slice(normal='z', origin=(0, 0, ndim(DOMAIN_Z) / 2))

    pl = pv.Plotter(window_size=(800, 600))
    pl.add_mesh(slice_z, show_edges=True, edge_color="gray", color="lightblue")

    # Add fault plane as a quad
    fp = fault.control_points
    fault_quad = pv.Quadrilateral([fp[0], fp[1], fp[2], fp[3]])
    pl.add_mesh(fault_quad, color="red", opacity=0.5)

    pl.add_title("Adapted Mesh (z-slice) with Oblique Fault")
    pl.view_xy()
    pl.show()

# %%
if render:
    # 3D view showing mesh refinement
    pvmesh = vis.mesh_to_pv_mesh(mesh)

    pl = pv.Plotter(window_size=(900, 700))

    # Show mesh with edges
    pl.add_mesh(pvmesh, show_edges=True, edge_color="gray",
                color="lightblue", opacity=0.3)

    # Add fault plane
    fp = fault.control_points
    fault_quad = pv.Quadrilateral([fp[0], fp[1], fp[2], fp[3]])
    pl.add_mesh(fault_quad, color="red", opacity=0.7, label="Fault")

    pl.add_title("3D Adapted Mesh with Oblique Fault")
    pl.show()

# %% [markdown]
"""
## Stokes Solver Setup

Configure the 3D Stokes problem with:
- Anisotropic rheology near the fault
- Shearing boundary conditions
"""

# %%
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel

# %% [markdown]
r"""
### Configure Anisotropic Rheology

The director is the fault normal (3D vector), defining the weak slip plane.
"""

# %%
# 3D director from fault normal
director = sympy.Matrix([fault_normal[0], fault_normal[1], fault_normal[2]]).T

# Influence function (uses distance field recomputed after adaptation)
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

- **Left/Right (x-normal)**: Driving velocity in y-direction (simple shear)
- **Top/Bottom (z-normal)**: Free slip
- **Front/Back (y-normal)**: Free slip
"""

# %%
V_nd = ndim(V_PLATE)

# Shearing: Left wall moves +y, Right wall moves -y
stokes.add_dirichlet_bc((0.0, V_nd, 0.0), "Left")    # +y velocity
stokes.add_dirichlet_bc((0.0, -V_nd, 0.0), "Right")  # -y velocity

# Free slip on top/bottom (no z-velocity, free x,y)
stokes.add_dirichlet_bc((None, None, 0.0), "Top")
stokes.add_dirichlet_bc((None, None, 0.0), "Bottom")

# Free slip on front/back (no y-velocity component perpendicular to boundary)
# Actually for simple shear, we want y-velocity free on these faces
# stokes.add_dirichlet_bc((0.0, None, None), "Front")
# stokes.add_dirichlet_bc((0.0, None, None), "Back")

# %% [markdown]
"""
## Solve on Adapted Mesh
"""

# %%
print("Solving 3D Stokes on adapted mesh...")
stokes.solve(verbose=False)
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
uw.function.evaluate( stokes.Unknowns.Einv2, mesh.X.coords)

# %% [markdown]
"""
## Visualize Results
"""

# %%
if render:
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)
    pvmesh.point_data["Vmag"] = np.linalg.norm(pvmesh.point_data["V"], axis=1)

    # Slice through middle of domain
    slice_mesh = pvmesh.slice(normal='z', origin=(0, 0, ndim(DOMAIN_Z) / 2))

    pl = pv.Plotter(window_size=(800, 600))
    pl.add_mesh(slice_mesh, scalars="Vmag", cmap="viridis", show_edges=True)

    # Fault plane
    fp = fault.control_points
    fault_quad = pv.Quadrilateral([fp[0], fp[1], fp[2], fp[3]])
    pl.add_mesh(fault_quad, color="red", opacity=0.5, line_width=2)

    pl.add_title("Velocity Magnitude (z-slice)")
    pl.view_xy()
    pl.show()

# %%
if render:
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["SR"] = vis.scalar_fn_to_pv_points(pvmesh, strain_rate.sym[0])

    # Slice through middle
    slice_mesh = pvmesh.slice(normal='z', origin=(0, 0, ndim(DOMAIN_Z) / 2))

    pl = pv.Plotter(window_size=(800, 600))
    pl.add_mesh(slice_mesh, scalars="SR", cmap="hot", show_edges=False, log_scale=True)

    fp = fault.control_points
    fault_quad = pv.Quadrilateral([fp[0], fp[1], fp[2], fp[3]])
    pl.add_mesh(fault_quad, color="cyan", opacity=0.5)

    pl.add_title("Strain Rate (z-slice, log scale)")
    pl.view_xy()
    pl.show()

# %% [markdown]
"""
### 3D Strain Rate Visualization

Show the strain rate in 3D, highlighting the fault tip region.
"""

# %%
if render:
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["SR"] = vis.scalar_fn_to_pv_points(pvmesh, strain_rate.sym[0])

    # Threshold to show only high strain rate regions
    sr_data = pvmesh.point_data["SR"]
    sr_threshold = np.percentile(sr_data[sr_data > 0], 90)

    pl = pv.Plotter(window_size=(900, 700))

    # Show mesh outline
    pl.add_mesh(pvmesh.outline(), color="black", line_width=1)

    # High strain rate isosurface or threshold
    try:
        high_sr = pvmesh.threshold(value=sr_threshold, scalars="SR")
        pl.add_mesh(high_sr, scalars="SR", cmap="hot", opacity=0.7)
    except:
        pass

    # Fault plane
    fp = fault.control_points
    fault_quad = pv.Quadrilateral([fp[0], fp[1], fp[2], fp[3]])
    pl.add_mesh(fault_quad, color="cyan", opacity=0.5, label="Fault")

    pl.add_title("High Strain Rate Regions (3D)")
    pl.show()

# %% [markdown]
"""
## Strain Rate Profile Across Fault

Sample strain rate along a line crossing the fault at mid-depth.
"""

# %%
Lx_nd = ndim(DOMAIN_X)
Ly_nd = ndim(DOMAIN_Y)
Lz_nd = ndim(DOMAIN_Z)
fault_center_x_nd = ndim(FAULT_CENTER_X)
fault_center_y_nd = ndim(FAULT_CENTER_Y)
z_sample = ndim(DOMAIN_Z) / 2  # Mid-height

try:
    n_samples = 80
    # Sample line perpendicular to fault strike (i.e., along fault normal direction)
    # This crosses the fault at different distances
    x_sample = np.linspace(0.1, Lx_nd - 0.1, n_samples)

    # At each x, compute y to stay along a line through fault center perpendicular to strike
    # For simplicity, sample at constant y = fault center y
    profile_coords = np.column_stack([
        x_sample,
        np.full(n_samples, fault_center_y_nd),
        np.full(n_samples, z_sample)
    ])

    sr_profile = uw.function.evaluate(strain_rate.sym[0], profile_coords).flatten()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(x_sample, sr_profile, 'b-', linewidth=2, label='Strain rate')
    ax.axvline(x=fault_center_x_nd, color='r', linestyle='--', linewidth=2, label='Fault')

    ax.set_xlabel('x (model units)')
    ax.set_ylabel('Strain rate invariant')
    ax.set_title(f'Strain Rate Profile at y={fault_center_y_nd:.1f}, z={z_sample:.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/strain_rate_profile_3d_adaptive.png", dpi=150)
    plt.close()
    print("Saved strain rate profile to output/strain_rate_profile_3d_adaptive.png")

except Exception as e:
    print(f"Note: Profile skipped: {e}")

# %% [markdown]
"""
## Vertical Profile: Fault Tip Effect

Sample strain rate along the fault, from surface to depth, to see the tip effect.
"""

# %%
try:
    n_z = 50
    z_profile = np.linspace(0.2, Lz_nd - 0.2, n_z)

    # Sample along the fault center line
    vertical_coords = np.column_stack([
        np.full(n_z, fault_center_x_nd),
        np.full(n_z, fault_center_y_nd),
        z_profile
    ])

    sr_vertical = uw.function.evaluate(strain_rate.sym[0], vertical_coords).flatten()

    # Fault tip location
    fault_tip_z = ndim(FAULT_BOTTOM_Z)

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.semilogx(sr_vertical, z_profile, 'b-', linewidth=2)
    ax.axhline(y=fault_tip_z, color='r', linestyle='--', linewidth=2, label='Fault tip')
    ax.axhline(y=ndim(FAULT_TOP_Z), color='r', linestyle='-', linewidth=1, alpha=0.5)

    ax.fill_betweenx([fault_tip_z, ndim(FAULT_TOP_Z)],
                     ax.get_xlim()[0], ax.get_xlim()[1],
                     alpha=0.2, color='red', label='Fault extent')

    ax.set_xlabel('Strain rate invariant')
    ax.set_ylabel('z (model units)')
    ax.set_title('Vertical Strain Rate Profile at Fault Center')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/strain_rate_vertical_3d_adaptive.png", dpi=150)
    plt.close()
    print("Saved vertical profile to output/strain_rate_vertical_3d_adaptive.png")

except Exception as e:
    print(f"Note: Vertical profile skipped: {e}")

# %% [markdown]
"""
## Save Output
"""

# %%
fault.save("output/fault_surface_3d_adaptive.vtk")
mesh.write_timestep("shear_box_3d_adaptive", meshVars=[v, p, strain_rate],
                    outputPath="./output", index=0)

# %% [markdown]
r"""
## Summary

This 3D example demonstrated mesh adaptation for:

1. **Oblique fault geometry**: 20° strike from axis-aligned, testing metric field
   on non-trivial geometries

2. **Fault tip termination**: Fault stops at 40% depth, creating 3D stress
   concentrations visible in strain rate field

3. **3D simple shear**: Side walls driving flow, top/bottom free slip

### Key Observations

- Mesh refines along the **entire oblique fault plane**, not just axis-aligned regions
- **Fault tip** should show elevated strain rate where fault terminates at depth
- Strain localization concentrates both on the fault and at its tip

### Running with AMR Environment

```bash
pixi run -e amr python shear_box_3d_fault_adaptive.py
```

### Command-line Override

```bash
python shear_box_3d_fault_adaptive.py \
    -uw_fault_strike_deg 30.0 \
    -uw_adapt_h_near_km 0.1
```
"""
