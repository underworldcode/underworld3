# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
r"""
# Geographic Mesh: Stokes Flow with Free-Slip Boundaries

**PHYSICS:** Stokes flow, free-slip boundary conditions, dimensional analysis
**DIFFICULTY:** intermediate

## Overview

The `RegionalGeographicBox` mesh provides **ellipsoidal Earth geometry** (WGS84).
This example demonstrates how to set up a Stokes problem with free-slip
boundary conditions on the top and bottom surfaces, using **physical units**.

The key difference from spherical meshes: **"up" is the geodetic normal**
(perpendicular to the ellipsoid), not the radial direction.

## Units System

This example uses Underworld's units system to:
- Define physical parameters with explicit units (Pa·s, kg/m³, etc.)
- Automatically nondimensionalise the problem for numerical stability
- Track dimensions through calculations

## Coordinate Access Pattern

**Cartesian coordinates are always available:**
- `mesh.X.coords` → $(x, y, z)$ data array (in km)
- `mesh.X[0], mesh.X[1], mesh.X[2]` → Symbolic $x, y, z$

**Geographic coordinates for GEOGRAPHIC meshes:**
- `mesh.X.geo.coords` → (lon, lat, depth) data array
- `mesh.X.geo.view()` → See all available properties
"""

# %%
import nest_asyncio
nest_asyncio.apply()

import underworld3 as uw
import numpy as np
import sympy

# %% [markdown]
r"""
## Set Up the Units System

For Stokes flow with buoyancy forcing, we need reference scales for:
- **Length**: 1000 km (mantle scale)
- **Viscosity**: $10^{21}$ Pa·s (mantle viscosity)
- **Diffusivity**: $10^{-6}$ m²/s (sets time scale)
"""

# %%
# Create Model and define reference quantities
model = uw.Model()

model.set_reference_quantities(
    length=uw.quantity(1000, "km"),
    viscosity=uw.quantity(1e21, "Pa.s"),
    diffusivity=uw.quantity(1e-6, "m**2/s"),
    verbose=False,
)

# %% [markdown]
r"""
## Define Physical Parameters

Named expressions with explicit units:
"""

# %%
# Material properties with units
eta_0 = uw.expression(r"\eta_0", uw.quantity(1e21, "Pa.s"), "reference viscosity")
delta_rho = uw.expression(r"\Delta\rho", uw.quantity(-100, "kg/m**3"), "buoyancy anomaly")
g = uw.expression(r"g", uw.quantity(10, "m/s**2"), "gravitational acceleration")
blob_radius = uw.expression(r"r_{blob}", uw.quantity(100, "km"), "anomaly radius")

print("Physical parameters defined with units:")
print(f"  η₀ = {eta_0.value}")
print(f"  Δρ = {delta_rho.value}")
print(f"  g  = {g.value}")
print(f"  r  = {blob_radius.value}")

# %% [markdown]
"""
## Create the Geographic Mesh

A regional mesh for southeastern Australia, 400 km deep.
When units are active, depth_range must have units.
"""

# %%
mesh = uw.meshing.RegionalGeographicBox(
    lon_range=(135, 145),                     # Degrees (raw = degrees)
    lat_range=(-38, -30),                     # Degrees
    depth_range=(uw.quantity(0, "km"),        # Depth with units
                 uw.quantity(400, "km")),
    ellipsoid="WGS84",
    numElements=(8, 8, 4),
    degree=1,
)

print(f"Coordinate system: {mesh.CoordinateSystem.coordinate_type}")
print(f"Nodes: {mesh.X.coords.shape[0]}")

# Quick look at coordinate ranges
geo = mesh.X.geo
print(f"\nGeographic ranges:")
print(f"  Longitude: [{geo.lon.min():.1f}°, {geo.lon.max():.1f}°]")
print(f"  Latitude:  [{geo.lat.min():.1f}°, {geo.lat.max():.1f}°]")
depth = geo.depth
if hasattr(depth, "units"):
    print(f"  Depth:     [{depth.min():.1f} {depth.units}, {depth.max():.1f} {depth.units}]")
else:
    print(f"  Depth:     [{depth.min():.1f}, {depth.max():.1f}] km")

# %% [markdown]
r"""
## Set Up the Stokes Problem

Create velocity and pressure variables for incompressible Stokes flow.
Use the Parameters pattern with expressions.
"""

# %%
# Velocity (vector) and pressure (scalar) variables
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2,
                                    varsymbol=r"\mathbf{v}", units="cm/yr")
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1,
                                    varsymbol="p", units="MPa")

# Create the Stokes solver
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)

# Use the Parameters pattern - assign expression with units
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0

print(f"Viscosity: {stokes.constitutive_model.Parameters.shear_viscosity_0}")

# %% [markdown]
r"""
## Free-Slip Boundary Conditions

Free-slip means:
- **No normal flow**: $\mathbf{v} \cdot \hat{\mathbf{n}} = 0$
- **No tangential stress**: Flow slides freely along the boundary

On a geographic mesh, the normal direction at the top/bottom surfaces is the
**geodetic normal** (perpendicular to the ellipsoid), accessed via `geo.unit_down`.

### Penalty Method Implementation

We enforce free-slip using a **penalty method** via `add_natural_bc`:

$$\mathbf{t} = \lambda \, (\mathbf{v} \cdot \hat{\mathbf{n}}) \, \hat{\mathbf{n}}$$

where $\lambda$ is a large penalty. This applies a restoring traction proportional
to the normal velocity component, driving $\mathbf{v} \cdot \hat{\mathbf{n}} \to 0$.
"""

# %%
# Get the geodetic unit vectors from the geo accessor
unit_down = geo.unit_down    # Into planet (geodetic normal)

# Free-slip via penalty method:
# Penalize normal velocity component: penalty * (v·n) * n
# This enforces v·n = 0 (no normal flow) while allowing tangential flow

# Penalty scales with viscosity to maintain relative magnitude
penalty_factor = 1e6
Vel_penalty = penalty_factor * uw.non_dimensionalise(eta_0)

print(f"Penalty = {penalty_factor:.0e} × η₀ (nondim) = {Vel_penalty:.2e}")

# Free-slip on surface (depth = 0)
stokes.add_natural_bc(Vel_penalty * unit_down.dot(v.sym) * unit_down, "Surface")

# Free-slip on bottom surface (depth = 400 km)
stokes.add_natural_bc(Vel_penalty * unit_down.dot(v.sym) * unit_down, "Bottom")

# No-slip on sides (for this demo)
for side in ["East", "West", "North", "South"]:
    stokes.add_dirichlet_bc((0.0, 0.0, 0.0), side)

print("Boundary conditions applied:")
print("  Surface, Bottom: Free-slip (penalty on normal velocity)")
print("  Sides: No-slip (fixed velocity)")

# %% [markdown]
r"""
## Add a Simple Buoyancy Force

Add a density anomaly to drive flow - a warm blob rising.

The body force is $\mathbf{f} = \Delta\rho \cdot g \cdot \hat{\mathbf{n}}_{down}$
where the density anomaly has a Gaussian spatial distribution.
"""

# %%
# Define a density anomaly (negative = buoyant)
# Centered at lon=140°, lat=-34°, depth=200 km
x, y, z = mesh.X

# Convert center location to Cartesian
# Use nondimensional depth (200 km / 1000 km = 0.2)
depth_center = uw.quantity(200, "km")
geo_center = geo.to_cartesian(140, -34, uw.non_dimensionalise(depth_center))

# Distance from anomaly center (using nondimensional coords)
dist_sq = (x - geo_center[0])**2 + (y - geo_center[1])**2 + (z - geo_center[2])**2

# Nondimensionalise the blob radius for the Gaussian
blob_radius_nd = uw.non_dimensionalise(blob_radius)

# Gaussian spatial variation (0 to 1)
gaussian = sympy.exp(-dist_sq / (2 * blob_radius_nd**2))

# Body force = delta_rho * g * gaussian * unit_down
# The expressions handle nondimensionalisation automatically
stokes.bodyforce = delta_rho * g * gaussian * unit_down

print(f"Buoyancy anomaly centered at: lon=140°, lat=-34°, depth=200 km")
print(f"Anomaly radius: {blob_radius.value}")
print(f"Peak density anomaly: {delta_rho.value}")

# %% [markdown]
"""
## Solve the Stokes Problem
"""

# %%
# Solve
stokes.solve()

print("Stokes solve complete!")

# Check velocity magnitude
v_mag = np.sqrt(v.data[:, 0]**2 + v.data[:, 1]**2 + v.data[:, 2]**2)
print(f"Velocity magnitude: [{v_mag.min():.2e}, {v_mag.max():.2e}]")

# %%
uw.pause("Visualise the solution when ready", explanation="pyvista may reset the kernel if you rush !")

# %% [markdown]
"""
## Visualize the Solution

Use PyVista to visualize the velocity field with arrows. This helps debug
the physics and understand the flow pattern.

Note: PyVista is optional. Install with `pip install pyvista` for visualization.
"""

# %%
if uw.is_notebook():
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)
    pvmesh.point_data["dRho"] = vis.scalar_fn_to_pv_points(pvmesh, delta_rho)

    # Streamlines from cell centers show flow pattern
    pvstream = pvmesh.streamlines_from_source(
        pvmesh.cell_centers(),
        vectors="V",
        integration_direction="both",
        integrator_type=45,
        surface_streamlines=False,
        # initial_step_length=0.01,
    )

    pl = pv.Plotter(window_size=(750, 750))

    # Mesh with density anomaly coloring
    pl.add_mesh(
        pvmesh,
        style="wireframe",
        cmap="coolwarm",
        edge_color="Grey",
        edge_opacity=0.33,
        scalars="dRho",
        show_edges=True,
        use_transparency=False,
        opacity=1.0,
        show_scalar_bar=False,
    )

    # Streamlines showing flow
    pl.add_mesh(
        pvstream,
        opacity=1,
        show_scalar_bar=False,
        cmap="Greens",
        render_lines_as_tubes=True,
    )

    pl.show()

# %% [markdown]
"""
## Analyze the Velocity Field

Even without visualization, we can examine the velocity components
to understand the flow pattern.
"""

# %%

# %%
# Compute the "up" component of velocity at each velocity DOF
# by projecting onto the geodetic normal

print("Analyzing velocity field...")

# Get coordinates at velocity DOFs (not mesh nodes - velocity is degree 2)
v_coords = v.coords  # Coordinates at velocity DOF locations
n_dofs = v_coords.shape[0]

print(f"Velocity DOFs: {n_dofs}")
print(f"Mesh nodes: {mesh.X.coords.shape[0]}")

# Evaluate geodetic down direction at each velocity DOF
unit_down_numeric = np.zeros((n_dofs, 3))
for j in range(3):
    vals = uw.function.evaluate(unit_down[j], v_coords, mesh.N)
    unit_down_numeric[:, j] = vals.flatten()

# Compute vertical (up) component of velocity at each DOF
v_vertical = -np.sum(v.data * unit_down_numeric, axis=1)  # Negative because up = -down

print(f"\nVertical velocity component (positive = upward):")
print(f"  Min:  {v_vertical.min():+.2e}")
print(f"  Max:  {v_vertical.max():+.2e}")
print(f"  Mean: {v_vertical.mean():+.2e}")

# Check where the max upward velocity is
max_up_idx = np.argmax(v_vertical)

# Get raw nondimensional coordinates from mesh DM
# (bypasses unit wrapping to get values matching ellipsoid scale)
dm_coords = mesh.dm.getCoordinates().array.reshape(-1, 3)

# For velocity DOFs (degree 2), we need to interpolate or use a mapping
# For simplicity, use the closest mesh node
mesh_coords_nd = dm_coords  # Nondimensional mesh node coords
v_coord_nd = v_coords[max_up_idx]

# Strip units if present
if hasattr(v_coord_nd, "magnitude"):
    v_coord_nd = v_coord_nd.magnitude / 1e6  # meters to km, then km to nondim
elif hasattr(v_coord_nd, "__array__"):
    # For UnitAwareArray, the values are in meters - convert to nondimensional
    L_ref_km = geo.cs.ellipsoid.get("L_ref_km", 1000.0)
    v_coord_nd = np.asarray(v_coord_nd) / (L_ref_km * 1000)  # m → km → nondim

# Convert to geographic (returns nondimensional depth)
max_up_lon, max_up_lat, max_up_depth_nd = geo.from_cartesian(
    v_coord_nd[0], v_coord_nd[1], v_coord_nd[2]
)

# Redimensionalize depth (multiply by L_ref in km)
L_ref_km = geo.cs.ellipsoid.get("L_ref_km", 1000.0)
max_up_depth = max_up_depth_nd * L_ref_km

print(f"\nMax upward velocity at:")
print(f"  lon={max_up_lon:.1f}°, lat={max_up_lat:.1f}°, depth={max_up_depth:.0f} km")

# %% [markdown]
"""
## Interpret the Results

The velocity analysis shows:
- **Positive vertical velocity** indicates upward flow (buoyancy-driven)
- **Location of max upward flow** should be near the buoyant blob (lon=140°, lat=-34°, depth=200km)

If flow is downward (negative), check the boundary condition setup.
"""

# %% [markdown]
r"""
## Summary

This example demonstrated:

1. **Units system setup** with `uw.Model()` and `set_reference_quantities()`
2. **Named expressions** using `uw.expression(symbol, quantity, description)`
3. **Geographic mesh creation** with `RegionalGeographicBox`
4. **Accessing geographic coordinates** via `mesh.X.geo`
5. **Parameters pattern** for constitutive models: `stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0`
6. **Properly-scaled penalty method**: `penalty_factor * uw.non_dimensionalise(eta_0)`
7. **Free-slip boundaries** using: `add_natural_bc(λ * n.dot(v.sym) * n, "boundary")`

## Key Points

### Units and Dimensional Analysis
- Create `uw.Model()` first, then set reference quantities
- Use `uw.expression()` to create named parameters with units
- Use `uw.non_dimensionalise()` when mixing dimensional and nondimensional values
- MeshVariables can have units for output: `units="cm/yr"`

### Geographic Coordinates
- The geodetic normal (`.unit_down`) differs from the radial direction by ~10 arcmin
- For regional models at 10-100 km scale, this difference matters
- Free-slip BCs on ellipsoidal surfaces use the geodetic normal, not radial

## See Also

- `Ex_Spherical_Meshes.py` — Spherical (non-ellipsoidal) meshes
- `15-Thermal-convection-with-units.py` — Full units example with time evolution
- `mesh.X.geo.view()` — Full property reference
"""
