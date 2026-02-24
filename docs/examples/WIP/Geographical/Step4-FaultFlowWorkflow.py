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

# %% [markdown]
# # Regional Fault-Controlled Flow (H2Ex Workflow)
#
# Demonstrates the complete workflow for modelling fluid flow controlled by
# regional fault structures, using native Underworld3 features:
#
# 1. Load fault point cloud data (geographic coordinates)
# 2. Convert to model coordinates and create Surface objects
# 3. Build refinement metric and adapt mesh around faults
# 4. Set up Stokes solver with transverse isotropic rheology
# 5. Solve for fault-controlled flow
#
# **Study Region**: Southeastern Australia (~135.5–137.5°E, ~34.5–33.0°S), 0–50 km depth.
# Fault data from the H2Ex project (Adelaide geoscience region).
#
# **Requirements**: `amr-dev` pixi environment (for mesh adaptation with MMG)

# %% [markdown]
# ## 1. Imports and Setup

# %%
import numpy as np
import underworld3 as uw
from underworld3.coordinates import ELLIPSOIDS, geographic_to_cartesian
import sympy

# Reset any stale model, then set up units for geographic workflow
uw.reset_default_model()

model = uw.Model()
model.set_reference_quantities(
    length=uw.quantity(1000, "km"),
    viscosity=uw.quantity(1e21, "Pa.s"),
    diffusivity=uw.quantity(1e-6, "m**2/s"),
    verbose=False,
)

# Reference length for manual nondimensionalisation of external data (e.g. fault coords)
L_REF_KM = 1000.0

uw.timing.start()


# %% [markdown]
# ## 2. Load Fault Point Cloud
#
# The fault data is stored as an NPZ file with:
# - `arr_0`: geographic coordinates (lon, lat, depth_m, dip, fault_id, segment_id)
# - `arr_1`: normal vectors (nx, ny, nz, dip, fault_id, segment_id)

# %%
fault_data = np.load("Structures/faults_as_swarm_points_xyz.npz")
geo_coords = fault_data["arr_0"]
normal_vecs = fault_data["arr_1"]

lon = geo_coords[:, 0]
lat = geo_coords[:, 1]
depth_m = geo_coords[:, 2]  # meters, negative (below surface)
dip = geo_coords[:, 3]
fault_id = geo_coords[:, 4]  # composite: integer.segment
segment_id = np.round(geo_coords[:, 5]).astype(int)

print(f"Loaded {len(lon)} fault points across {len(np.unique(segment_id))} segments")
print(f"Geographic extent: lon=[{lon.min():.2f}, {lon.max():.2f}], "
      f"lat=[{lat.min():.2f}, {lat.max():.2f}]")
print(f"Depth range: {depth_m.min()/1000:.0f} to {depth_m.max()/1000:.0f} km")

# %% [markdown]
# ## 3. Convert Geographic → Cartesian Coordinates
#
# The Surface class expects control points in the same coordinate system as
# the mesh. With units active, mesh coordinates are nondimensional Cartesian.
#
# The normals in the NPZ file are in a local geographic frame
# (east, north, up components) because they were computed from fault surface
# traces in geographic coordinates (by Faults-2-PointCloud.py).
# The mesh uses global Cartesian (x, y, z), so we must rotate the normals
# from geographic to Cartesian frame at each fault point.

# %%
ell = ELLIPSOIDS["WGS84"]
a_km, b_km = ell["a"], ell["b"]

# depth_m is negative (below surface), geographic_to_cartesian expects
# positive depth, so negate the negative depth
depth_km = depth_m / 1000.0
fault_x, fault_y, fault_z = geographic_to_cartesian(
    lon, lat, -depth_km, a_km, b_km
)
fault_xyz_all = np.column_stack([fault_x, fault_y, fault_z])

# Nondimensionalise to match mesh coordinate system
fault_xyz_nd = fault_xyz_all / L_REF_KM

print(f"Cartesian coordinates (km):")
print(f"  x: [{fault_x.min():.1f}, {fault_x.max():.1f}]")
print(f"  y: [{fault_y.min():.1f}, {fault_y.max():.1f}]")
print(f"  z: [{fault_z.min():.1f}, {fault_z.max():.1f}]")
print(f"Nondimensionalised (÷ {L_REF_KM:.0f} km)")

# %%
# Rotate normals from geographic frame to Cartesian frame.
#
# Geographic basis vectors at each point (from ellipsoid geometry):
#   unit_east  = (-sin(lon), cos(lon), 0)
#   unit_up    = (x/a², y/a², z/b²) / |...| (geodetic normal)
#   unit_north = unit_up × unit_east (normalised)
#
# A normal expressed as (n_east, n_north, n_up) becomes:
#   n_cartesian = n_east * unit_east + n_north * unit_north + n_up * unit_up

fault_normals_geo = normal_vecs[:, :3]  # (n_east, n_north, n_up) — geographic frame

lon_rad = np.radians(lon)
lat_rad = np.radians(lat)

# Unit east: (-sin(lon), cos(lon), 0)
ue = np.column_stack([-np.sin(lon_rad), np.cos(lon_rad), np.zeros_like(lon)])

# Unit up (geodetic normal): gradient of ellipsoid equation, normalised
gn_x = fault_x / a_km**2
gn_y = fault_y / a_km**2
gn_z = fault_z / b_km**2
gn_mag = np.sqrt(gn_x**2 + gn_y**2 + gn_z**2)
uu = np.column_stack([gn_x / gn_mag, gn_y / gn_mag, gn_z / gn_mag])

# Unit north: up × east, normalised
un = np.cross(uu, ue)
un = un / np.linalg.norm(un, axis=1, keepdims=True)

# Transform: n_cartesian = n_e * unit_east + n_n * unit_north + n_u * unit_up
fault_normals_all = (
    fault_normals_geo[:, 0:1] * ue +
    fault_normals_geo[:, 1:2] * un +
    fault_normals_geo[:, 2:3] * uu
)

print(f"Normal transformation (geographic → Cartesian):")
print(f"  Input magnitude:  [{np.linalg.norm(fault_normals_geo, axis=1).min():.3f}, "
      f"{np.linalg.norm(fault_normals_geo, axis=1).max():.3f}]")
print(f"  Output magnitude: [{np.linalg.norm(fault_normals_all, axis=1).min():.3f}, "
      f"{np.linalg.norm(fault_normals_all, axis=1).max():.3f}]")

# %% [markdown]
# ## 4. Create Geographic Mesh
#
# `RegionalGeographicBox` creates a mesh with WGS84 ellipsoidal geometry.
# With units active, mesh coordinates are nondimensional Cartesian (÷ L_ref).
# Geographic coordinates are accessible via `mesh.CoordinateSystem.geo`.

# %%
mesh = uw.meshing.RegionalGeographicBox(
    lon_range=(135.5, 137.5),
    lat_range=(-34.5, -33.0),
    depth_range=(uw.quantity(0, "km"), uw.quantity(50, "km")),
    ellipsoid="WGS84",
    numElements=(16, 16, 8),
    simplex=True,
)

print(f"Mesh: {mesh.dim}D, {mesh.X.coords.shape[0]} nodes")

# %% [markdown]
# ## 5. Create Fault Surfaces
#
# One `Surface` per fault segment. Each Surface provides:
# - triangulation (`discretize()`)
# - signed distance field (lazy, on demand)
# - `influence_function()` for smooth rheological transitions
# - `refinement_metric()` for mesh adaptation

# %%
fault_surfaces = {}
unique_segments = np.unique(segment_id)

for sid in unique_segments:
    mask = segment_id == sid
    pts = fault_xyz_nd[mask]
    norms = fault_normals_all[mask]

    if pts.shape[0] < 3:
        print(f"  Segment {sid}: only {pts.shape[0]} points (skipped)")
        continue

    name = f"fault_{sid}"
    s = uw.meshing.Surface(name, mesh, pts, symbol=f"F{sid}")
    s.discretize()

    # Store pre-computed normals as a SurfaceVariable
    n_var = s.add_variable("normals", size=3)

    norms_magnitude = np.linalg.norm(norms, axis=1)
    if norms_magnitude.min() < 0.5 or norms_magnitude.max() > 2.0:
        print(f"  WARNING: fault_{sid} normals have unexpected magnitudes"
              f" [{norms_magnitude.min():.3f}, {norms_magnitude.max():.3f}]")

    # Normalise to unit vectors before storing
    norms = norms / norms_magnitude[:, np.newaxis]
    n_var.data[:] = norms

    # Sanity check against triangulation normals
    pv_normals = s.normals
    if pv_normals is not None and len(pv_normals) == len(norms):
        alignment = np.abs(np.sum(norms * pv_normals, axis=1)).mean()
        print(f"  {name}: {s.n_vertices} vertices, alignment: {alignment:.3f}")
    else:
        print(f"  {name}: {s.n_vertices} vertices")

    fault_surfaces[sid] = s

print(f"\nCreated {len(fault_surfaces)} fault surfaces")

# %% [markdown]
# ## 6. Build Refinement Metric and Adapt Mesh

# %%
H_NEAR = 3.0  / L_REF_KM     # target edge length near faults (3 km, nondimensional)
H_FAR = 30.0  / L_REF_KM      # target edge length far from faults (30 km, nondimensional)
TRANSITION = 10.0 / L_REF_KM  # transition width (20 km, nondimensional)

combined_metric = None
for sid, surface in fault_surfaces.items():
    m = surface.refinement_metric(
        h_near=H_NEAR, h_far=H_FAR, width=TRANSITION, profile="smoothstep",
    )
    if combined_metric is None:
        combined_metric = m
    else:
        combined_metric.data[:] = np.maximum(combined_metric.data, m.data)

print(f"Metric range: [{combined_metric.data.min():.6f}, {combined_metric.data.max():.6f}]")

print(f"\nBefore adaptation: {mesh.X.coords.shape[0]} nodes")
mesh.adapt(combined_metric, verbose=True)
print(f"After adaptation: {mesh.X.coords.shape[0]} nodes")

# %%
# mesh.view()

# %% [markdown]
# ## 7. Set Up Stokes Solver

# %%
v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2,
                                    varsymbol=r"\mathbf{v}", units="cm/yr")
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1,
                                    varsymbol="p", units="MPa")

print(f"Velocity DOFs: {v.data.shape}")
print(f"Pressure DOFs: {p.data.shape}")

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)

stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["ksp_monitor"] = None

stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")

# %% [markdown]
# ### Rheology
#
# Set `USE_ISOVISCOUS = True` for debugging (constant viscosity, no fault influence).
# Set `USE_ISOVISCOUS = False` for the full transverse isotropic rheology.

# %%
USE_ISOVISCOUS = False

# Physical viscosity as expression with units
eta_0 = uw.expression(r"\eta_0", uw.quantity(1e21, "Pa.s"), "reference viscosity")
eta_0_nd = uw.non_dimensionalise(eta_0)

if USE_ISOVISCOUS:
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0
    print(f"Constitutive model: ISOVISCOUS (eta = {eta_0.value})")

else:
    stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel

    eta_1_weak_ratio = 0.01
    eta_1_weak = uw.expression(
        r"\eta_1", uw.quantity(eta_1_weak_ratio * 1e21, "Pa.s"), "weak fault viscosity"
    )

    primary_fault = fault_surfaces.get(11) or list(fault_surfaces.values())[0]
    print(f"Primary fault: '{primary_fault.name}' ({primary_fault.n_vertices} vertices)")

    fault_influence = primary_fault.influence_function(
        width=10.0 / L_REF_KM, value_near=1.0, value_far=0.0, profile="gaussian",
    )
    eta_1_expr = eta_0 - (eta_0 - eta_1_weak) * fault_influence

    normals_var = primary_fault.get_variable("normals")
    director = normals_var.sym

    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0
    stokes.constitutive_model.Parameters.shear_viscosity_1 = eta_1_expr
    stokes.constitutive_model.Parameters.director = director

    print(f"Constitutive model: TransverseIsotropicFlowModel")
    print(f"  eta_0 = {eta_0.value}, eta_1/eta_0 = {eta_1_weak_ratio}")

stokes.saddle_preconditioner = 1.0 / eta_0_nd

# %% [markdown]
# ### Boundary Conditions
#
# On a geographic (ellipsoidal) mesh, BCs must use **geographic basis vectors**.
# The Cartesian x-direction is NOT "east" on an ellipsoidal mesh.
#
# We use the **penalty method** via `add_natural_bc`:
# - **Surface**: free (stress-free) — open to accommodate incompressibility
# - **Bottom**: free-slip — no radial flow, tangential sliding allowed
# - **East/West**: driven shear in geographic east direction
# - **North/South**: free-slip — no N–S flow

# %%
geo = mesh.CoordinateSystem.geo
unit_down = geo.unit_down
unit_east = geo.unit_east
unit_north = geo.unit_north

# Driving velocity with units
V0 = uw.expression(r"V_0", uw.quantity(1, "cm/yr"), "driving velocity")
V0_nd = uw.non_dimensionalise(V0)

# Penalty scales with viscosity for proper balance
penalty_factor = 1.0e6
Vel_penalty = penalty_factor * eta_0_nd

print(f"V0 = {V0.value} → nondimensional: {V0_nd:.2e}")
print(f"Penalty = {penalty_factor:.0e} × η₀(nd) = {Vel_penalty:.2e}")

# Bottom: free-slip (no normal flow)
stokes.add_natural_bc(Vel_penalty * unit_down.dot(v.sym) * unit_down, "Bottom")

# East/West: driven shear (constrain only the east component)
stokes.add_natural_bc(Vel_penalty * (unit_east.dot(v.sym) + V0_nd) * unit_east, "East")
stokes.add_natural_bc(Vel_penalty * (unit_east.dot(v.sym) - V0_nd) * unit_east, "West")

# North/South: free-slip (no N–S flow)
stokes.add_natural_bc(Vel_penalty * unit_north.dot(v.sym) * unit_north, "North")
stokes.add_natural_bc(Vel_penalty * unit_north.dot(v.sym) * unit_north, "South")

# %% [markdown]
# ## 8. Solve

# %%
stokes.solve(verbose=False, debug=False)

print(f"\nSolution statistics:")
print(f"  |v|_max = {np.abs(v.data).max():.4f}")
print(f"  |p|_max = {np.abs(p.data).max():.4f}")

# %%
uw.timing.print_summary()

# %%
uw.pause("visualise when ready")

# %% [markdown]
# ## 9. Visualization

# %%
import pyvista as pv
import underworld3.visualisation as vis

pvmesh = vis.mesh_to_pv_mesh(mesh)

pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)
pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v.sym.dot(v.sym))
pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p.sym)
# simplify=False is CRITICAL — sympy.simplify() hangs on geographic basis vectors
pvmesh.point_data["N"] = vis.vector_fn_to_pv_points(pvmesh, unit_north, simplify=False)
pvmesh.point_data["E"] = vis.vector_fn_to_pv_points(pvmesh, unit_east, simplify=False)
pvmesh.point_data["Z"] = vis.vector_fn_to_pv_points(pvmesh, unit_down, simplify=False)

# %%
pl = pv.Plotter(window_size=(750, 750))

pl.add_mesh(pvmesh, cmap="coolwarm", scalars="Vmag", style="wireframe")
pl.add_arrows(pvmesh.points, pvmesh.point_data["V"], mag=1e32, color="Blue")
# pl.add_arrows(pvmesh.points, pvmesh.point_data["N"], mag=1e4, color="Red", opacity=0.33)

pl.show()
