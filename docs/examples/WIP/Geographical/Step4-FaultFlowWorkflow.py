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

# Reset any stale model, then set up units for geographic workflow
uw.reset_default_model()

model = uw.Model()
model.set_reference_quantities(
    length=uw.quantity(1000, "km"),
    viscosity=uw.quantity(1e21, "Pa.s"),
    diffusivity=uw.quantity(1e-6, "m**2/s"),
    verbose=False,
)

uw.timing.start()


# %% [markdown]
# ## 2. Load Fault Point Cloud
#
# The fault data is stored as an NPZ file with:
# - `arr_0`: geographic coordinates (lon, lat, depth_m, dip, fault_id, segment_id)
#
# Column 4 (`fault_id`) is a composite float like `11.3` where the integer part
# groups related faults and the fractional part identifies individual segments
# digitised by the operator.  Each unique `fault_id` is an independent trace
# that should be processed separately.
#
# `Surface.from_trace()` interpolates each trace to a target resolution and
# extrudes it to depth, producing a well-conditioned triangulated surface.

# %%
fault_data = np.load("Structures/faults_as_swarm_points_xyz.npz")
geo_coords = fault_data["arr_0"]

lon = geo_coords[:, 0]
lat = geo_coords[:, 1]
depth_m = geo_coords[:, 2]  # meters, negative (below surface)
dip = geo_coords[:, 3]
fault_id = geo_coords[:, 4]  # composite: e.g. 11.3 = fault 11, segment 3

# Each unique fault_id is an independent digitised segment
unique_fault_ids = np.unique(np.round(fault_id, 4))

print(f"Loaded {len(lon)} fault points across {len(unique_fault_ids)} segments")
print(f"Geographic extent: lon=[{lon.min():.2f}, {lon.max():.2f}], "
      f"lat=[{lat.min():.2f}, {lat.max():.2f}]")
print(f"Depth range: {depth_m.min()/1000:.0f} to {depth_m.max()/1000:.0f} km")

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
# `Surface.from_trace()` takes each fault's surface trace (lon, lat),
# interpolates it to ~3 km resolution, and extrudes to depth with a
# **parabolic dip profile**: vertical at the surface and reaching the
# recorded dip angle at maximum depth.
#
# Each Surface provides:
# - signed distance field (lazy, on demand)
# - `influence_function()` for smooth rheological transitions
# - `refinement_metric()` for mesh adaptation

# %%
fault_surfaces = {}

for fid in unique_fault_ids:
    mask = np.abs(np.round(fault_id, 4) - fid) < 1e-6
    pts = geo_coords[mask]

    # Extract surface trace (depth ≈ 0 layer)
    surface_mask = np.abs(pts[:, 2]) < 100  # within 100 m of surface
    if surface_mask.sum() < 2:
        continue

    trace = pts[surface_mask][:, :2]  # (lon, lat)
    depth_km_max = abs(pts[:, 2].min()) / 1000.0

    # Dip angle for this segment (degrees from horizontal)
    segment_dip = float(np.median(pts[:, 3]))

    # Detect depth spacing from the data
    unique_depths = np.unique(np.round(pts[:, 2] / 1000, 1))
    ds_km = float(np.median(np.abs(np.diff(np.sort(unique_depths))))) if len(unique_depths) > 1 else 5.0

    name = f"fault_{fid}"
    s = uw.meshing.Surface.from_trace(
        name, mesh, trace,
        depth_range=(uw.quantity(0, "km"), uw.quantity(depth_km_max, "km")),
        depth_spacing=uw.quantity(ds_km, "km"),
        trace_resolution=uw.quantity(3, "km"),
        dip=segment_dip,
        dip_direction="right",
        symbol=f"F{fid}",
    )
    print(f"  {name}: dip={segment_dip:.0f}°, {s.n_vertices} vertices, {s.n_triangles} triangles")
    fault_surfaces[fid] = s

print(f"\nCreated {len(fault_surfaces)} fault surfaces")

# %% [markdown]
# ## 6. Build Refinement Metric and Adapt Mesh
#
# `SurfaceCollection.refinement_metric()` computes a single combined metric
# using the minimum unsigned distance across all surfaces.  This creates only
# 2 MeshVariables (distance + metric) instead of 2 per surface, avoiding
# the O(N²) DM-rebuild cost that would occur with a per-surface loop.

# %%
faults = uw.meshing.SurfaceCollection()
for s in fault_surfaces.values():
    faults.add(s)

H_NEAR = uw.quantity(2.0, "km")      # target edge length near faults
H_FAR = uw.quantity(20.0, "km")      # target edge length far from faults
TRANSITION = uw.quantity(10.0, "km")  # transition width

combined_metric = faults.refinement_metric(
    mesh, h_near=H_NEAR, h_far=H_FAR, width=TRANSITION, profile="smoothstep",
)

print(f"Before adaptation: {mesh.X.coords.shape[0]} nodes")
mesh.adapt(combined_metric, verbose=True)
print(f"After adaptation:  {mesh.X.coords.shape[0]} nodes")

# %% [markdown]
# ## 7. Set Up Stokes Solver

# %%
v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2,
                                    varsymbol=r"\mathbf{v}", units="cm/yr")
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1,
                                    varsymbol="p", units="MPa")

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
# **Transverse isotropic (TI) model**: the viscosity is anisotropic with a weak
# shear direction aligned to the nearest fault normal.  Two mesh-wide fields are
# constructed from *all* fault surfaces:
#
# - `fault_normal` — unit normal of the nearest fault face (vector, 3 components)
# - `fault_weight` — Gaussian influence (1 on fault, 0 far away)
#
# A combined "nearest fault distance" field drives a Gaussian influence function
# that smoothly transitions the weak viscosity `eta_1` between the reference
# value (far from faults) and the weakened value (near faults).
#
# Three rheology modes are available:
# - `"anisotropic"` — transverse isotropic with fault-normal director (full model)
# - `"isotropic"` — isotropic weak zones aligned with faults
# - `"isoviscous"` — constant viscosity, no fault influence (baseline)

# %%
RHEOLOGY = "anisotropic"

# Physical parameters as expressions with units
eta_0 = uw.expression(r"\eta_0", uw.quantity(1e21, "Pa.s"), "reference viscosity")
eta_1_ratio = 0.1  # weak-to-strong viscosity ratio in fault zones
eta_1_weak = uw.expression(
    r"\eta_1", uw.quantity(eta_1_ratio * 1e21, "Pa.s"), "weak fault viscosity"
)
fault_width = uw.quantity(10.0, "km")  # Gaussian half-width for fault influence

if RHEOLOGY == "isoviscous":
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0

else:
    # Both "isotropic" and "anisotropic" need fault data fields.
    # compute_nearest_fields() builds a single KDTree over all surface
    # vertices, creating only 4 MeshVariables (normal, id, distance, weight)
    # instead of 2 per surface.

    fields = faults.compute_nearest_fields(mesh, fault_width=fault_width)
    fault_normal = fields["normal"]
    fault_id_var = fields["id"]
    nearest_dist = fields["distance"]
    fault_weight = fields["weight"]

    # Build composite viscosity using the precomputed Gaussian weight.
    # Keeping exp() in the symbolic expression would make the TI constitutive
    # tensor extremely complex and cause sympy.simplify() / solver hangs.
    eta_1_expr = eta_0 - (eta_0 - eta_1_weak) * fault_weight.sym[0]

    if RHEOLOGY == "isotropic":
        # Isotropic weak zones: composite viscosity in standard ViscousFlowModel.
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_1_expr

    else:  # "anisotropic"
        # Transverse isotropic: anisotropic model with fault-normal director
        stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0
        stokes.constitutive_model.Parameters.shear_viscosity_1 = eta_1_expr
        stokes.constitutive_model.Parameters.director = fault_normal.sym

stokes.saddle_preconditioner = 1.0 / uw.non_dimensionalise(eta_0)

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

# Driving velocity
V0 = uw.expression(r"V_0", uw.quantity(1, "cm/yr"), "driving velocity")
V0_nd = uw.non_dimensionalise(V0)

# Penalty scales with viscosity for proper balance
penalty = 1.0e6 * uw.non_dimensionalise(eta_0)

# Bottom: free-slip (no normal flow)
stokes.add_natural_bc(penalty * unit_down.dot(v.sym) * unit_down, "Bottom")

# East/West: driven shear (constrain only the east component)
stokes.add_natural_bc(penalty * (unit_east.dot(v.sym) + V0_nd) * unit_east, "East")
stokes.add_natural_bc(penalty * (unit_east.dot(v.sym) - V0_nd) * unit_east, "West")

# North/South: free-slip (no N–S flow)
stokes.add_natural_bc(penalty * unit_north.dot(v.sym) * unit_north, "North")
stokes.add_natural_bc(penalty * unit_north.dot(v.sym) * unit_north, "South")

# %% [markdown]
# ## 8. Solve

# %%
stokes.solve(verbose=False, debug=False)

print(f"Rheology: {RHEOLOGY}")
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

if RHEOLOGY != "isoviscous":
    pvmesh.point_data["fault_n"] = vis.vector_fn_to_pv_points(pvmesh, fault_normal.sym)
    pvmesh.point_data["fault_id"] = vis.scalar_fn_to_pv_points(pvmesh, fault_id_var.sym)
    pvmesh.point_data["fault_dist"] = vis.scalar_fn_to_pv_points(pvmesh, nearest_dist.sym)
    pvmesh.point_data["fault_w"] = vis.scalar_fn_to_pv_points(pvmesh, fault_weight.sym)

# %%
pl = pv.Plotter(window_size=(750, 750))

pl.add_mesh(pvmesh, cmap="coolwarm", scalars="Vmag", style="wireframe")
pl.add_mesh(pvmesh, cmap="coolwarm", scalars="fault_dist", style="surface", opacity=0.5)

pl.add_arrows(pvmesh.points, pvmesh.point_data["V"], mag=2e13, color="Blue")
# pl.add_arrows(pvmesh.points, pvmesh.point_data["fault_n"], mag=1e4, color="Red", opacity=0.33)

# Add fault surfaces with distinct colours
fault_colors = [
    "Red", "Orange", "Blue", "Green", "Purple",
    "Cyan", "Yellow", "Magenta", "Lime", "Pink",
]
for i, (sid, surface) in enumerate(fault_surfaces.items()):
    if surface.pv_mesh is not None:
        pl.add_mesh(
            surface.pv_mesh,
            style="surface",
            color=fault_colors[i % len(fault_colors)],
            opacity=1.0,
            show_edges=True,
            label=f"fault_{sid}",
        )

pl.show()

# %%

# %%
