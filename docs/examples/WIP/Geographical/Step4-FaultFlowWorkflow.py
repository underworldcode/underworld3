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

# %%
fault_data = np.load("Structures/faults_as_swarm_points_xyz.npz")
geo_coords = fault_data["arr_0"]

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
# Convert fault point locations from geographic (lon, lat, depth) to
# Cartesian (x, y, z) and nondimensionalise for the solver.
# Fault-plane normals are computed later by PyVista from the triangulated surfaces.

# %%
ell = ELLIPSOIDS["WGS84"]
a_km, b_km = ell["a"], ell["b"]

# depth_m is negative (below surface), geographic_to_cartesian expects
# positive depth, so negate the negative depth
depth_km = depth_m / 1000.0
fault_x, fault_y, fault_z = geographic_to_cartesian(
    lon, lat, -depth_km, a_km, b_km
)
# Cartesian coordinates in km
fault_xyz_km = np.column_stack([fault_x, fault_y, fault_z])

# Nondimensionalise via the units system (not manual division)
fault_xyz_nd = np.asarray(
    uw.non_dimensionalise(uw.quantity(fault_xyz_km, "km"))
)

print(f"Cartesian coordinates (km):")
print(f"  x: [{fault_x.min():.1f}, {fault_x.max():.1f}]")
print(f"  y: [{fault_y.min():.1f}, {fault_y.max():.1f}]")
print(f"  z: [{fault_z.min():.1f}, {fault_z.max():.1f}]")

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

    if pts.shape[0] < 3:
        print(f"  Segment {sid}: only {pts.shape[0]} points (skipped)")
        continue

    name = f"fault_{sid}"
    s = uw.meshing.Surface(name, mesh, pts, symbol=f"F{sid}")
    s.discretize()

    print(f"  {name}: {s.n_vertices} vertices")
    fault_surfaces[sid] = s

print(f"\nCreated {len(fault_surfaces)} fault surfaces")

# %% [markdown]
# ## 6. Build Refinement Metric and Adapt Mesh

# %%
H_NEAR = uw.quantity(3.0, "km")      # target edge length near faults
H_FAR = uw.quantity(30.0, "km")      # target edge length far from faults
TRANSITION = uw.quantity(10.0, "km")  # transition width

combined_metric = None
for sid, surface in fault_surfaces.items():
    m = surface.refinement_metric(
        h_near=H_NEAR, h_far=H_FAR, width=TRANSITION, profile="smoothstep",
    )
    if combined_metric is None:
        combined_metric = m
    else:
        combined_metric.data[:] = np.maximum(combined_metric.data, m.data)

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
    # Both "isotropic" and "anisotropic" need fault data fields

    # --- Composite fault normal and ID fields (all faults) ---
    # For each mesh node, find the nearest fault vertex across all surfaces
    # and copy its geological normal vector and fault segment ID.

    all_vertices = []
    all_normals = []
    all_fault_ids = []

    for sid, surface in fault_surfaces.items():
        pts = np.array(surface._pv_mesh.points)  # ND model coords
        norms = surface.normals  # PyVista triangulation normals (fault-plane normals)

        all_vertices.append(pts)
        all_normals.append(norms)
        all_fault_ids.append(np.full(len(pts), sid))

    combined_vertices = np.vstack(all_vertices)
    combined_normals = np.vstack(all_normals)
    combined_ids = np.concatenate(all_fault_ids)

    kdtree = uw.kdtree.KDTree(combined_vertices)
    mesh_coords = np.asarray(mesh._coords)
    _, closest_idx = kdtree.query(mesh_coords)
    closest_idx = closest_idx.flatten()

    fault_normal = uw.discretisation.MeshVariable(
        "fault_n", mesh, mesh.dim, degree=1, varsymbol=r"\hat{\mathbf{n}}_f"
    )
    fault_id_var = uw.discretisation.MeshVariable(
        "fault_id", mesh, 1, degree=1, varsymbol=r"f_{id}"
    )

    with uw.synchronised_array_update():
        fault_normal.data[:] = combined_normals[closest_idx]
        fault_id_var.data[:, 0] = combined_ids[closest_idx]

    # --- Combined influence function ---
    # Minimum absolute distance across all fault surfaces gives a single
    # "nearest fault" distance field. A Gaussian profile turns this into
    # a smooth rheological transition.

    fault_width_nd = float(uw.non_dimensionalise(fault_width))

    nearest_dist = uw.discretisation.MeshVariable(
        "d_faults", mesh, 1, degree=1, varsymbol=r"d_f"
    )

    min_dist = np.full(mesh_coords.shape[0], np.inf)
    for sid, surface in fault_surfaces.items():
        d = surface.distance  # lazy signed distance computation
        abs_dist = np.abs(d.data[:, 0])
        min_dist = np.minimum(min_dist, abs_dist)

    nearest_dist.data[:, 0] = min_dist

    # Precompute the Gaussian influence as a MeshVariable (not symbolic exp).
    # Keeping exp() in the symbolic expression makes the TI constitutive tensor
    # extremely complex and causes sympy.simplify() / solver hangs.
    fault_weight = uw.discretisation.MeshVariable(
        "fault_w", mesh, 1, degree=1, varsymbol=r"w_f"
    )
    fault_weight.data[:, 0] = np.exp(-0.5 * (min_dist / fault_width_nd) ** 2)

    # Build composite viscosity using the precomputed weight
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
pl.add_arrows(pvmesh.points, pvmesh.point_data["fault_n"], mag=1e4, color="Red", opacity=0.33)

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
