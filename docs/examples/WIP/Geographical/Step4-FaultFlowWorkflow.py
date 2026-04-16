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
# ## 2. Load Fault Trace Data
#
# The canonical fault data is a CSV with columns:
# `Lon, Lat, Depth, Dip Angle at depth, Name, id, Description`
#
# The `Name` column is a composite float (e.g. `11.3`) where the integer part
# groups related faults and the fractional part identifies individual segments
# digitised by the operator.  Each unique `Name` is an independent trace
# that should be processed separately.
#
# `Surface.from_trace()` takes these surface traces and extrudes them to depth,
# producing well-conditioned triangulated surfaces — replacing the earlier
# pre-processed NPZ intermediate.

# %%
import csv
from pathlib import Path

# Look for the CSV in several locations
_csv_name = "CombinedInferredFaults_2025_04_22.csv"
_search_paths = [
    Path("Structures") / _csv_name,                                          # local
    Path.home() / "Library/CloudStorage/Box-Box/+Claude/H2Ex-Data/Structures" / _csv_name,  # Box (macOS)
    Path.home() / "Box/+Claude/H2Ex-Data/Structures" / _csv_name,            # Box (alias)
]
_csv_path = next((p for p in _search_paths if p.exists()), None)
if _csv_path is None:
    raise FileNotFoundError(
        f"Cannot find {_csv_name} in any of:\n" +
        "\n".join(f"  {p}" for p in _search_paths)
    )

# Parse CSV into structured arrays — no pandas dependency
_rows = []
with open(_csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        _rows.append(row)

# Build arrays from parsed rows
fault_lon   = np.array([float(r["Lon"]) for r in _rows])
fault_lat   = np.array([float(r["Lat"]) for r in _rows])
fault_depth = np.array([float(r["Depth"]) for r in _rows])
fault_dip   = np.array([float(r["Dip Angle at depth"]) for r in _rows])
fault_name  = np.array([float(r["Name"]) for r in _rows])
fault_desc  = [r["Description"].strip() for r in _rows]

unique_fault_ids = np.unique(fault_name)

print(f"Loaded {len(fault_lon)} fault trace points from {_csv_path.name}")
print(f"Geographic extent: lon=[{fault_lon.min():.2f}, {fault_lon.max():.2f}], "
      f"lat=[{fault_lat.min():.2f}, {fault_lat.max():.2f}]")
print(f"Depth range: {fault_depth.min():.0f} to {fault_depth.max():.0f} km")
print(f"Fault segments: {len(unique_fault_ids)}")

# %% [markdown]
# ## 3. Create Geographic Mesh
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
# ## 4. Create Fault Surfaces
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
    mask = fault_name == fid
    if mask.sum() < 2:
        continue

    trace = np.column_stack([fault_lon[mask], fault_lat[mask]])  # (N, 2)
    depth_km = float(fault_depth[mask][0])
    segment_dip = float(fault_dip[mask][0])
    description = fault_desc[np.argmax(mask)]  # first matching description

    name = f"fault_{fid}"
    s = uw.meshing.Surface.from_trace(
        name, mesh, trace,
        depth_range=(uw.quantity(0, "km"), uw.quantity(depth_km, "km")),
        depth_spacing=uw.quantity(5, "km"),
        trace_resolution=uw.quantity(3, "km"),
        dip=segment_dip,
        dip_direction="right",
        symbol=f"F{fid}",
    )
    print(f"  {name} ({description}): dip={segment_dip:.0f}°, depth={depth_km:.0f} km, "
          f"{s.n_vertices} vertices, {s.n_triangles} triangles")
    fault_surfaces[fid] = s

print(f"\nCreated {len(fault_surfaces)} fault surfaces")

# %% [markdown]
# ## 5. Build Refinement Metric and Adapt Mesh
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
# ## 6. Set Up Stokes Solver

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
# ## 7. Solve

# %%
stokes.solve(verbose=False)

print(f"Rheology: {RHEOLOGY}")
print(f"  v DOFs: {v.coords.shape[0] * mesh.dim}")
print(f"  |v|_max = {np.abs(v.data).max():.4f}")
print(f"  |p|_max = {np.abs(p.data).max():.4f}")

# %% [markdown]
# ## 8. Strain Rate Projection
#
# Project the second invariant of the strain rate tensor to a P1 field.
# This is a scalar measure of deformation intensity at each node.

# %%
strain_rate = uw.discretisation.MeshVariable(
    "eps", mesh, 1, degree=1, varsymbol=r"\dot\varepsilon"
)

proj = uw.systems.Projection(mesh, strain_rate)
proj.uw_function = stokes.constitutive_model.Unknowns.Einv2
proj.smoothing = 1.0e-6
proj.solve()

print(f"Strain rate invariant: [{strain_rate.data.min():.4e}, {strain_rate.data.max():.4e}]")

# %% [markdown]
# ## 9. Reference Stress (No Faults)
#
# Solve the same BCs with isotropic (uniform) viscosity to get the
# background strain rate.  The **anomaly** (fault minus reference) reveals
# where faults concentrate or shadow deformation.

# %%
v_ref = uw.discretisation.MeshVariable("v_ref", mesh, mesh.dim, degree=2,
                                        varsymbol=r"\mathbf{v}_{ref}", units="cm/yr")
p_ref = uw.discretisation.MeshVariable("p_ref", mesh, 1, degree=1,
                                        varsymbol="p_{ref}", units="MPa")

stokes_ref = uw.systems.Stokes(mesh, velocityField=v_ref, pressureField=p_ref)
stokes_ref.petsc_options["snes_monitor"] = None
stokes_ref.petsc_options["ksp_monitor"] = None
stokes_ref.petsc_options["snes_type"] = "newtonls"
stokes_ref.petsc_options["ksp_type"] = "fgmres"
stokes_ref.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes_ref.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")

# Isotropic: uniform viscosity (same eta_0, no fault weakness)
stokes_ref.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes_ref.constitutive_model.Parameters.shear_viscosity_0 = eta_0
stokes_ref.saddle_preconditioner = 1.0 / uw.non_dimensionalise(eta_0)

# Same BCs as the fault solve
stokes_ref.add_natural_bc(penalty * unit_down.dot(v_ref.sym) * unit_down, "Bottom")
stokes_ref.add_natural_bc(penalty * (unit_east.dot(v_ref.sym) + V0_nd) * unit_east, "East")
stokes_ref.add_natural_bc(penalty * (unit_east.dot(v_ref.sym) - V0_nd) * unit_east, "West")
stokes_ref.add_natural_bc(penalty * unit_north.dot(v_ref.sym) * unit_north, "North")
stokes_ref.add_natural_bc(penalty * unit_north.dot(v_ref.sym) * unit_north, "South")

stokes_ref.solve(verbose=False)
print(f"Reference solve: |v_ref|_max = {np.abs(v_ref.data).max():.4f}")

# Project reference strain rate
strain_rate_ref = uw.discretisation.MeshVariable(
    "eps_ref", mesh, 1, degree=1, varsymbol=r"\dot\varepsilon_{ref}"
)
proj_ref = uw.systems.Projection(mesh, strain_rate_ref)
proj_ref.uw_function = stokes_ref.constitutive_model.Unknowns.Einv2
proj_ref.smoothing = 1.0e-6
proj_ref.solve()

print(f"Reference strain rate: [{strain_rate_ref.data.min():.4e}, {strain_rate_ref.data.max():.4e}]")

# %% [markdown]
# ## 10. Permeability from Strain-Rate Anomaly
#
# Permeability is only modified **near faults** — the fault influence weight
# (Gaussian, ~1 on faults, ~0 far away) masks the strain-rate delta so
# background regions keep unit permeability.
#
# Within the fault zone, positive delta (strain concentration) enhances
# permeability; negative delta (strain shadow) reduces it.  The log₁₀(k)
# is blended by the fault weight for a smooth transition.

# %%
permeability = uw.discretisation.MeshVariable(
    "perm", mesh, 1, degree=1, varsymbol=r"\kappa"
)

delta = strain_rate.data[:, 0] - strain_rate_ref.data[:, 0]
w = fault_weight.data[:, 0]

# Only modify permeability where faults have significant influence
WEIGHT_THRESHOLD = 0.1
near_fault = w > WEIGHT_THRESHOLD

delta_near = delta[near_fault]
q25 = np.percentile(delta_near, 25)
q75 = np.percentile(delta_near, 75)

log_perm = np.zeros_like(delta)  # background = 10^0 = 1

# Only the tails: upper quartile enhanced, lower quartile reduced
log_perm[near_fault & (delta > q75)] = 1
log_perm[near_fault & (delta < q25)] = -1

# Blend by fault weight for smooth transition
permeability.data[:, 0] = np.power(10.0, log_perm * w)

print(f"Strain-rate delta: [{delta.min():.4e}, {delta.max():.4e}]")
print(f"Near-fault quartiles: q25={q25:.1f}, q75={q75:.1f}")
print(f"Permeability: [{permeability.data[:,0].min():.2e}, {permeability.data[:,0].max():.2e}]")
print(f"  Enhanced (k>1): {(permeability.data[:,0] > 1.01).sum()}")
print(f"  Reduced  (k<1): {(permeability.data[:,0] < 0.99).sum()}")

# %% [markdown]
# ## 11. Darcy Flow
#
# Solve steady-state Darcy flow with the fault-derived permeability field.
# A pressure source at depth drives fluid upward; the surface is free-draining
# ($h = 0$).  Gravity acts radially inward (downward on the geographic mesh).

# %%
import sympy

h_darcy = uw.discretisation.MeshVariable(
    "h_darcy", mesh, 1, degree=2, varsymbol=r"h_d"
)
v_darcy = uw.discretisation.MeshVariable(
    "v_darcy", mesh, mesh.dim, degree=1, continuous=True, varsymbol=r"\mathbf{v}_d"
)

darcy = uw.systems.SteadyStateDarcy(mesh, h_Field=h_darcy, v_Field=v_darcy)
darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel
darcy.constitutive_model.Parameters.permeability = permeability.sym[0]

# Free-draining top surface
darcy.add_essential_bc(0.0, "Surface")

# Source at depth > 30 km — both sides carry units, JIT nondimensionalises
depth_sym = geo[2]  # = a_ellipsoid - r (unit-aware)
source_depth = uw.expression(r"d_{source}", uw.quantity(30, "km"), "source depth threshold")

darcy.f = sympy.Piecewise(
    (10.0, depth_sym > source_depth),
    (0.0, True),
)

# Gravity: s = unit_down (dimensionless, consistent with head formulation).
# At hydrostatic equilibrium grad(h) = unit_down, so v = -K(grad(h) - s) = 0.
# Overpressure from the source breaks equilibrium → drives flow through faults.
darcy.constitutive_model.Parameters.s = unit_down

darcy.petsc_options["snes_monitor"] = None
darcy.tolerance = 1.0e-3
darcy._v_projector.tolerance = 1.0e-3
darcy._v_projector.smoothing = 0.0

print("Solving Darcy flow...")
darcy.solve()
print(f"  |h|_max = {np.abs(h_darcy.data).max():.4e}")
print(f"  |v_darcy|_max = {np.abs(v_darcy.data).max():.4e}")

# %% [markdown]
# ## 12. Visualization
#
# Overview plots: strain rate anomaly, permeability field, and Darcy velocity
# with fault surfaces overlaid.

# %%
uw.timing.print_summary()

# %%
import pyvista as pv
import underworld3.visualisation as vis

pvmesh = vis.mesh_to_pv_mesh(mesh)

pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)
pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v.sym.dot(v.sym))
pvmesh.point_data["eps"] = vis.scalar_fn_to_pv_points(pvmesh, strain_rate.sym)
pvmesh.point_data["eps_ref"] = vis.scalar_fn_to_pv_points(pvmesh, strain_rate_ref.sym)
pvmesh.point_data["delta_eps"] = pvmesh.point_data["eps"] - pvmesh.point_data["eps_ref"]
pvmesh.point_data["log_perm"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.log(permeability.sym[0], 10))
pvmesh.point_data["h_darcy"] = vis.scalar_fn_to_pv_points(pvmesh, h_darcy.sym)
pvmesh.point_data["Vd"] = vis.vector_fn_to_pv_points(pvmesh, v_darcy.sym)
pvmesh.point_data["Vd_mag"] = np.linalg.norm(pvmesh.point_data["Vd"], axis=1)

if RHEOLOGY != "isoviscous":
    pvmesh.point_data["fault_dist"] = vis.scalar_fn_to_pv_points(pvmesh, nearest_dist.sym)
    pvmesh.point_data["fault_w"] = vis.scalar_fn_to_pv_points(pvmesh, fault_weight.sym)

# Fault surface meshes
fault_colors = [
    "Red", "Orange", "Blue", "Green", "Purple",
    "Cyan", "Yellow", "Magenta", "Lime", "Pink",
]

# %%
# Darcy flow + fault overlay
pl = pv.Plotter(window_size=(1000, 800))

pl.add_mesh(pvmesh, scalars="Vd_mag", cmap="viridis", opacity=0.3, show_edges=False)
pl.add_arrows(pvmesh.points, pvmesh.point_data["Vd"], mag=5e-3, color="Blue", opacity=0.5)

for i, (sid, surface) in enumerate(fault_surfaces.items()):
    if surface.pv_mesh is not None:
        pl.add_mesh(
            surface.pv_mesh,
            style="surface",
            color=fault_colors[i % len(fault_colors)],
            opacity=0.7,
            show_edges=True,
        )

pl.add_title("Darcy Flow with Fault-Controlled Permeability")
pl.show()

# %%
# Permeability field
pl2 = pv.Plotter(window_size=(1000, 800))

pl2.add_mesh(pvmesh, scalars="log_perm", cmap="RdYlBu_r", opacity=0.5, show_edges=False,
             scalar_bar_args={"title": "log10(k)"})

for i, (sid, surface) in enumerate(fault_surfaces.items()):
    if surface.pv_mesh is not None:
        pl2.add_mesh(
            surface.pv_mesh,
            style="surface",
            color=fault_colors[i % len(fault_colors)],
            opacity=0.7,
            show_edges=True,
        )

pl2.add_title("Permeability (log10 scale)")
pl2.show()

# %%
