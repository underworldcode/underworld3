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
# # H2Ex Pipeline — Compute and Save
#
# Runs the full fault-controlled flow pipeline and saves all fields to VTK
# for interactive inspection in the companion visualization notebook.
#
# **Stages**: CSV load → mesh → faults → adapt → Stokes (TI) → strain rate
# → reference Stokes → permeability → Darcy flow
#
# **Output**: `output/h2ex_results.vtk` + individual fault surface VTK files
#
# **Runtime**: ~5 minutes on a single core (139k Stokes DOFs)

# %% [markdown]
# ## 1. Setup

# %%
import numpy as np
import csv
import sympy
import time
from pathlib import Path
import underworld3 as uw

uw.reset_default_model()

model = uw.Model()
model.set_reference_quantities(
    length=uw.quantity(1000, "km"),
    viscosity=uw.quantity(1e21, "Pa.s"),
    diffusivity=uw.quantity(1e-6, "m**2/s"),
    verbose=False,
)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

uw.timing.start()
t_start = time.time()

# %% [markdown]
# ## 2. Load Fault Traces from CSV

# %%
_csv_name = "CombinedInferredFaults_2025_04_22.csv"
_search_paths = [
    Path("Structures") / _csv_name,
    Path.home() / "Library/CloudStorage/Box-Box/+Claude/H2Ex-Data/Structures" / _csv_name,
    Path.home() / "Box/+Claude/H2Ex-Data/Structures" / _csv_name,
]
_csv_path = next((p for p in _search_paths if p.exists()), None)
if _csv_path is None:
    raise FileNotFoundError(
        f"Cannot find {_csv_name} in any of:\n" +
        "\n".join(f"  {p}" for p in _search_paths)
    )

_rows = []
with open(_csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        _rows.append(row)

fault_lon   = np.array([float(r["Lon"]) for r in _rows])
fault_lat   = np.array([float(r["Lat"]) for r in _rows])
fault_depth = np.array([float(r["Depth"]) for r in _rows])
fault_dip   = np.array([float(r["Dip Angle at depth"]) for r in _rows])
fault_name  = np.array([float(r["Name"]) for r in _rows])
fault_desc  = [r["Description"].strip() for r in _rows]
unique_fault_ids = np.unique(fault_name)

print(f"Loaded {len(fault_lon)} trace points, {len(unique_fault_ids)} segments from {_csv_path.name}")

# %% [markdown]
# ## 3. Create Geographic Mesh

# %%
mesh = uw.meshing.RegionalGeographicBox(
    lon_range=(135.5, 137.5),
    lat_range=(-34.5, -33.0),
    depth_range=(uw.quantity(0, "km"), uw.quantity(50, "km")),
    ellipsoid="WGS84",
    numElements=(16, 16, 8),
    simplex=True,
)
print(f"Initial mesh: {mesh.X.coords.shape[0]} P1 nodes")

# %% [markdown]
# ## 4. Build Fault Surfaces

# %%
fault_surfaces = {}

for fid in unique_fault_ids:
    mask = fault_name == fid
    if mask.sum() < 2:
        continue

    trace = np.column_stack([fault_lon[mask], fault_lat[mask]])
    depth_km = float(fault_depth[mask][0])
    segment_dip = float(fault_dip[mask][0])
    description = fault_desc[np.argmax(mask)]

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
    fault_surfaces[fid] = s

faults = uw.meshing.SurfaceCollection()
for s in fault_surfaces.values():
    faults.add(s)

print(f"Created {len(fault_surfaces)} fault surfaces")

# %% [markdown]
# ## 5. Adapt Mesh Near Faults

# %%
combined_metric = faults.refinement_metric(
    mesh,
    h_near=uw.quantity(2.0, "km"),
    h_far=uw.quantity(20.0, "km"),
    width=uw.quantity(10.0, "km"),
    profile="smoothstep",
)

n_before = mesh.X.coords.shape[0]
mesh.adapt(combined_metric, verbose=False)
n_after = mesh.X.coords.shape[0]
print(f"Adapted: {n_before} -> {n_after} P1 nodes")

# %% [markdown]
# ## 6. Stokes Solve (Transverse Isotropic)

# %%
v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2,
                                    varsymbol=r"\mathbf{v}", units="cm/yr")
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1,
                                    varsymbol="p", units="MPa")

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")

# Anisotropic rheology
eta_0 = uw.expression(r"\eta_0", uw.quantity(1e21, "Pa.s"), "reference viscosity")
eta_1_weak = uw.expression(r"\eta_1", uw.quantity(0.1e21, "Pa.s"), "weak fault viscosity")
fault_width = uw.quantity(10.0, "km")

fields = faults.compute_nearest_fields(mesh, fault_width=fault_width)
fault_normal = fields["normal"]
fault_weight = fields["weight"]
nearest_dist = fields["distance"]
fault_id_var = fields["id"]

eta_1_expr = eta_0 - (eta_0 - eta_1_weak) * fault_weight.sym[0]

stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_0
stokes.constitutive_model.Parameters.shear_viscosity_1 = eta_1_expr
stokes.constitutive_model.Parameters.director = fault_normal.sym
stokes.saddle_preconditioner = 1.0 / uw.non_dimensionalise(eta_0)

# Boundary conditions (geographic basis)
geo = mesh.CoordinateSystem.geo
unit_down = geo.unit_down
unit_east = geo.unit_east
unit_north = geo.unit_north

V0 = uw.expression(r"V_0", uw.quantity(1, "cm/yr"), "driving velocity")
V0_nd = uw.non_dimensionalise(V0)
penalty = 1.0e6 * uw.non_dimensionalise(eta_0)

stokes.add_natural_bc(penalty * unit_down.dot(v.sym) * unit_down, "Bottom")
stokes.add_natural_bc(penalty * (unit_east.dot(v.sym) + V0_nd) * unit_east, "East")
stokes.add_natural_bc(penalty * (unit_east.dot(v.sym) - V0_nd) * unit_east, "West")
stokes.add_natural_bc(penalty * unit_north.dot(v.sym) * unit_north, "North")
stokes.add_natural_bc(penalty * unit_north.dot(v.sym) * unit_north, "South")

t0 = time.time()
stokes.solve(verbose=False)
print(f"Stokes TI: {time.time()-t0:.0f}s, v DOFs={v.coords.shape[0]*mesh.dim}")

# %% [markdown]
# ## 7. Strain Rate Projection

# %%
strain_rate = uw.discretisation.MeshVariable("eps", mesh, 1, degree=1,
                                              varsymbol=r"\dot\varepsilon")
proj = uw.systems.Projection(mesh, strain_rate)
proj.uw_function = stokes.constitutive_model.Unknowns.Einv2
proj.smoothing = 1.0e-6
proj.solve()
print(f"Strain rate: [{strain_rate.data.min():.2e}, {strain_rate.data.max():.2e}]")

# %% [markdown]
# ## 8. Reference Stress (Isotropic)

# %%
v_ref = uw.discretisation.MeshVariable("v_ref", mesh, mesh.dim, degree=2, units="cm/yr")
p_ref = uw.discretisation.MeshVariable("p_ref", mesh, 1, degree=1, units="MPa")

stokes_ref = uw.systems.Stokes(mesh, velocityField=v_ref, pressureField=p_ref)
stokes_ref.petsc_options["snes_type"] = "newtonls"
stokes_ref.petsc_options["ksp_type"] = "fgmres"
stokes_ref.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes_ref.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")

stokes_ref.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes_ref.constitutive_model.Parameters.shear_viscosity_0 = eta_0
stokes_ref.saddle_preconditioner = 1.0 / uw.non_dimensionalise(eta_0)

stokes_ref.add_natural_bc(penalty * unit_down.dot(v_ref.sym) * unit_down, "Bottom")
stokes_ref.add_natural_bc(penalty * (unit_east.dot(v_ref.sym) + V0_nd) * unit_east, "East")
stokes_ref.add_natural_bc(penalty * (unit_east.dot(v_ref.sym) - V0_nd) * unit_east, "West")
stokes_ref.add_natural_bc(penalty * unit_north.dot(v_ref.sym) * unit_north, "North")
stokes_ref.add_natural_bc(penalty * unit_north.dot(v_ref.sym) * unit_north, "South")

t0 = time.time()
stokes_ref.solve(verbose=False)
print(f"Stokes ref: {time.time()-t0:.0f}s")

strain_rate_ref = uw.discretisation.MeshVariable("eps_ref", mesh, 1, degree=1,
                                                   varsymbol=r"\dot\varepsilon_{ref}")
proj_ref = uw.systems.Projection(mesh, strain_rate_ref)
proj_ref.uw_function = stokes_ref.constitutive_model.Unknowns.Einv2
proj_ref.smoothing = 1.0e-6
proj_ref.solve()
print(f"Ref strain rate: [{strain_rate_ref.data.min():.2e}, {strain_rate_ref.data.max():.2e}]")

# %% [markdown]
# ## 9. Permeability from Strain-Rate Anomaly
#
# Permeability is only modified **near faults** — the fault influence weight
# (Gaussian, ~1 on faults, ~0 far away) masks the strain-rate delta so
# background regions keep unit permeability.
#
# Within the fault zone, positive delta (strain concentration) enhances
# permeability; negative delta (strain shadow) reduces it.  The thresholds
# are heuristic — roughly upper/lower quartile of the near-fault delta.

# %%
delta = strain_rate.data[:, 0] - strain_rate_ref.data[:, 0]
w = fault_weight.data[:, 0]

# Only consider delta where faults have significant influence
WEIGHT_THRESHOLD = 0.1  # only modify perm where fault influence > 10%
near_fault = w > WEIGHT_THRESHOLD

# Heuristic thresholds from the near-fault delta distribution
delta_near = delta[near_fault]
print(f"Near-fault delta: [{delta_near.min():.2e}, {delta_near.max():.2e}], "
      f"median={np.median(delta_near):.2e}")

permeability = uw.discretisation.MeshVariable("perm", mesh, 1, degree=1,
                                               varsymbol=r"\kappa")

# Quartile thresholds from near-fault delta
q25 = np.percentile(delta_near, 25)
q75 = np.percentile(delta_near, 75)
print(f"  25th percentile: {q25:.1f}, 75th percentile: {q75:.1f}")

# Start at background (log_perm = 0 → k = 1)
log_perm = np.zeros_like(delta)

# Only the tails get modified — upper quartile enhanced, lower quartile reduced
log_perm[near_fault & (delta > q75)] = 1
log_perm[near_fault & (delta < q25)] = -1

# Weight-blended: smooth transition between background and fault-zone perm
permeability.data[:, 0] = np.power(10.0, log_perm * w)

print(f"Permeability: [{permeability.data[:,0].min():.2e}, {permeability.data[:,0].max():.2e}]")
print(f"  Near fault: {near_fault.sum()} nodes ({100*near_fault.sum()/len(delta):.1f}%)")
print(f"  Enhanced (k>1): {(permeability.data[:,0] > 1.01).sum()}")
print(f"  Reduced  (k<1): {(permeability.data[:,0] < 0.99).sum()}")
print(f"  Background:     {((permeability.data[:,0] >= 0.99) & (permeability.data[:,0] <= 1.01)).sum()}")

# %% [markdown]
# ## 10. Darcy Flow

# %%
h_darcy = uw.discretisation.MeshVariable("h_darcy", mesh, 1, degree=2, varsymbol=r"h_d")
v_darcy = uw.discretisation.MeshVariable("v_darcy", mesh, mesh.dim, degree=1,
                                          continuous=True, varsymbol=r"\mathbf{v}_d")

darcy = uw.systems.SteadyStateDarcy(mesh, h_Field=h_darcy, v_Field=v_darcy)
darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel
darcy.constitutive_model.Parameters.permeability = permeability.sym[0]

darcy.add_essential_bc(0.0, "Surface")

# Source at depth > 30 km — both sides carry units, JIT nondimensionalises
depth_sym = geo[2]  # = a_ellipsoid - r (unit-aware)
source_depth = uw.expression(r"d_{source}", uw.quantity(30, "km"), "source depth threshold")
darcy.f = sympy.Piecewise(
    (10.0, depth_sym > source_depth),
    (0.0, True),
)

# Gravity: s = g_scale * unit_down (head formulation).
# At full strength (g_scale=1), gravity dominates over 50 km.
# Reduced gravity lets the source overpressure drive upward flow
# through high-permeability fault zones.
G_SCALE = 0.01
darcy.constitutive_model.Parameters.s = G_SCALE * unit_down
darcy.tolerance = 1.0e-3
darcy._v_projector.tolerance = 1.0e-3
darcy._v_projector.smoothing = 0.0

t0 = time.time()
darcy.solve()
print(f"Darcy: {time.time()-t0:.1f}s")
print(f"  |h|_max = {np.abs(h_darcy.data).max():.4e}")
print(f"  |v_darcy|_max = {np.abs(v_darcy.data).max():.4e}")

# %% [markdown]
# ## 11. Tracer Advection — Flow Accumulation
#
# Seed particles near the base and advect upward through the Darcy
# velocity field.  Particles reaching the surface are captured; arrival
# density maps where fluid preferentially accumulates.
#
# **Key**: all coordinates are `UnitAwareArray` (dimensional, meters).
# `uw.function.evaluate` nondimensionalises internally.  Never strip
# units with `np.array()`.

# %%
from underworld3.utilities.unit_aware_array import UnitAwareArray
from underworld3.coordinates import geographic_to_cartesian

# Mesh geometry — use the user-facing (dimensional) coords
mesh_coords = mesh.X.coords  # UnitAwareArray in meters
r_mesh = np.sqrt(np.sum(np.asarray(mesh_coords)**2, axis=1))
r_surface = r_mesh.max()
r_bottom = r_mesh.min()
depth_range_m = r_surface - r_bottom

# Seed tracers near the base on a regular lon/lat grid
TRACER_N = 80
SEED_DEPTH_KM = 40.0  # seed at 40 km depth

# Get nondimensional a, b for geographic_to_cartesian
ellipsoid = mesh.CoordinateSystem.ellipsoid
a_raw, b_raw = ellipsoid["a"], ellipsoid["b"]
if hasattr(a_raw, "to"):
    a_km = float(a_raw.to("km").magnitude)
    b_km = float(b_raw.to("km").magnitude)
else:
    a_km = float(a_raw)
    b_km = float(b_raw)

lon_min, lon_max = 135.5, 137.5
lat_min, lat_max = -34.5, -33.0
lons = np.linspace(lon_min, lon_max, TRACER_N)
lats = np.linspace(lat_min, lat_max, TRACER_N)
grid_lon, grid_lat = np.meshgrid(lons, lats)

# Convert to Cartesian in km, then wrap as UnitAwareArray in meters
x_km, y_km, z_km = geographic_to_cartesian(
    grid_lon.ravel(), grid_lat.ravel(), SEED_DEPTH_KM, a_km, b_km
)
tracers_km = np.column_stack([x_km, y_km, z_km])
tracers = UnitAwareArray(tracers_km * 1000.0, units="m")  # km → m
tracers_0 = tracers.copy()
n_tracers = len(tracers)

print(f"Seeded {n_tracers} tracers at {SEED_DEPTH_KM:.0f} km depth")

# %%
# Advect tracers upward in v_darcy (midpoint integration).
# evaluate() handles unit conversion; velocity is returned in m/s.
# Timestep in seconds for consistent arithmetic.

# Estimate timestep from max velocity (nondimensional → m/s)
v_data_raw = np.asarray(v_darcy.data)
v_ref = float(uw.dimensionalise(1.0, "m/s").magnitude)
max_speed_ms = np.linalg.norm(v_data_raw, axis=1).max() * v_ref

# dt such that max displacement is ~2% of depth range
dt_s = 0.02 * depth_range_m / max(max_speed_ms, 1e-30)

MAX_STEPS = 500
active = np.ones(n_tracers, dtype=bool)
arrived = np.zeros(n_tracers, dtype=bool)

print(f"dt = {dt_s:.2e} s ({dt_s/3.156e7:.2e} yr), max_speed = {max_speed_ms:.2e} m/s")
print(f"Advecting (max {MAX_STEPS} steps)...")

t0 = time.time()
for step in range(MAX_STEPS):
    if not active.any():
        break

    # evaluate returns dimensional velocity (UnitAwareArray in m/s)
    active_pts = UnitAwareArray(np.asarray(tracers[active]), units="m")
    v1 = np.asarray(uw.function.evaluate(v_darcy.sym, active_pts)).reshape(-1, 3)

    # Midpoint
    mid_vals = np.asarray(tracers[active]) + 0.5 * dt_s * v1
    mid_pts = UnitAwareArray(mid_vals, units="m")
    v2 = np.asarray(uw.function.evaluate(v_darcy.sym, mid_pts)).reshape(-1, 3)

    # Update positions (meters)
    new_pos = np.asarray(tracers[active]) + dt_s * v2
    tracers[active] = new_pos

    # Check for surface arrival
    r_tracers = np.sqrt(np.sum(np.asarray(tracers[active])**2, axis=1))
    hit_surface = r_tracers >= r_surface

    active_idx = np.where(active)[0]
    for j in np.where(hit_surface)[0]:
        idx = active_idx[j]
        pos = np.asarray(tracers[idx])
        tracers[idx] = pos * (r_surface / r_tracers[j])
        arrived[idx] = True
        active[idx] = False

    if step % 50 == 0:
        print(f"  step {step}: {active.sum()} active, {arrived.sum()} arrived")

print(f"Advection done: {time.time()-t0:.1f}s, {step+1} steps")
print(f"  Arrived: {arrived.sum()} / {n_tracers} ({100*arrived.sum()/n_tracers:.1f}%)")
print(f"  Still active: {active.sum()}")

# %%
# Surface accumulation density via KDTree binning

surface_mask = r_mesh > r_surface - 0.01 * depth_range_m
surface_pts = np.asarray(mesh_coords[surface_mask])
arrival_pts = np.asarray(tracers[arrived])

if arrived.sum() > 10:
    from scipy.spatial import cKDTree

    tree = cKDTree(surface_pts)
    dists, indices = tree.query(arrival_pts)

    accumulation = np.zeros(len(surface_pts))
    for idx in indices:
        accumulation[idx] += 1
    accumulation /= max(accumulation.max(), 1)

    print(f"Surface accumulation: {(accumulation > 0).sum()} / {len(surface_pts)} nodes with arrivals")
else:
    accumulation = np.zeros(max(1, surface_mask.sum()))
    print("Too few arrivals for accumulation map")

# Save tracer data
np.savez(
    str(OUTPUT_DIR / "tracers.npz"),
    tracers_0=np.asarray(tracers_0),
    tracers_final=np.asarray(tracers),
    arrived=arrived,
    active=active,
    surface_pts=surface_pts,
    accumulation=accumulation,
)
print(f"Saved tracer data to {OUTPUT_DIR / 'tracers.npz'}")

# %% [markdown]
# ## 12. Save Results to VTK
#
# Write all computed fields to a single VTK file for visualization.
# Fault surfaces saved as separate VTK files.

# %%
import pyvista as pv
import underworld3.visualisation as vis

pvmesh = vis.mesh_to_pv_mesh(mesh)

# Stokes fields
pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)
pvmesh.point_data["Vmag"] = np.linalg.norm(pvmesh.point_data["V"], axis=1)
pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p.sym)

# Strain rates and anomaly
pvmesh.point_data["eps"] = vis.scalar_fn_to_pv_points(pvmesh, strain_rate.sym)
pvmesh.point_data["eps_ref"] = vis.scalar_fn_to_pv_points(pvmesh, strain_rate_ref.sym)
pvmesh.point_data["delta_eps"] = pvmesh.point_data["eps"] - pvmesh.point_data["eps_ref"]

# Permeability
pvmesh.point_data["log_perm"] = np.log10(np.maximum(
    vis.scalar_fn_to_pv_points(pvmesh, permeability.sym), 1e-10
)).ravel()

# Darcy fields
pvmesh.point_data["h_darcy"] = vis.scalar_fn_to_pv_points(pvmesh, h_darcy.sym)
pvmesh.point_data["Vd"] = vis.vector_fn_to_pv_points(pvmesh, v_darcy.sym)
pvmesh.point_data["Vd_mag"] = np.linalg.norm(pvmesh.point_data["Vd"], axis=1)

# Fault proximity
pvmesh.point_data["fault_dist"] = vis.scalar_fn_to_pv_points(pvmesh, nearest_dist.sym)
pvmesh.point_data["fault_weight"] = vis.scalar_fn_to_pv_points(pvmesh, fault_weight.sym)
pvmesh.point_data["fault_id"] = vis.scalar_fn_to_pv_points(pvmesh, fault_id_var.sym)

# Geographic coordinates for cross-section slicing
pvmesh.point_data["geo_lon"] = vis.scalar_fn_to_pv_points(pvmesh, geo[0])
pvmesh.point_data["geo_lat"] = vis.scalar_fn_to_pv_points(pvmesh, geo[1])
pvmesh.point_data["geo_depth"] = vis.scalar_fn_to_pv_points(pvmesh, geo[2])

pvmesh.save(str(OUTPUT_DIR / "h2ex_results.vtk"))
print(f"Saved mesh with {len(pvmesh.point_data)} fields to {OUTPUT_DIR / 'h2ex_results.vtk'}")

# Save fault surfaces
for fid, surface in fault_surfaces.items():
    if surface.pv_mesh is not None:
        surface.pv_mesh.save(str(OUTPUT_DIR / f"fault_{fid}.vtk"))

print(f"Saved {len(fault_surfaces)} fault surface VTK files")

# %%
elapsed = time.time() - t_start
print(f"\nTotal pipeline time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
uw.timing.print_summary()

# %%
