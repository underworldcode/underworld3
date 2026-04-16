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
# # W4: Tracer Advection and Flow Accumulation
#
# Loads the adapted mesh and Darcy velocity from checkpoints, seeds
# particles near the base, advects them upward, and computes the
# surface accumulation density.
#
# **Requires**: `adapted_mesh` (shared), `darcy_velocity` (run)
#
# **Produces**: `tracers` (npz with positions, arrivals, accumulation)
#
# **Next**: `W5-Visualise.py`

# %% [markdown]
# ## Configuration

# %%
STRESS_AZIMUTH = 0    # must match W2/W3
TRACER_N = 22         # grid points per horizontal dimension (~500 tracers)
SEED_DEPTH_KM = 40.0  # seed depth in km
MAX_STEPS = 2000      # maximum advection steps

# %% [markdown]
# ## Workflow Status

# %%
import numpy as np
import time
from pathlib import Path
import underworld3 as uw
from underworld3.workflows import WorkflowProducts
from underworld3.coordinates import geographic_to_cartesian

uw.reset_default_model()
model = uw.Model()
model.set_reference_quantities(
    length=uw.quantity(1000, "km"),
    viscosity=uw.quantity(1e21, "Pa.s"),
    diffusivity=uw.quantity(1e-6, "m**2/s"),
    verbose=False,
)

SHARED_DIR = Path("output/h2ex/shared")
RUN_DIR = Path(f"output/h2ex/azimuth_{STRESS_AZIMUTH:03d}")

shared = WorkflowProducts(products_dir=SHARED_DIR / "products")
run = WorkflowProducts(products_dir=RUN_DIR / "products")

print(f"=== H2Ex Workflow: Stage 4 (Tracers) ===")
print(f"Run: azimuth={STRESS_AZIMUTH}")
print()

missing = []
print("Dependencies:")
for dep, store, hint in [
    ("adapted_mesh", shared, "W1-MeshAndFaults.py"),
    ("darcy_velocity", run, "W3-DarcyFlow.py"),
]:
    ok = store.exists(dep)
    status = "ready" if ok else f"MISSING -> run {hint}"
    print(f"  {dep}: {status}")
    if not ok:
        missing.append(dep)

if missing:
    raise RuntimeError(f"Missing products: {missing}")

print("\nThis stage products:")
status = "ready" if run.exists("tracers") else "not built"
print(f"  tracers: {status}")

# %% [markdown]
# ## 1. Load Mesh and Darcy Velocity

# %%
t0 = time.time()
mesh = shared.load("adapted_mesh")

v_darcy = uw.discretisation.MeshVariable("v_darcy", mesh, mesh.dim, degree=1,
                                          continuous=True, varsymbol=r"\mathbf{v}_d")
run.load("darcy_velocity", mesh_variable=v_darcy)

print(f"Mesh: {mesh.X.coords.shape[0]} P1 nodes")
print(f"|v_darcy|_max = {np.abs(v_darcy.data).max():.4e}")
print(f"Loaded in {time.time()-t0:.1f}s")

# %% [markdown]
# ## 2. Seed Tracers
#
# All tracer work is in **nondimensional DM coordinates**.
# `uw.function.evaluate` treats plain `ndarray` as nondimensional,
# returning nondimensional velocity.  This keeps all arithmetic
# consistent without unit conversions in the advection loop.

# %%
# Nondimensional ellipsoid for geographic_to_cartesian
ellipsoid = mesh.CoordinateSystem.ellipsoid
a_nd = float(uw.non_dimensionalise(ellipsoid["a"]))
b_nd = float(uw.non_dimensionalise(ellipsoid["b"]))

seed_depth_nd = float(uw.non_dimensionalise(uw.quantity(SEED_DEPTH_KM, "km")))

lon_min, lon_max = 135.5, 137.5
lat_min, lat_max = -34.5, -33.0
lons = np.linspace(lon_min, lon_max, TRACER_N)
lats = np.linspace(lat_min, lat_max, TRACER_N)
grid_lon, grid_lat = np.meshgrid(lons, lats)

# geographic_to_cartesian with nd ellipsoid + nd depth → nd Cartesian
x_nd, y_nd, z_nd = geographic_to_cartesian(
    grid_lon.ravel(), grid_lat.ravel(), seed_depth_nd, a_nd, b_nd
)
tracers = np.column_stack([x_nd, y_nd, z_nd])  # plain ndarray, nondimensional
tracers_0 = tracers.copy()
n_tracers = len(tracers)

# DM geometry for arrival detection (nondimensional)
dm_coords = mesh.dm.getCoordinatesLocal().array.reshape(-1, 3)
r_dm = np.sqrt(np.sum(dm_coords**2, axis=1))
r_surface_nd = r_dm.max()
r_bottom_nd = r_dm.min()
depth_range_nd = r_surface_nd - r_bottom_nd

print(f"Seeded {n_tracers} tracers at {SEED_DEPTH_KM:.0f} km depth (nd={seed_depth_nd:.4f})")
print(f"Surface r_nd={r_surface_nd:.4f}, depth range nd={depth_range_nd:.4f}")

# %% [markdown]
# ## 3. Advect Tracers
#
# Midpoint integration in `v_darcy`.  All coordinates and velocities
# are **nondimensional** — plain `ndarray` passed to `evaluate` is
# treated as nondimensional, returning nondimensional velocity.

# %%
# Timestep: 5% of depth range per step, scaled by max velocity at seed depth
# (global max is at surface, much faster than at depth)
v_at_seed = np.asarray(
    uw.function.evaluate(v_darcy.sym, tracers, mode="fast")
).reshape(-1, 3)
max_speed_nd = max(np.linalg.norm(v_at_seed, axis=1).max(), 1e-10)
dt_nd = 0.05 * depth_range_nd / max_speed_nd

active = np.ones(n_tracers, dtype=bool)
arrived = np.zeros(n_tracers, dtype=bool)

print(f"dt_nd = {dt_nd:.4e}, max_speed_nd = {max_speed_nd:.4f}")
print(f"Advecting {n_tracers} tracers (max {MAX_STEPS} steps)...")

t0 = time.time()
for step in range(MAX_STEPS):
    if not active.any():
        break

    n_active = active.sum()
    t_step = time.time()

    # Evaluate v_darcy at active positions using fast RBF mode
    active_pts = tracers[active]
    v1 = np.asarray(
        uw.function.evaluate(v_darcy.sym, active_pts, mode="fast")
    ).reshape(-1, 3)

    # Midpoint
    mid_pts = active_pts + 0.5 * dt_nd * v1
    v2 = np.asarray(
        uw.function.evaluate(v_darcy.sym, mid_pts, mode="fast")
    ).reshape(-1, 3)

    # Update positions
    active_idx = np.where(active)[0]
    tracers[active_idx] += dt_nd * v2

    # Check for surface arrival
    r_tracers = np.sqrt(np.sum(tracers[active_idx]**2, axis=1))
    hit_surface = r_tracers >= r_surface_nd

    for j in np.where(hit_surface)[0]:
        idx = active_idx[j]
        tracers[idx] *= r_surface_nd / r_tracers[j]
        arrived[idx] = True
        active[idx] = False

    dt_wall = time.time() - t_step
    if step % 10 == 0 or hit_surface.sum() > 0:
        r_min = r_tracers[~hit_surface].min() if (~hit_surface).any() else 0
        print(f"  step {step}: {n_active} active, {arrived.sum()} arrived, "
              f"r_min={r_min:.4f}/{r_surface_nd:.4f}, {dt_wall:.1f}s")

print(f"Advection done: {time.time()-t0:.1f}s, {step+1} steps")
print(f"  Arrived: {arrived.sum()} / {n_tracers} ({100*arrived.sum()/n_tracers:.1f}%)")
print(f"  Still active: {active.sum()}")

# %% [markdown]
# ## 4. Surface Accumulation

# %%
surface_mask = r_dm > r_surface_nd - 0.05 * depth_range_nd
surface_pts = dm_coords[surface_mask]
arrival_pts = tracers[arrived]

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

# %% [markdown]
# ## 5. Checkpoint

# %%
run.save("tracers", np.column_stack([
    np.asarray(tracers_0),
    np.asarray(tracers),
    arrived.astype(float)[:, None],
]))

# Also save accumulation as a separate product
np.savez(
    str(run.products_dir / "accumulation.npz"),
    surface_pts=surface_pts,
    accumulation=accumulation,
    arrived_count=arrived.sum(),
    total_tracers=n_tracers,
)
print(f"Saved tracer and accumulation data")
print()
run.list()

# %% [markdown]
# ## Done
#
# **Next step**: `W5-Visualise.py`

# %%
print(f"\n=== Stage 4 Complete (azimuth={STRESS_AZIMUTH}) ===")
print(f"\nNext: W5-Visualise.py")

# %%
