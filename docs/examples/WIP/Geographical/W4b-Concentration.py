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
# # W4b: Concentration Field (Advection-Diffusion)
#
# Advect a concentration field from the base upward through the Darcy
# velocity field with weak diffusion.  At steady state, the near-surface
# concentration shows where flow from depth preferentially accumulates.
#
# Uses the semi-Lagrangian AdvDiffusion solver (no CFL constraint on dt).
#
# **Requires**: `adapted_mesh` (shared), `darcy_velocity` (run)
#
# **Produces**: `concentration` (steady-state C field)

# %% [markdown]
# ## Configuration

# %%
STRESS_AZIMUTH = 0
RES = 16       # must match W1
N_STEPS = 100

# %% [markdown]
# ## Workflow Status

# %%
import numpy as np
import time
from pathlib import Path
import underworld3 as uw
from underworld3.workflows import WorkflowProducts

uw.reset_default_model()
model = uw.Model()
model.set_reference_quantities(
    length=uw.quantity(1000, "km"),
    viscosity=uw.quantity(1e21, "Pa.s"),
    diffusivity=uw.quantity(1e-6, "m**2/s"),
    verbose=False,
)

SHARED_DIR = Path(f"output/h2ex/res_{RES}/shared")
RUN_DIR = Path(f"output/h2ex/res_{RES}/azimuth_{STRESS_AZIMUTH:03d}")

shared = WorkflowProducts(products_dir=SHARED_DIR / "products")
run = WorkflowProducts(products_dir=RUN_DIR / "products")

print(f"=== H2Ex Workflow: Stage 4b (Concentration) ===")
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

# %% [markdown]
# ## 1. Load Mesh and Darcy Velocity

# %%
t0 = time.time()
mesh = shared.load("adapted_mesh")

v_darcy = uw.discretisation.MeshVariable("v_darcy", mesh, mesh.dim, degree=1,
                                          continuous=True, varsymbol=r"\mathbf{v}_d",
                                          units="m/s")
run.load("darcy_velocity", mesh_variable=v_darcy)

print(f"Mesh: {mesh.X.coords.shape[0]} P1 nodes")
print(f"|v_darcy|_max = {np.abs(v_darcy.data).max():.4e}")
print(f"Loaded in {time.time()-t0:.1f}s")

# %% [markdown]
# ## 2. Advection-Diffusion Solver
#
# Solve $\partial C/\partial t + \mathbf{v}\cdot\nabla C = \kappa \nabla^2 C$
#
# - $C = 1$ at Bottom (material source at depth)
# - $C = 0$ at Surface (drain — material removed on arrival)
# - Weak diffusion ($\kappa$) for stability; spatial patterns from advection
# - Semi-Lagrangian: no CFL constraint, large timesteps OK
#
# At steady state, high $C$ near the surface = where flow accumulates.

# %%
# All quantities carry units
# Diffusivity must be small enough that advection dominates (high Peclet).
# Reference diffusivity is 1e-6 m²/s; using 1e-9 gives Pe ~ 250.
kappa = uw.expression(r"\kappa_C", uw.quantity(1e-10, "m**2/s"), "tracer diffusivity")

# Concentration is dimensionless (a ratio, no units)
C = uw.discretisation.MeshVariable("C", mesh, 1, degree=1, varsymbol=r"C")

adv_diff = uw.systems.AdvDiffusion(
    mesh,
    u_Field=C,
    V_fn=v_darcy,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = kappa

adv_diff.add_essential_bc(1.0, "Bottom")
adv_diff.add_essential_bc(0.0, "Surface")

# Large timestep — semi-Lagrangian has no CFL constraint
# Transit time ~ depth / v ~ 50 km / (1e-12 m/s) ~ 1.6 Gyr
# Use ~50 Myr steps to reach steady state in ~5 steps
dt = uw.quantity(200, "Myr")

print(f"Diffusivity: {kappa}")
print(f"dt: {dt}")

# %% [markdown]
# ## 3. Evolve to Steady State

# %%
dm_coords = mesh.dm.getCoordinatesLocal().array.reshape(-1, 3)
r_dm = np.sqrt(np.sum(dm_coords**2, axis=1))
r_surface = r_dm.max()
depth_range = r_surface - r_dm.min()
near_surface = (r_dm > r_surface - 0.12 * depth_range) & (r_dm < r_surface - 0.05 * depth_range)

print(f"Stepping {N_STEPS} times ({dt} each)...")
print(f"Near-surface evaluation layer: {near_surface.sum()} nodes")
print(f"{'step':>6s}  {'C_max':>8s}  {'C_near_max':>10s}  {'C_near_mean':>11s}  {'dC_rms':>10s}  {'time':>6s}")

t0 = time.time()
C_prev = np.zeros(len(np.asarray(C.data).ravel()))

for step in range(N_STEPS):
    adv_diff.solve(timestep=dt, zero_init_guess=(step == 0))

    C_now = np.asarray(C.data).ravel()
    dC = np.sqrt(np.mean((C_now - C_prev)**2))
    C_prev = C_now.copy()

    if step % 5 == 0 or step == N_STEPS - 1:
        C_near = np.asarray(
            uw.function.evaluate(C.sym, dm_coords[near_surface], mode="fast")
        ).ravel()
        dt_wall = time.time() - t0
        print(f"{step:6d}  {C_now.max():8.3f}  {C_near.max():10.4f}  "
              f"{C_near.mean():11.4f}  {dC:10.2e}  {dt_wall:6.0f}s")

    if dC < 1e-5 and step > 5:
        print(f"Converged at step {step} (dC_rms={dC:.2e})")
        break

print(f"\nDone: {time.time()-t0:.0f}s, {step+1} steps")
print(f"C range: [{C_now.min():.4f}, {C_now.max():.4f}]")

# %% [markdown]
# ## 4. Checkpoint

# %%
run.save("concentration", C)
print()
run.list()

# %% [markdown]
# ## Done
#
# **Next step**: `W5-Visualise.py` — the near-surface concentration
# shows where fluid from depth preferentially accumulates through
# high-permeability fault zones.

# %%
print(f"\n=== Stage 4b Complete (azimuth={STRESS_AZIMUTH}) ===")
print(f"\nNext: W5-Visualise.py")

# %%
