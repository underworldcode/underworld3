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
# # W3: Darcy Flow
#
# Loads the adapted mesh (shared) and permeability field (from W2),
# solves steady-state Darcy flow with a depth source and gravity.
#
# **Requires**: `adapted_mesh` (shared), `permeability` (run)
#
# **Produces**: `darcy_head`, `darcy_velocity`
#
# **Next**: `W4-Tracers.py`

# %% [markdown]
# ## Configuration

# %%
STRESS_AZIMUTH = 0  # must match W2
RES = 16             # must match W1
G_SCALE = 0.01      # gravity scaling (head formulation)

# %% [markdown]
# ## Workflow Status

# %%
import numpy as np
import sympy
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

print(f"=== H2Ex Workflow: Stage 3 (Darcy Flow) ===")
print(f"Run: azimuth={STRESS_AZIMUTH}, g_scale={G_SCALE}")
print()

missing = []
print("Dependencies:")
for dep, store in [("adapted_mesh", shared), ("permeability", run)]:
    ok = store.exists(dep)
    source = "shared" if store is shared else f"azimuth_{STRESS_AZIMUTH:03d}"
    hint = "W1-MeshAndFaults.py" if store is shared else "W2-Stokes.py"
    status = "ready" if ok else f"MISSING -> run {hint}"
    print(f"  {dep} ({source}): {status}")
    if not ok:
        missing.append(dep)

if missing:
    raise RuntimeError(f"Missing products: {missing}")

print("\nThis stage products:")
for product in ["darcy_head", "darcy_velocity"]:
    status = "ready" if run.exists(product) else "not built"
    print(f"  {product}: {status}")

# %% [markdown]
# ## 1. Load Mesh and Permeability from Checkpoints

# %%
t0 = time.time()
mesh = shared.load("adapted_mesh")
print(f"Mesh: {mesh.X.coords.shape[0]} P1 nodes")

permeability = uw.discretisation.MeshVariable("perm", mesh, 1, degree=1,
                                               varsymbol=r"\kappa")
run.load("permeability", mesh_variable=permeability)
print(f"Permeability: [{permeability.data[:,0].min():.2e}, {permeability.data[:,0].max():.2e}]")
print(f"Loaded in {time.time()-t0:.1f}s")

# %% [markdown]
# ## 2. Darcy Solve

# %%
geo = mesh.CoordinateSystem.geo
unit_down = geo.unit_down

h_darcy = uw.discretisation.MeshVariable("h_darcy", mesh, 1, degree=2,
                                         varsymbol=r"h_d", units="m")
v_darcy = uw.discretisation.MeshVariable("v_darcy", mesh, mesh.dim, degree=1,
                                          continuous=True, varsymbol=r"\mathbf{v}_d",
                                          units="m/s")

darcy = uw.systems.SteadyStateDarcy(mesh, h_Field=h_darcy, v_Field=v_darcy)
darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel
darcy.constitutive_model.Parameters.permeability = permeability.sym[0]

darcy.add_essential_bc(0.0, "Surface")

# Source at depth > 30 km
depth_sym = geo[2]
source_depth = uw.expression(r"d_{source}", uw.quantity(30, "km"), "source depth threshold")
darcy.f = sympy.Piecewise(
    (10.0, depth_sym > source_depth),
    (0.0, True),
)

# Gravity (head formulation)
darcy.constitutive_model.Parameters.s = G_SCALE * unit_down

darcy.tolerance = 1.0e-3
darcy._v_projector.tolerance = 1.0e-3
darcy._v_projector.smoothing = 0.0

t0 = time.time()
darcy.solve()
print(f"Darcy: {time.time()-t0:.1f}s")
print(f"  |h|_max = {np.abs(h_darcy.data).max():.4e}")
print(f"  |v|_max = {np.abs(v_darcy.data).max():.4e}")

# %% [markdown]
# ## 3. Checkpoint

# %%
run.save("darcy_head", h_darcy)
run.save("darcy_velocity", v_darcy)

print()
run.list()

# %% [markdown]
# ## Done
#
# **Next step**: `W4-Tracers.py` (same `STRESS_AZIMUTH`)

# %%
print(f"\n=== Stage 3 Complete (azimuth={STRESS_AZIMUTH}) ===")
print(f"\nNext: W4-Tracers.py (set STRESS_AZIMUTH={STRESS_AZIMUTH})")

# %%
