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
# # W2: Stokes Stress and Permeability
#
# Loads the adapted mesh and fault surfaces from shared checkpoints,
# solves Stokes with transverse isotropic rheology (fault-controlled
# anisotropy), computes the reference strain rate, and derives a
# fault-localised permeability field.
#
# **Requires**: `adapted_mesh`, `fault_surfaces` (from W1)
#
# **Produces**: `strain_rate`, `strain_rate_ref`, `permeability`
#
# **Parameter**: `STRESS_AZIMUTH` — orientation of the driving stress (degrees)
#
# **Next**: `W3-DarcyFlow.py`

# %% [markdown]
# ## Configuration

# %%
STRESS_AZIMUTH = 0  # degrees — change for parameter study
RES = 16             # must match W1

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

print(f"=== H2Ex Workflow: Stage 2 (Stokes) ===")
print(f"Stress azimuth: {STRESS_AZIMUTH} degrees")
print(f"Run directory:  {RUN_DIR}")
print()

# Check dependencies
print("Dependencies (shared):")
missing = []
for dep in ["adapted_mesh", "fault_surfaces"]:
    ok = shared.exists(dep)
    status = "ready" if ok else "MISSING -> run W1-MeshAndFaults.py"
    print(f"  {dep}: {status}")
    if not ok:
        missing.append(dep)

if missing:
    raise RuntimeError(
        f"Missing shared products: {missing}. Run W1-MeshAndFaults.py first."
    )

print("\nThis stage products:")
for product in ["strain_rate", "strain_rate_ref", "permeability"]:
    status = "ready" if run.exists(product) else "not built"
    print(f"  {product}: {status}")

# %% [markdown]
# ## 1. Load Mesh and Faults from Checkpoint

# %%
t0 = time.time()
mesh = shared.load("adapted_mesh")
faults = shared.load("fault_surfaces", mesh=mesh)
print(f"Loaded mesh ({mesh.X.coords.shape[0]} nodes) + faults in {time.time()-t0:.1f}s")

# %% [markdown]
# ## 2. Stokes Solve (Transverse Isotropic)

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
# ## 3. Strain Rate Projection

# %%
strain_rate = uw.discretisation.MeshVariable("eps", mesh, 1, degree=1,
                                              varsymbol=r"\dot\varepsilon")
proj = uw.systems.Projection(mesh, strain_rate)
proj.uw_function = stokes.constitutive_model.Unknowns.Einv2
proj.smoothing = 1.0e-6
proj.solve()
print(f"Strain rate: [{strain_rate.data.min():.2e}, {strain_rate.data.max():.2e}]")

# %% [markdown]
# ## 4. Reference Stress (Isotropic)

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
# ## 5. Permeability from Strain-Rate Anomaly

# %%
delta = strain_rate.data[:, 0] - strain_rate_ref.data[:, 0]
w = fault_weight.data[:, 0]

WEIGHT_THRESHOLD = 0.1
near_fault = w > WEIGHT_THRESHOLD
delta_near = delta[near_fault]
q25 = np.percentile(delta_near, 25)
q75 = np.percentile(delta_near, 75)

permeability = uw.discretisation.MeshVariable("perm", mesh, 1, degree=1,
                                               varsymbol=r"\kappa")
log_perm = np.zeros_like(delta)
log_perm[near_fault & (delta > q75)] = 1
log_perm[near_fault & (delta < q25)] = -1
permeability.data[:, 0] = np.power(10.0, log_perm * w)

print(f"Permeability: [{permeability.data[:,0].min():.2e}, {permeability.data[:,0].max():.2e}]")
print(f"  Near-fault quartiles: q25={q25:.1f}, q75={q75:.1f}")
print(f"  Enhanced: {(permeability.data[:,0] > 1.01).sum()}, "
      f"Reduced: {(permeability.data[:,0] < 0.99).sum()}")

# %% [markdown]
# ## 6. Checkpoint

# %%
t0 = time.time()

run.save("stokes_velocity", v)
run.save("stokes_pressure", p)
run.save("strain_rate", strain_rate)
run.save("strain_rate_ref", strain_rate_ref)
run.save("permeability", permeability)

print(f"Checkpointed in {time.time()-t0:.1f}s")
print()
run.list()

# %% [markdown]
# ## Done
#
# **Next step**: `W3-DarcyFlow.py` (use same `STRESS_AZIMUTH`)

# %%
print(f"\n=== Stage 2 Complete (azimuth={STRESS_AZIMUTH}) ===")
print(f"Products saved to: {RUN_DIR / 'products'}")
print(f"\nNext: W3-DarcyFlow.py (set STRESS_AZIMUTH={STRESS_AZIMUTH})")

# %%
