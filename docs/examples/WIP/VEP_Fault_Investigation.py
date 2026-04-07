# %% [markdown]
"""
# VEP Embedded Fault — Investigation Notebook

Horizontal fault at y=0.5 with gaussian influence function for tau_y.
Step through the model and visualise to find the source of SNES divergence.
"""

# %%
#|  echo: false  # Hide in html version

# This is required to fix pyvista 
# (visualisation) crashes in interactive notebooks (including on binder)

import nest_asyncio
nest_asyncio.apply()

# %%
import numpy as np
import sympy
import underworld3 as uw
from underworld3.systems import Stokes

# %% [markdown]
"""
## Parameters
"""

# %%
ETA = 1.0
MU = 1.0
TAU_Y_FAULT = 0.2
TAU_Y_BULK = 200.0
FAULT_WIDTH = 0.04
DT = 0.05
V_TOP = 0.5

# %% [markdown]
"""
## Mesh and variables
"""

# %%
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(16, 32), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=2)

v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2, vtype=uw.VarType.VECTOR)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True,
                                    vtype=uw.VarType.SCALAR)

# %% [markdown]
"""
## Fault surface and yield stress
"""

# %%
fault_points = np.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]])
fault = uw.meshing.Surface("fault", mesh, fault_points)
fault.discretize()

# Interpolate weakness (1/tau_y) — avoids steep gaussian ramp in tau_y
weakness = fault.influence_function(
    width=FAULT_WIDTH,
    value_near=1 / TAU_Y_FAULT,
    value_far=1 / TAU_Y_BULK,
    profile="gaussian",
)
tau_y_field = 1 / weakness

print(f"Fault: {fault.n_vertices} vertices")

# %% [markdown]
"""
## Solver setup
"""

# %%

# %%
stokes = Stokes(mesh, velocityField=v, pressureField=p)
# Create model instance BEFORE assigning — DFDt order is set at assignment time
cm = uw.constitutive_models.ViscoElasticPlasticFlowModel(stokes.Unknowns, order=1)
stokes.constitutive_model = cm

# %%

cm.Parameters.shear_viscosity_0 = ETA
cm.Parameters.shear_modulus = MU
cm.Parameters.yield_stress = tau_y_field
# yield_mode="smooth" is the default (corrected harmonic, no Min/Max)
cm.Parameters.shear_viscosity_min = ETA * 1.0e-2
cm.Parameters.strainrate_inv_II_min = 1.0e-5

stokes.saddle_preconditioner = 1 / cm.K
stokes.tolerance = 1.0e-4

stokes.add_essential_bc(sympy.Matrix([V_TOP, 0.0]), "Top")
stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
stokes.add_essential_bc((sympy.oo, 0.0), "Left")
stokes.add_essential_bc((sympy.oo, 0.0), "Right")
stokes.bodyforce = sympy.Matrix([0.0, 0.0])
stokes.petsc_options["ksp_type"] = "fgmres"
stokes.petsc_options["snes_force_iteration"] = None  # always do at least 1 SNES iteration

# %% [markdown]
"""
## Visualisation helpers
"""

# %%
def plot_state(step_num, title_extra=""):
    """Plot velocity, pressure, and stress for the current state."""

    if not uw.is_notebook():
        print("Skipping visualisation (not a notebook)")
        return

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v.sym)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p.sym)
    pvmesh.point_data["Vmag"] = np.linalg.norm(pvmesh.point_data["V"], axis=1)

    # Stress from tau
    tau_data = stokes.tau.data
    tau_coords = stokes.tau.coords

    # Velocity arrows
    velocity_points = vis.meshVariable_to_pv_cloud(v)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v.sym)

    pl = pv.Plotter(shape=(1, 3), window_size=(1500, 500))

    # Panel 1: Velocity magnitude
    pl.subplot(0, 0)
    pl.add_mesh(pvmesh, scalars="Vmag", cmap="viridis", show_edges=False)
    pl.add_arrows(velocity_points.points[::3], velocity_points.point_data["V"][::3],
                  mag=0.15, color="white")
    pl.add_title(f"Velocity (step {step_num}){title_extra}")
    pl.camera_position = "xy"

    # Panel 2: Pressure
    pl.subplot(0, 1)
    pl.add_mesh(pvmesh, scalars="P", cmap="coolwarm", show_edges=False)
    pl.add_title("Pressure")
    pl.camera_position = "xy"

    # Panel 3: Stress sigma_xy from tau
    tau_cloud = pv.PolyData(np.column_stack([tau_coords, np.zeros(len(tau_coords))]))
    tau_cloud.point_data["sigma_xy"] = tau_data[:, 2]
    pl.subplot(0, 2)
    pl.add_mesh(tau_cloud, scalars="sigma_xy", cmap="RdBu_r",
                point_size=8, render_points_as_spheres=True)
    pl.add_title("sigma_xy (from tau)")
    pl.camera_position = "xy"

    pl.show()

# %% [markdown]
"""
## Check tau_y field before solving
"""

# %%
# Evaluate tau_y at mesh nodes to verify the fault zone
tau_y_vals = uw.function.evaluate(tau_y_field, mesh.X.coords)
print(f"tau_y range: [{tau_y_vals.min():.4f}, {tau_y_vals.max():.4f}]")

if uw.is_notebook():
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["tau_y"] = tau_y_vals.flatten()

    pl = pv.Plotter(window_size=(600, 500))
    pl.add_mesh(pvmesh, scalars="tau_y", cmap="coolwarm", show_edges=True)
    pl.add_title("Yield stress field")
    pl.camera_position = "xy"
    pl.show()

# %% [markdown]
"""
## Step through the model

Run a few steps in the elastic regime, then step carefully through yield onset.
"""

# %%
import numpy as _np

for step in range(1 + int(2/DT)):
    # Inspect psi_star before solve
    s0 = stokes.DFDt.psi_star[0].data
    s1 = stokes.DFDt.psi_star[1].data if stokes.DFDt.order >= 2 else None
    s0_xy = f"[{s0[:,2].min():.4f}, {s0[:,2].max():.4f}]"
    s1_xy = f"[{s1[:,2].min():.4f}, {s1[:,2].max():.4f}]" if s1 is not None else "n/a"

    stokes.solve(timestep=DT, zero_init_guess=(step == 0))
    t = (step + 1) * DT
    reason = stokes.snes.getConvergedReason()
    its = stokes.snes.getIterationNumber()
    sigma_xy = stokes.tau.data[:, 2]

    flag = " ***" if reason < 0 else ""
    print(f"Step {step+1:3d}, t={t:.2f}: σ[{sigma_xy.min():.4f},{sigma_xy.max():.4f}] "
          f"σ*{s0_xy} σ**{s1_xy} SNES={reason} its={its}{flag}")


# %% [markdown]
"""
## Yield onset — step carefully
"""

# %%
# Continue stepping — yield should start around step 9-10
# for step in range(9, 15):
#     stokes.solve(timestep=DT, zero_init_guess=False)
#     t = (step + 1) * DT
#     reason = stokes.snes.getConvergedReason()
#     its = stokes.snes.getIterationNumber()
#     sigma_xy = stokes.tau.data[:, 2]
#     print(f"Step {step+1}, t={t:.2f}: sigma_xy [{sigma_xy.min():.4f}, {sigma_xy.max():.4f}], "
#           f"SNES={reason}, its={its}")

#     if reason < 0:
#         print(f"  *** DIVERGED at step {step+1} ***")
#         plot_state(step + 1, f" (DIVERGED, SNES={reason})")
#         break

# %% [markdown]
"""
## Continue past divergence
"""

# %%
# # Keep going to see if the solution stabilises
# for step in range(15, 25):
#     stokes.solve(timestep=DT, zero_init_guess=False)
#     t = (step + 1) * DT
#     reason = stokes.snes.getConvergedReason()
#     its = stokes.snes.getIterationNumber()
#     sigma_xy = stokes.tau.data[:, 2]
#     print(f"Step {step+1}, t={t:.2f}: sigma_xy [{sigma_xy.min():.4f}, {sigma_xy.max():.4f}], SNES={reason}, its={its}")

plot_state(25, " (final)")

# %% [markdown]
"""
## Stress profile across fault
"""

# %%
import matplotlib
if not uw.is_notebook():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

coords = stokes.tau.coords
sd = stokes.tau.data
x_coords = coords[:, 0]
y_coords = coords[:, 1]

# Get nodes near x=0.5
near_centre = np.abs(x_coords - 0.5) < 0.05
y_profile = y_coords[near_centre]
sigma_profile = sd[near_centre, 2]
sort_idx = y_profile.argsort()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(y_profile[sort_idx], sigma_profile[sort_idx], 'r-o', markersize=3)
ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label="fault")
ax.set_xlabel("y")
ax.set_ylabel(r"$\sigma_{xy}$")
ax.set_title("Stress profile at x=0.5")
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
