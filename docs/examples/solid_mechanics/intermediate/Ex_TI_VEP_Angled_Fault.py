# %% [markdown]
r"""
# Viscoelastic-Plastic Shear Box with Angled Fault

**PHYSICS:** solid_mechanics
**DIFFICULTY:** intermediate
**RUNTIME:** ~2 minutes

## Description

A 2D shear box with an embedded fault at 15 degrees from horizontal, using the
`TransverseIsotropicVEPFlowModel` constitutive model. This combines:

- **Transverse isotropy**: anisotropic viscosity with the fault normal as director
- **Viscoelasticity**: Maxwell stress buildup with BDF-1 time integration
- **Plastic yield**: resolved fault-plane shear limits the stress

The key physical result: stress builds elastically until the **resolved shear
stress on the fault plane** reaches the yield stress $\tau_y$. At that point,
fault-plane shear yields while the stress component normal to the fault
continues to build as pure viscoelastic.

## Physical Setup

| Parameter | Value |
|-----------|-------|
| Domain | 1 x 1 |
| Fault | centred, 15 deg from horizontal |
| $\eta_0$ (bulk) | 1 |
| $\eta_1$ (fault-plane) | 1 |
| $\mu$ (shear modulus) | 1 |
| $\tau_y$ (fault yield) | 0.15 |
| Fault width | 0.08 |
| Top velocity | 0.5 |
"""

# %%
#| echo: false
import nest_asyncio
nest_asyncio.apply()

# %%
import os
import numpy as np
import sympy
import underworld3 as uw

import matplotlib
if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("output", exist_ok=True)

# %% [markdown]
"""
## Parameters
"""

# %%
# Physical parameters
ETA_0 = 1.0          # bulk viscosity
ETA_1 = 1.0          # fault-plane viscosity (same as bulk for this test)
MU = 1.0             # shear modulus
TAU_Y = 0.15         # fault-plane yield stress
V_TOP = 0.5          # top boundary velocity
DT = 0.025           # timestep
N_STEPS = 80         # number of steps

# Mesh
RES = 64             # mesh resolution (RES x RES)

# Fault geometry
FAULT_ANGLE_DEG = 15.0   # angle from horizontal (degrees)
FAULT_WIDTH = 0.08       # influence function width
FAULT_LENGTH = 0.6       # fault length (centered in domain)

# %% [markdown]
r"""
## Mesh and Variables
"""

# %%
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(RES, RES),
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    qdegree=3,
)

v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2, vtype=uw.VarType.VECTOR)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1,
                                    continuous=True, vtype=uw.VarType.SCALAR)

# %% [markdown]
r"""
## Fault Surface

The fault is a 1D polyline at 15 degrees from horizontal, centered in the domain.
The `Surface` class computes the signed distance field and normals automatically.
"""

# %%
theta = np.radians(FAULT_ANGLE_DEG)
cx, cy = 0.5, 0.5  # centre of domain

# Fault endpoints
dx = FAULT_LENGTH / 2 * np.cos(theta)
dy = FAULT_LENGTH / 2 * np.sin(theta)
fault_points = np.array([
    [cx - dx, cy - dy],
    [cx + dx, cy + dy],
])

fault = uw.meshing.Surface("fault", mesh, fault_points, symbol="F")
fault.discretize()

# Director: fault normal (perpendicular to fault, pointing "up")
n_x = -np.sin(theta)
n_y = np.cos(theta)
director = sympy.Matrix([n_x, n_y])

print(f"Fault angle: {FAULT_ANGLE_DEG} deg")
print(f"Director (fault normal): [{n_x:.4f}, {n_y:.4f}]")

# %% [markdown]
r"""
## Yield Stress Field

The yield stress varies spatially: low near the fault, high in the bulk.
We interpolate the weakness (1/$\tau_y$) to avoid steep gradients.
"""

# %%
TAU_Y_BULK = 200.0  # effectively infinite for the bulk

weakness = fault.influence_function(
    width=FAULT_WIDTH,
    value_near=1 / TAU_Y,
    value_far=1 / TAU_Y_BULK,
    profile="gaussian",
)
tau_y_field = 1 / weakness

# %% [markdown]
r"""
## Solver Setup

The `TransverseIsotropicVEPFlowModel` combines the anisotropic viscosity tensor
(Muhlhaus-Moresi) with viscoelastic stress history and plastic yield on the
fault plane.
"""

# %%
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)

# Create model with BDF-1 time integration
cm = uw.constitutive_models.TransverseIsotropicVEPFlowModel(
    stokes.Unknowns, order=1
)
stokes.constitutive_model = cm

# Set parameters
cm.Parameters.shear_viscosity_0 = ETA_0
cm.Parameters.shear_viscosity_1 = ETA_1
cm.Parameters.shear_modulus = MU
cm.Parameters.yield_stress = tau_y_field
cm.Parameters.director = director
cm.Parameters.shear_viscosity_min = ETA_0 * 1.0e-3
cm.Parameters.strainrate_inv_II_min = 1.0e-6
cm.yield_mode = "softmin"  # smooth approximation to min (default delta=0.1)

# Solver settings
stokes.saddle_preconditioner = 1 / cm.K
stokes.tolerance = 1.0e-4
stokes.petsc_options["ksp_type"] = "fgmres"

# Boundary conditions: simple shear
stokes.add_essential_bc(sympy.Matrix([V_TOP, 0.0]), "Top")
stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
stokes.add_essential_bc((sympy.oo, 0.0), "Left")
stokes.add_essential_bc((sympy.oo, 0.0), "Right")
stokes.bodyforce = sympy.Matrix([0.0, 0.0])

# %% [markdown]
r"""
## Time Stepping

Track the stress components at a point on the fault to show elastic buildup
and yield cap behavior.
"""

# %%
# Monitoring point: centre of the fault
monitor_coord = np.array([[0.5, 0.5]])

times = []
sigma_xy_history = []
sigma_resolved_history = []

# Analytical VE reference (no yield): sigma_xy = eta * gamma_dot * (1 - exp(-t / t_r))
gamma_dot = V_TOP  # approximate global shear rate
t_relax = ETA_1 / MU

for step in range(N_STEPS):
    stokes.solve(timestep=DT, zero_init_guess=(step == 0))

    t = (step + 1) * DT
    reason = stokes.snes.getConvergedReason()
    its = stokes.snes.getIterationNumber()

    # Sample stress at monitoring point
    tau = stokes.tau
    tau_data = tau.data
    tau_coords = tau.coords

    # Find nearest stress evaluation point to monitor location
    dists = np.linalg.norm(tau_coords - monitor_coord, axis=1)
    idx = np.argmin(dists)
    s_xx, s_yy, s_xy = tau_data[idx, 0], tau_data[idx, 1], tau_data[idx, 2]

    # Resolved shear on fault plane: tau_resolved = n^T sigma n_perp
    # For a fault with normal (n_x, n_y), the tangent is (n_y, -n_x)
    # Resolved shear = t^T sigma n = sigma_ij * t_i * n_j
    t_x, t_y = n_y, -n_x  # tangent vector
    resolved_shear = (s_xx * t_x * n_x + s_xy * (t_x * n_y + t_y * n_x)
                      + s_yy * t_y * n_y)

    times.append(t)
    sigma_xy_history.append(s_xy)
    sigma_resolved_history.append(resolved_shear)

    flag = " ***" if reason < 0 else ""
    if step % 10 == 0 or reason < 0:
        print(f"Step {step+1:3d}, t={t:.3f}: "
              f"sigma_xy={s_xy:.4f}, resolved={resolved_shear:.4f}, "
              f"SNES={reason}, its={its}{flag}")

# %% [markdown]
r"""
## Stress Evolution

The plot shows stress at the fault centre over time. The dashed line shows
the analytical VE solution (no yield). The resolved shear stress on the
fault plane should cap at $\tau_y = 0.15$.
"""

# %%
times = np.array(times)
sigma_xy_history = np.array(sigma_xy_history)
sigma_resolved_history = np.array(sigma_resolved_history)

# Analytical VE solution (no yield)
t_analytical = np.linspace(0, times[-1], 200)
sigma_ve_analytical = ETA_1 * gamma_dot * (1 - np.exp(-t_analytical / t_relax))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: sigma_xy and resolved shear vs time
ax = axes[0]
ax.plot(times, sigma_xy_history, 'b-o', markersize=2, label=r'$\sigma_{xy}$ (global)')
ax.plot(times, sigma_resolved_history, 'r-s', markersize=2,
        label=r'$\tau_{\mathrm{resolved}}$ (fault plane)')
ax.plot(t_analytical, sigma_ve_analytical, 'k--', alpha=0.5, label='VE analytical (no yield)')
ax.axhline(y=TAU_Y, color='gray', linestyle=':', linewidth=2,
           label=rf'$\tau_y = {TAU_Y}$')
ax.set_xlabel('Time')
ax.set_ylabel('Stress')
ax.set_title(f'Stress at fault centre (angle={FAULT_ANGLE_DEG} deg)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right panel: stress profile across fault at end of simulation
ax = axes[1]
n_samples = 100
# Profile perpendicular to the fault, through the centre
profile_dist = np.linspace(-0.4, 0.4, n_samples)
profile_x = 0.5 + profile_dist * n_x  # along fault normal direction
profile_y = 0.5 + profile_dist * n_y
# Clip to domain
valid = (profile_x > 0.02) & (profile_x < 0.98) & (profile_y > 0.02) & (profile_y < 0.98)
profile_coords = np.column_stack([profile_x[valid], profile_y[valid]])

# Evaluate tau_y field along profile
tau_y_profile = uw.function.evaluate(tau_y_field, profile_coords).flatten()

# For stress, find nearest tau evaluation points
tau_coords = stokes.tau.coords
tau_data = stokes.tau.data
stress_profile = np.zeros(len(profile_coords))
for i, pc in enumerate(profile_coords):
    dists = np.linalg.norm(tau_coords - pc, axis=1)
    idx = np.argmin(dists)
    s_xx, s_yy, s_xy = tau_data[idx, 0], tau_data[idx, 1], tau_data[idx, 2]
    t_x, t_y = n_y, -n_x
    stress_profile[i] = (s_xx * t_x * n_x + s_xy * (t_x * n_y + t_y * n_x)
                          + s_yy * t_y * n_y)

ax.plot(profile_dist[valid], np.abs(stress_profile), 'r-', linewidth=2,
        label=r'$|\tau_{\mathrm{resolved}}|$')
ax.plot(profile_dist[valid], tau_y_profile, 'k--', linewidth=1,
        label=r'$\tau_y$ (yield stress)')
ax.axvline(0, color='gray', linestyle=':', alpha=0.5, label='Fault centre')
ax.set_xlabel('Distance from fault (along normal)')
ax.set_ylabel('Stress')
ax.set_title('Final stress profile across fault')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/ti_vep_angled_fault.png", dpi=150)
plt.show()

# %% [markdown]
r"""
## Summary

This example demonstrates the `TransverseIsotropicVEPFlowModel`:

1. **Elastic stress buildup**: stress grows from zero following the Maxwell solution
2. **Fault-plane yield**: the resolved shear stress on the fault plane caps at $\tau_y$
3. **Anisotropic yield**: only the fault-parallel shear component yields; the normal
   component continues to build elastically
4. **Smooth spatial transition**: the Gaussian influence function localises yield to
   the fault zone, with background material remaining elastic

The resolved shear criterion ($|\dot\gamma| = \sqrt{|T|^2 - \dot\varepsilon_n^2}$)
correctly identifies fault-plane shear regardless of the fault orientation, avoiding
the orientation-dependent errors that arise from using the global strain rate invariant.
"""
