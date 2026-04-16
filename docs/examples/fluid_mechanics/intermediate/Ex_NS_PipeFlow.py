# %% [markdown]
"""
# Navier-Stokes: Poiseuille (Pipe) Flow

Transient development of parabolic flow between parallel plates driven by
a uniform inlet velocity. The flow starts from rest and evolves toward
the classical Poiseuille profile.

This is a useful test case because:

- The steady-state solution is known analytically
- The transient spin-up exercises the time-derivative (DDt) machinery
- The Reynolds number controls how quickly inertial effects appear

**Reference:** Cengel, Y. A. (2010). *Fluid Mechanics: Fundamentals
and Applications*. Tata McGraw Hill.

**Original example by** Juan Carlos Graciosa.
"""

# %%
import underworld3 as uw
import numpy as np
import sympy

# %% [markdown]
"""
## Problem parameters

Uniform flow enters from the left between no-slip walls (top and bottom).
The right boundary is open (zero traction). The flow develops a parabolic
profile whose centreline velocity is 1.5x the inlet velocity.

Try changing `RE` to see how inertia affects the spin-up. At Re = 10 the
flow reaches steady state quickly; at Re = 1000 it takes much longer and
the entrance effects extend further downstream.
"""

# %%
# --- Physical parameters (dimensional) ---
INLET_VELOCITY = 0.034          # m/s, uniform inlet
DENSITY        = 910.0          # kg/m^3
DYN_VISCOSITY  = 0.3094         # Pa.s
HEIGHT         = 0.10           # m, channel height (2 * half-width)
ASPECT_RATIO   = 8              # channel length / height

RE = DENSITY * INLET_VELOCITY * HEIGHT / DYN_VISCOSITY  # ~ 10
print(f"Reynolds number: {RE:.1f}")

# %%
# --- Mesh ---
RESOLUTION = 16                 # elements across the channel height
CELL_SIZE  = HEIGHT / RESOLUTION

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-0.5 * ASPECT_RATIO * HEIGHT, -0.5 * HEIGHT),
    maxCoords=( 0.5 * ASPECT_RATIO * HEIGHT,  0.5 * HEIGHT),
    cellSize=CELL_SIZE,
    qdegree=3,
    regular=False,
)

# %%
v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=False)

# %% [markdown]
r"""
## Solver setup

`NavierStokesSLCN` uses a Semi-Lagrangian Crank-Nicolson scheme for the
inertial term $\rho \, Du/Dt$. The material derivative is computed by
tracking particles upstream along characteristics (via a swarm).

The `order` parameter controls the time-integration accuracy (1 or 2).
"""

# %%
navier_stokes = uw.systems.NavierStokesSLCN(
    mesh,
    velocityField=v_soln,
    pressureField=p_soln,
    rho=DENSITY,
    order=1,
    verbose=False,
)

navier_stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
navier_stokes.constitutive_model.Parameters.viscosity = DYN_VISCOSITY
navier_stokes.penalty = 0
navier_stokes.bodyforce = sympy.Matrix([0, 0])

# Boundary conditions: inlet velocity on the left, no-slip top/bottom, open right
navier_stokes.add_dirichlet_bc((INLET_VELOCITY, 0.0), "Left")
navier_stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
navier_stokes.add_dirichlet_bc((0.0, 0.0), "Top")

navier_stokes.tolerance = 1e-6

# %% [markdown]
"""
## Timestepping

The timestep is set from a CFL condition based on the minimum cell radius
and the inlet velocity. We run enough steps to let the flow develop.
"""

# %%
C_MAX   = 0.5                      # target Courant number
NSTEPS  = 20                       # number of timesteps (increase for full spin-up)

delta_x = mesh.get_min_radius()
delta_t = C_MAX * delta_x / INLET_VELOCITY

print(f"Cell radius: {delta_x:.4e}")
print(f"Timestep:    {delta_t:.4e}")

# %%
for step in range(NSTEPS):
    navier_stokes.solve(timestep=delta_t, zero_init_guess=(step == 0))

    if step % 5 == 0 or step == NSTEPS - 1:
        # Sample centreline velocity at the outlet (x = max, y = 0)
        vx_max = np.abs(v_soln.data[:, 0]).max()
        print(f"step {step:3d}  |  max |vx| = {vx_max:.4e}")

# %% [markdown]
r"""
## Compare with the analytic solution

The fully-developed Poiseuille profile between plates at $y = \pm h/2$ is:

$$u_x(y) = \frac{3}{2} \, U_{\mathrm{mean}} \left(1 - \frac{4 y^2}{h^2}\right)$$

where $U_{\mathrm{mean}}$ is the mean (inlet) velocity. The centreline
velocity is $\frac{3}{2} U_{\mathrm{mean}}$.
"""

# %%
# Sample the velocity along a vertical line near the outlet
x_sample = 0.4 * ASPECT_RATIO * HEIGHT * 0.5  # near the outlet
n_sample = 50
y_sample = np.linspace(-0.5 * HEIGHT * 0.95, 0.5 * HEIGHT * 0.95, n_sample)
sample_coords = np.column_stack([np.full(n_sample, x_sample), y_sample])

vx_numerical = uw.function.evaluate(v_soln.sym[0], sample_coords)

# Analytic Poiseuille profile
h = HEIGHT
vx_analytic = 1.5 * INLET_VELOCITY * (1.0 - (2.0 * y_sample / h) ** 2)

# %%
if uw.mpi.size == 1 and uw.is_notebook():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(vx_analytic, y_sample / h, "k--", label="Analytic (steady)")
    ax.plot(vx_numerical.flatten(), y_sample / h, "o", ms=4, label=f"UW3 (step {NSTEPS})")
    ax.set_xlabel("$v_x$ (m/s)")
    ax.set_ylabel("$y / h$")
    ax.legend()
    ax.set_title(f"Poiseuille flow, Re = {RE:.0f}")
    plt.tight_layout()
    plt.show()

# %% [markdown]
"""
## Things to try

- Increase `NSTEPS` to see the profile converge to the analytic solution
- Change `RE` by adjusting `INLET_VELOCITY` or `DYN_VISCOSITY`
- Compare `order=1` vs `order=2` in the solver constructor
- Increase `RESOLUTION` — does the solution remain stable at finer meshes?
"""
