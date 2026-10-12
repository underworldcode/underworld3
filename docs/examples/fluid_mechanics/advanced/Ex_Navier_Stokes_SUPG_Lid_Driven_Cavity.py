# %% [markdown]
"""
# Navier-Stokes Lid-Driven Cavity with the Eulerian SUPG solver (Re = 100)

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** advanced
**RUNTIME:** ~4 minutes

## Description

The lid-driven cavity at Re = 100 with `uw.systems.NavierStokes`, the
Navier-Stokes solver that assembles the momentum advection on the grid and
stabilises it with streamline-upwind Petrov-Galerkin weighting. Each step is
one linear Oseen solve with the advecting velocity extrapolated from the two
stored levels. The centreline velocity extrema are compared with Ghia, Ghia &
Shin (1982).

## Key Concepts

- Eulerian (grid-based) Navier-Stokes with SUPG stabilisation
- The advecting velocity: extrapolated, Picard-corrected, or implicit
- The cell-Peclet weight of the stabilisation (on by default)
- Marching to a steady state and reading centreline profiles

## Reference

Ghia, Ghia & Shin (1982), "High-Re solutions for incompressible flow using
the Navier-Stokes equations and a multigrid method", J. Comp. Physics 48, 387-411.
"""

# %% [markdown]
"""
## Parameters
"""

# %%
RE = 100.0          # PARAM: Reynolds number (unit lid speed, unit cavity, viscosity 1/Re)
CELLSIZE = 1 / 32   # PARAM: mesh element size
COURANT = 1.0       # PARAM: time step as a multiple of the cell-crossing time at the lid
NSTEPS = 400        # PARAM: number of time steps
PICARD = 0          # PARAM: extra Picard passes per step (0 = one linear solve per step)

# %%
import numpy as np
import sympy
from mpi4py import MPI
import underworld3 as uw

# %% [markdown]
"""
## Mesh and fields

P2 velocity and P1 pressure (Taylor-Hood) on an unstructured simplex mesh.
"""

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=CELLSIZE, qdegree=3)
v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

# %% [markdown]
"""
## The solver

`NavierStokes` is a subclass of the Stokes solver: it takes the same
constitutive model and boundary conditions. `rho=1` with viscosity `1/RE`
gives Re on the unit cavity. `advection="extrapolated"` (the default) makes
each step one linear solve; `picard_iterations` re-solves with the latest
iterate for the fully implicit fixed point; `advection="implicit"` lets the
nonlinear solver take Newton steps instead.
"""

# %%
ns = uw.systems.NavierStokes(
    mesh, v, p, rho=1.0, order=1, advection="extrapolated", picard_iterations=PICARD)
ns.constitutive_model = uw.constitutive_models.ViscousFlowModel
ns.constitutive_model.Parameters.shear_viscosity_0 = 1.0 / RE
ns.tolerance = 1.0e-6

for boundary in ("Left", "Right", "Bottom"):
    ns.add_dirichlet_bc((0.0, 0.0), boundary)
ns.add_dirichlet_bc((1.0, 0.0), "Top")          # the lid, singular at the corners
ns.bodyforce = sympy.Matrix([[0.0, 0.0]])

# %% [markdown]
"""
## Time stepping

The Courant number is the number of cells the lid crosses in a step. The
implicit scheme has no stability limit on it; Courant 1 is a good accuracy
choice for the transient, and the run is stopped when the velocity stops
changing.
"""

# %%
dt = COURANT * CELLSIZE / 1.0
line = np.linspace(0.0, 1.0, 201)
vertical = np.c_[0.5 * np.ones_like(line), line]        # x = 0.5: u(y)
horizontal = np.c_[line, 0.5 * np.ones_like(line)]      # y = 0.5: v(x)

def centreline_extrema():
    u_c = uw.function.evaluate(v.sym[0], vertical).reshape(-1)
    v_c = uw.function.evaluate(v.sym[1], horizontal).reshape(-1)
    comm = uw.mpi.comm
    # evaluate() answers for the points this rank owns: reduce the extrema.
    return (comm.allreduce(float(u_c.min()), op=MPI.MIN),
            comm.allreduce(float(v_c.max()), op=MPI.MAX),
            comm.allreduce(float(v_c.min()), op=MPI.MIN))

for step in range(NSTEPS):
    before = np.array(v.array[...])
    ns.solve(timestep=dt, zero_init_guess=False)
    change = float(np.abs(np.asarray(v.array[...]) - before).max()) if before.size else 0.0
    change = uw.mpi.comm.allreduce(change, op=MPI.MAX)
    if (step + 1) % 50 == 0 or step == 0:
        u_min, v_max, v_min = centreline_extrema()
        uw.pprint(f"step {step + 1:4d}  t {dt * (step + 1):.3f}  "
                  f"u_min {u_min:.4f}  v_max {v_max:.4f}  v_min {v_min:.4f}  change {change:.2e}")
    if change < 1.0e-6:
        break

# %% [markdown]
"""
## Comparison with Ghia et al. (1982)

On a 1/32 mesh the three extrema come within about 4% of the reference; the
difference is the mesh (the extrema are steady to four digits).
"""

# %%
GHIA = dict(u_min=-0.2109, v_max=0.1753, v_min=-0.2453)
u_min, v_max, v_min = centreline_extrema()
uw.pprint(f"u_min on x = 0.5:  {u_min:.4f}   (Ghia {GHIA['u_min']})")
uw.pprint(f"v_max on y = 0.5:  {v_max:.4f}   (Ghia {GHIA['v_max']})")
uw.pprint(f"v_min on y = 0.5:  {v_min:.4f}   (Ghia {GHIA['v_min']})")
assert abs(u_min - GHIA["u_min"]) < 0.02 and abs(v_min - GHIA["v_min"]) < 0.02

# %% [markdown]
"""
## Notes

- `ns.supg_weight = 0` gives the plain Galerkin form; on this mesh at Re 100
  it runs as well, because the element Reynolds number is small. Set the
  weight back to 1 and raise Re to see where the stabilisation starts to matter.
- `peclet_weight` (default 4) turns the stabilisation off in cells that are
  diffusion-dominated, where it is not needed and costs accuracy.
- The design note `docs/developer/design/eulerian-supg-transport.md` records
  the benchmarks (Kovasznay flow, this cavity to Re 1000, the DFG cylinder,
  Taylor-Green vortex decay).
"""
