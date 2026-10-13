# %% [markdown]
"""
# Taylor-Green vortex decay with the Eulerian SUPG Navier-Stokes solver

**PHYSICS:** fluid_mechanics
**DIFFICULTY:** advanced
**RUNTIME:** ~1 minute

## Description

An exact unsteady solution of the Navier-Stokes equations: a lattice of
counter-rotating vortices that decays in place,

    u = (-sin x cos y, cos x sin y) exp(-2 nu t),   p = (cos 2x + cos 2y)/4 exp(-4 nu t),

on the box [0, pi]^2. On that box the walls carry no normal flow and no
tangential stress, so free-slip walls (the normal component fixed) are exact
and nothing on the boundary depends on time. The velocity error against the
exact solution at the end of the run measures the scheme directly.

## Key Concepts

- Time-dependent validation against an exact Navier-Stokes solution
- Free-slip walls as partial Dirichlet conditions
- The kinetic energy decay, exp(-4 nu t), as a second check
- Where the SUPG stabilisation costs accuracy and how the Peclet weight removes it
"""

# %% [markdown]
"""
## Parameters
"""

# %%
NU = 0.01           # PARAM: viscosity (density 1)
RES = 16            # PARAM: cells across the box
DT = 0.025          # PARAM: time step
T_END = 0.5         # PARAM: end time
PECLET_WEIGHT = 4.0 # PARAM: cell-Peclet weight of the stabilisation (0 = uniform)

# %%
import numpy as np
import sympy
import underworld3 as uw

# %% [markdown]
"""
## Mesh, fields and the exact solution
"""

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(np.pi, np.pi), cellSize=np.pi / RES, regular=True, qdegree=3)
x, y = mesh.X
v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

def exact(t):
    F = sympy.exp(-2 * NU * t)
    U = sympy.Matrix([[-sympy.sin(x) * sympy.cos(y) * F, sympy.cos(x) * sympy.sin(y) * F]])
    P = (sympy.cos(2 * x) + sympy.cos(2 * y)) * F ** 2 / 4
    return U, P

U0, P0 = exact(0.0)
v.array[:, 0, :] = uw.function.evaluate(U0, v.coords).reshape(-1, 2)
p.array[:, 0, 0] = uw.function.evaluate(P0, p.coords).reshape(-1)

# %% [markdown]
"""
## The solver with free-slip walls

A partial Dirichlet condition fixes one component and leaves the other free:
`(0.0, None)` on the vertical walls, `(None, 0.0)` on the horizontal ones.
"""

# %%
ns = uw.systems.NavierStokes(mesh, v, p, rho=1.0, order=1, peclet_weight=PECLET_WEIGHT)
ns.constitutive_model = uw.constitutive_models.ViscousFlowModel
ns.constitutive_model.Parameters.shear_viscosity_0 = NU
ns.tolerance = 1.0e-8
ns.add_dirichlet_bc((0.0, None), "Left")
ns.add_dirichlet_bc((0.0, None), "Right")
ns.add_dirichlet_bc((None, 0.0), "Bottom")
ns.add_dirichlet_bc((None, 0.0), "Top")
ns.bodyforce = sympy.Matrix([[0.0, 0.0]])

# %% [markdown]
"""
## The error against the exact solution

The exact velocity is F(t) U0, so the L2 error expands into three integrals
that carry no time dependence: ||v - F U0||^2 = <v,v> - 2F <v,U0> + F^2 <U0,U0>.
"""

# %%
I_vv = uw.maths.Integral(mesh, v.sym.dot(v.sym))
I_vU = uw.maths.Integral(mesh, v.sym.dot(U0))
I_UU = float(uw.maths.Integral(mesh, U0.dot(U0)).evaluate())

def velocity_error(t):
    F = np.exp(-2 * NU * t)
    vv, vU = float(I_vv.evaluate()), float(I_vU.evaluate())
    return np.sqrt(max(vv - 2 * F * vU + F ** 2 * I_UU, 0.0) / (F ** 2 * I_UU))

E0 = float(I_vv.evaluate())
uw.pprint(f"interpolation error of the exact field on this mesh: {velocity_error(0.0):.2e}")

# %% [markdown]
"""
## March
"""

# %%
n_steps = int(round(T_END / DT))
t = 0.0
for step in range(n_steps):
    ns.solve(timestep=DT, zero_init_guess=False)
    t += DT
    if (step + 1) % 5 == 0 or step + 1 == n_steps:
        uw.pprint(f"step {step + 1:3d}  t {t:.3f}  velocity error {velocity_error(t):.3e}  "
                  f"E/E0 {float(I_vv.evaluate()) / E0:.6f}  exact {np.exp(-4 * NU * t):.6f}")

# %% [markdown]
"""
## What to expect

On the 1/16 mesh the velocity error at t = 0.5 is a few times 1e-4 (the P2
interpolation error is 1.4e-4) and the kinetic energy follows exp(-4 nu t)
to six digits. With `PECLET_WEIGHT = 0` the stabilisation acts in every cell
and the error rises by a fixed factor: this flow is resolved and needs no
stabilisation, which is what the weight detects. With `ns.supg_weight = 0`
(plain Galerkin) the error is the same as with the weight.
"""

# %%
err = velocity_error(t)
uw.pprint(f"final velocity error {err:.3e}, energy ratio {float(I_vv.evaluate()) / E0:.6f} (exact {np.exp(-4 * NU * t):.6f})")
assert err < 2.0e-3
