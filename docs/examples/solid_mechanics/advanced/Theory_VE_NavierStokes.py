# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Viscoelastic Navier-Stokes Theory

**PHYSICS:** solid_mechanics
**DIFFICULTY:** advanced
**STATUS:** Theory derivation with WIP implementation

## Description

Theoretical derivation of the combined Navier-Stokes and viscoelastic
formulation. This notebook develops the mathematical framework for
integrating viscous, elastic, and plastic constitutive behavior using
Adams-Moulton time integration schemes.

## Key Concepts

- **Navier-Stokes discretization**: Forward Euler and Adams-Moulton schemes
- **Viscoelastic constitutive law**: Maxwell model with stress history
- **Effective viscosity**: Time-stepping induced effective properties
- **BDF/Adams-Moulton coefficients**: Higher-order time integration

## Parameters

- `uw_resolution`: Mesh resolution (default: 0.033)
- `uw_mu`: Shear modulus parameter (default: 0.5)
- `uw_max_steps`: Maximum timesteps (default: 500)

## Note

The implementation section following the theory is work-in-progress.
The mathematical derivations are complete and serve as reference material.
"""

# %% [markdown]
"""
## Navier-Stokes Equation

$$
    \\rho\\frac{D{\\mathbf{u}}}{D t} + \\nabla \\cdot \\boldsymbol{\\tau} - \\nabla \\mathbf{p} = \\mathbf{\\rho g \\hat{\\mathbf{z}}}
$$

The viscous constitutive law connects the stress to gradients of $\\mathbf{v}$ as follows:

$$
\\boldsymbol{\\tau} = \\eta ( \\nabla \\mathbf{u} + (\\nabla \\mathbf{u})^T )
$$

We next write a discrete problem in terms of corresponding variables defined on a mesh.

$$
    \\rho\\frac{\\mathbf{u}_{[1]} - \\mathbf{u}^*}{\\Delta t}  = \\rho g - \\nabla \\cdot \\mathbf{\\tau} + \\nabla p
$$

where $\\mathbf{u}^*$ is the value of $\\mathbf{u}$ evaluated at upstream point at a time $t - \\delta t$.
Numerically, this is the value on a particle at the previous timestep. This approximation is the forward
Euler integration in time for velocity because $\\tau$ is defined in terms of the unknowns.
$\\mathbf{u}_{[1]}$ denotes that solution uses the 1st order Adams-Moulton scheme and higher order
updates are well known:

$$
    \\rho\\frac{\\mathbf{u}_{[2]} - \\mathbf{u}^*}{\\Delta t}  =\\rho g - \\nabla \\cdot \\left[
                                                                \\frac{1}{2} \\boldsymbol{\\tau} +
                                                                \\frac{1}{2} \\boldsymbol{\\tau^*}
                                                                   \\right]
                                                                - \\nabla p
$$

and

$$
     \\rho\\frac{\\mathbf{u}_{[3]} - \\mathbf{u}^*}{\\Delta t}
             = \\rho g - \\nabla \\cdot \\left[ \\frac{5}{12} \\boldsymbol{\\tau}
                                                        - \\frac{1}{12} \\boldsymbol{\\tau^{**}}
                                                         - \\frac{1}{12} \\boldsymbol{\\tau^{**}}
                                                          \\right] - \\nabla p
$$

Where $\\boldsymbol\\tau^*$ and $\\boldsymbol\\tau^{**}$ are the upstream history values at
$t - \\Delta t$ and $t - 2\\Delta t$ respectively.
"""

# %% [markdown]
"""
## Stress History Simplification

In the Navier-Stokes problem, it is common to write
$\\boldsymbol\\tau=\\eta \\left(\\nabla \\mathbf u + (\\nabla \\mathbf u)^T \\right)$
and $\\boldsymbol\\tau^*=\\eta \\left(\\nabla \\mathbf u^* + (\\nabla \\mathbf u^*)^T \\right)$
which ignores rotation and shearing of the stress during the interval $\\Delta T$.
This simplifies the implementation because only the velocity history is required,
not the history of the stress tensor.
"""

# %% [markdown]
"""
## Viscoelasticity

In viscoelasticity, the elastic part of the deformation is related to the stress rate.
If we approach this problem as a perturbation to the viscous Navier-Stokes equation,
we first consider the constitutive behaviour

$$
 \\frac{1}{2\\mu}\\frac{D{\\boldsymbol\\tau}}{Dt} + \\frac{\\boldsymbol\\tau}{2\\eta} = \\dot{\\boldsymbol\\varepsilon}
$$

A first order difference form for ${D \\tau}/{D t}$ then gives

$$
    \\frac{\\boldsymbol\\tau - \\boldsymbol\\tau^{*}}{2 \\Delta t \\mu} + \\frac{\\boldsymbol\\tau}{2 \\eta} = \\dot{\\boldsymbol\\varepsilon}
$$

where $\\tau^*$ is the stress history along the characteristics associated with the current
computational points. Rearranging to find an expression for the current stress in terms of
the strain rate:

$$
    \\boldsymbol\\tau = 2 \\dot\\varepsilon \\eta_{\\textrm{eff}_{(1)}} + \\frac{\\eta \\boldsymbol\\tau^{*}}{\\Delta t \\mu + \\eta}
$$

where an 'effective viscosity' is introduced, defined as follows:

$$
    \\eta_{\\textrm{eff}_{(1)}} = \\frac{\\Delta t \\eta \\mu}{\\Delta t \\mu + \\eta}
$$
"""

# %% [markdown]
"""
## Combined NS + VE Formulation

Substituting this definition of the stress into the forward-Euler form of the
Navier-Stokes discretisation then gives

$$
    \\rho\\frac{\\mathbf{u}_{[1]} - \\mathbf{u}^*}{\\Delta t}  = \\rho g - \\nabla \\cdot \\left[ 2 \\dot{\\boldsymbol\\varepsilon}\\eta_{\\textrm{eff}_{(1)}} +  \\frac{\\eta \\boldsymbol\\tau^{*}}{\\Delta t \\mu + \\eta}  \\right] + \\nabla p
$$

and the 2nd order (Crank-Nicholson) form becomes

$$
    \\rho\\frac{\\mathbf{u}_{[2]} - \\mathbf{u}^*}{\\Delta t}  = \\rho g - \\frac{1}{2} \\nabla \\cdot \\left[ 2 \\dot\\varepsilon \\eta_{\\textrm{eff}_{(1)}} + \\left[\\frac{\\eta}{\\Delta t \\mu + \\eta} + 1\\right]\\tau^*  \\right] + \\nabla p
$$
"""

# %% [markdown]
"""
## Second-Order Stress Rate

If we use $\\tau^{**}$ in the estimate for the stress rate, we have

$$
    \\frac{3 \\tau - 4 \\tau^{*} + \\tau^{**}} {4 \\Delta t \\mu} + \\frac{\\tau}{2 \\eta}  = \\dot\\varepsilon
$$

Giving

$$
    \\boldsymbol\\tau = 2 \\dot{\\boldsymbol\\varepsilon} \\eta_{\\textrm{eff}_{(2)}} + \\frac{4 \\eta \\boldsymbol\\tau^{*}}{2 \\Delta t \\mu + 3 \\eta} - \\frac{\\eta \\boldsymbol\\tau^{**}}{2 \\Delta t \\mu + 3 \\eta}
$$

$$
    \\eta_{\\textrm{eff}_{(2)}} = \\frac{2 \\Delta t \\eta \\mu}{2 \\Delta t \\mu + 3 \\eta}
$$

$$
   \\frac{ \\mathbf{u}_{[3]} - \\mathbf{u}^*}{\\Delta t} =
    \\rho g
    - \\nabla \\cdot \\left[
\\frac{5 \\dot\\varepsilon \\eta_\\textrm{eff}}{6} + \\frac{5 \\eta \\tau^{*}}{3 \\cdot \\left(2 \\Delta\\,\\!t \\mu + 3 \\eta\\right)} - \\frac{5 \\eta \\tau^{**}}{12 \\cdot \\left(2 \\Delta\\,\\!t \\mu + 3 \\eta\\right)} + \\frac{2 \\tau^{*}}{3} - \\frac{\\tau^{**}}{12} \\right] + \\nabla p
$$
"""

# %% [markdown]
"""
## Implementation Notes

The BDF/Adams-Moulton coefficients are used as follows:
- dot_f term uses BDF coefficients (see: https://en.wikipedia.org/wiki/Backward_differentiation_formula)
- Flux history follows Adams-Moulton
- Substitute for present stress in ADM
"""

# %% [markdown]
"""
## Setup (WIP)

The following implementation code is work-in-progress. The theory above
is complete, but the code below has incomplete sections marked with
debug breakpoints.
"""

# %%
# Fix trame async issue
import nest_asyncio
nest_asyncio.apply()

import os
os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
import underworld3 as uw
import numpy as np
import sympy

from underworld3 import timing

# %% [markdown]
"""
## Configurable Parameters

Override from command line:
```bash
python Theory_VE_NavierStokes.py -uw_resolution 0.05
python Theory_VE_NavierStokes.py -uw_mu 1.0
```
"""

# %%
params = uw.Params(
    uw_resolution = 0.033,        # Mesh resolution
    uw_mu = 0.5,                  # Shear modulus
    uw_max_steps = 500,           # Maximum timesteps
)

resolution = params.uw_resolution
mu = params.uw_mu
maxsteps = int(params.uw_max_steps)

# %% [markdown]
"""
## Mesh and Variables
"""

# %%
mesh1 = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.5, -0.5),
    maxCoords=(+1.5, +0.5),
    cellSize=resolution,
)

x, y = mesh1.X

# %%
U = uw.discretisation.MeshVariable(
    "U", mesh1, mesh1.dim, vtype=uw.VarType.VECTOR, degree=2
)
P = uw.discretisation.MeshVariable(
    "P", mesh1, 1, vtype=uw.VarType.SCALAR, degree=1, continuous=True
)
T = uw.discretisation.MeshVariable("T", mesh1, 1, vtype=uw.VarType.SCALAR, degree=3)

# Nodal values of deviatoric stress (symmetric tensor)
work = uw.discretisation.MeshVariable(
    "W", mesh1, 1, vtype=uw.VarType.SCALAR, degree=2, continuous=False
)
St = uw.discretisation.MeshVariable(
    r"Stress",
    mesh1,
    (2, 2),
    vtype=uw.VarType.SYM_TENSOR,
    degree=2,
    continuous=False,
    varsymbol=r"{\tau}",
)

# Strain rate invariant
Edot_inv_II = uw.discretisation.MeshVariable(
    "eps_II",
    mesh1,
    1,
    vtype=uw.VarType.SCALAR,
    degree=2,
    varsymbol=r"{|\dot\varepsilon|}",
)
St_inv_II = uw.discretisation.MeshVariable(
    "tau_II", mesh1, 1, vtype=uw.VarType.SCALAR, degree=2, varsymbol=r"{|\tau|}"
)

# %% [markdown]
"""
## Swarm for Stress History
"""

# %%
swarm = uw.swarm.Swarm(mesh=mesh1, recycle_rate=5)

material = uw.swarm.SwarmVariable(
    "M",
    swarm,
    size=1,
    vtype=uw.VarType.SCALAR,
    proxy_continuous=True,
    proxy_degree=1,
    dtype=int,
)

strain_inv_II = uw.swarm.SwarmVariable(
    "Strain",
    swarm,
    size=1,
    vtype=uw.VarType.SCALAR,
    proxy_continuous=True,
    proxy_degree=2,
    varsymbol=r"{|\varepsilon|}",
    dtype=float,
)

stress_star = uw.swarm.SwarmVariable(
    r"stress_dt",
    swarm,
    (2, 2),
    vtype=uw.VarType.SYM_TENSOR,
    proxy_continuous=True,
    proxy_degree=2,
    varsymbol=r"{\tau^{*}_{p}}",
)

stress_star_star = uw.swarm.SwarmVariable(
    r"stress_2dt",
    swarm,
    (2, 2),
    vtype=uw.VarType.SYM_TENSOR,
    proxy_continuous=True,
    proxy_degree=2,
    varsymbol=r"{\tau^{**}_{p}}",
)

swarm.populate(fill_param=2)

# %% [markdown]
"""
## Stokes System Setup
"""

# %%
stokes = uw.systems.Stokes(
    mesh1,
    velocityField=U,
    pressureField=P,
    verbose=False,
)

# %% [markdown]
"""
## WIP: Implementation continues below

The following code sections are work-in-progress and contain debug
breakpoints to prevent execution of incomplete code.
"""

# %%
# WIP: The following sections require completion
# Debug breakpoint to prevent execution of incomplete code
print("Theory derivation complete. Implementation is WIP.")
print("Remove the following line to continue with experimental code:")
raise SystemExit("WIP: Implementation incomplete - see theory sections above")

# %%
# Symbolic manipulation for effective viscosity (experimental)
from sympy import UnevaluatedExpr

st = sympy.UnevaluatedExpr(stokes.constitutive_model.viscosity) * sympy.UnevaluatedExpr(
    stokes.strainrate
)

eta_0 = sympy.sympify(10) ** -6
C_0 = sympy.log(10**6)

stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = sympy.symbols(r"\eta")
stokes.constitutive_model.Parameters.shear_modulus = sympy.symbols(r"\mu")
stokes.constitutive_model.Parameters.stress_star = stress_star.sym
stokes.constitutive_model.Parameters.dt_elastic = sympy.symbols(r"\Delta\ t")
stokes.constitutive_model.Parameters.strainrate_inv_II = stokes.Unknowns.Einv2
stokes.constitutive_model.Parameters.strainrate_inv_II_min = 0

print("Constitutive model configured")
print(stokes.constitutive_model)
