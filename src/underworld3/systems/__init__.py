r"""
PDE solver systems for Underworld3.

This module provides finite element solvers for partial differential equations
commonly encountered in geodynamics and continuum mechanics. All solvers use
PETSc's SNES (Scalable Nonlinear Equations Solvers) infrastructure.

Available Solvers
-----------------
Poisson : class
    Steady-state scalar Poisson equation.
SteadyStateDarcy : class
    Groundwater flow (Darcy equation).
Stokes : class
    Incompressible viscous flow (Stokes equations).
VE_Stokes : class
    Viscoelastic Stokes solver with stress history.
Projection : class
    L2 projection of fields onto mesh variables.
AdvDiffusion : class
    Advection-diffusion with semi-Lagrangian transport.
NavierStokes : class
    Navier-Stokes equations with inertia.
Diffusion : class
    Pure diffusion (no advection).

Time Derivative Schemes
-----------------------
Lagrangian_DDt, SemiLagragian_DDt, Eulerian_DDt
    Time derivative approximations for transient problems.

See Also
--------
underworld3.constitutive_models : Material rheology definitions.
underworld3.discretisation : Mesh and variable classes.
"""
from underworld3.cython.generic_solvers import (
    SNES_Scalar,
    SNES_Vector,
    SNES_Stokes_SaddlePt,
)

from .solvers import SNES_Poisson as Poisson
from .solvers import SNES_Darcy as SteadyStateDarcy
from .solvers import SNES_Stokes as Stokes
from .solvers import SNES_VE_Stokes as VE_Stokes
from .solvers import SNES_Projection as Projection
from .solvers import SNES_Vector_Projection as Vector_Projection
from .solvers import SNES_Tensor_Projection as Tensor_Projection

# from .solvers import SNES_Solenoidal_Vector_Projection as Solenoidal_Vector_Projection  ## WIP / maybe some issues
# from .solvers import (
#     SNES_AdvectionDiffusion_SLCN as AdvDiffusion,
# )  # fix examples then remove this


# These are now implemented the same way using the ddt module
from .solvers import SNES_AdvectionDiffusion as AdvDiffusionSLCN
from .solvers import SNES_AdvectionDiffusion as AdvDiffusion

# import diffusion-only solver
from .solvers import SNES_Diffusion as Diffusion

# These are now implemented the same way using the ddt module
from .solvers import SNES_NavierStokes as NavierStokesSwarm
from .solvers import SNES_NavierStokes as NavierStokesSLCN
from .solvers import SNES_NavierStokes as NavierStokes

# are the Lagrangian implementations actually distinct in reality ?
from .ddt import Lagrangian as Lagrangian_DDt
from .ddt import SemiLagrangian as SemiLagragian_DDt
from .ddt import Lagrangian_Swarm as Lagrangian_Swarm_DDt
from .ddt import Eulerian as Eulerian_DDt
