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
    Advection-diffusion composed from a DDt transport manager (the default
    manager, EulerianSUPG, assembles implicit advection with SUPG).
AdvDiffusionSLCN : class
    Advection-diffusion with semi-Lagrangian transport (flux history).
NavierStokes : class
    Navier-Stokes composed from a DDt transport manager (EulerianSUPG default).
NavierStokesSLCN : class
    Navier-Stokes with semi-Lagrangian transport and a stress history.
Diffusion : class
    Pure diffusion (no advection).
TransientDarcy : class
    Transient groundwater flow with constant storage.
Richards : class
    Richards equation for variably-saturated flow.

Time Derivative Schemes
-----------------------
Lagrangian_DDt, SemiLagragian_DDt, Eulerian_DDt, EulerianSUPG_DDt
    Time derivative approximations for transient problems; EulerianSUPG_DDt
    is the transport plugin of the Eulerian solvers (assembled advection, SUPG).

See Also
--------
underworld3.constitutive_models : Material rheology definitions.
underworld3.discretisation : Mesh and variable classes.
"""
from underworld3.cython.generic_solvers import (
    SNES_Scalar,
    SNES_Vector,
    SNES_Stokes_SaddlePt,
    SNES_MultiComponent,
)

from .solvers import SNES_Poisson as Poisson
from .solvers import SNES_Darcy as SteadyStateDarcy
from .solvers import SNES_Stokes as Stokes
from .solvers import SNES_Stokes_Constrained as Stokes_Constrained
from .solvers import SNES_VE_Stokes as VE_Stokes
from .solvers import SNES_Projection as Projection
from .solvers import SNES_Vector_Projection as Vector_Projection
from .solvers import SNES_Tensor_Projection as Tensor_Projection
from .solvers import SNES_MultiComponent_Projection as MultiComponent_Projection

# from .solvers import SNES_Solenoidal_Vector_Projection as Solenoidal_Vector_Projection  ## WIP / maybe some issues
# from .solvers import (
#     SNES_AdvectionDiffusion_SLCN as AdvDiffusion,
# )  # fix examples then remove this


# These are now implemented the same way using the ddt module
from .solvers import SNES_AdvectionDiffusion as AdvDiffusionSLCN
# The generic names are the composing solvers: the transport (assembled SUPG
# advection, or a semi-Lagrangian history) is the DDt manager they hold.
from .advection_diffusion_eulerian import SNES_AdvectionDiffusion_Composed as AdvDiffusion
from .navier_stokes_eulerian import SNES_NavierStokes_Composed as NavierStokes

# import diffusion-only solver
from .solvers import SNES_Diffusion as Diffusion

# Transient Darcy and Richards solvers
from .solvers import SNES_TransientDarcy as TransientDarcy
from .solvers import SNES_Richards as Richards

# These are now implemented the same way using the ddt module
from .solvers import SNES_NavierStokes as NavierStokesSwarm
from .solvers import SNES_NavierStokes as NavierStokesSLCN

from .free_surface import FreeSurface

# What solve_report.sub holds — one entry per fieldsplit block (see solver_health).
from .solver_health import SubSolveReport

# are the Lagrangian implementations actually distinct in reality ?
from .ddt import Lagrangian as Lagrangian_DDt
from .ddt import SemiLagrangian as SemiLagragian_DDt
from .ddt import IntegrationPointSemiLagrangian as IntegrationPointSemiLagrangian_DDt
from .ddt import Lagrangian_Swarm as Lagrangian_Swarm_DDt
from .ddt import Eulerian as Eulerian_DDt
from .ddt import EulerianSUPG as EulerianSUPG_DDt

# δ-continuation driver for hard viscoplastic (Drucker–Prager) yield
from .yield_continuation import yield_continuation, YieldHomotopyControl
from .solve_report import SolveReport
