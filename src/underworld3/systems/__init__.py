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
    Advection-diffusion; ``transport=`` chooses how the field is carried:
    "eulerian" (assembled, SUPG; the default) or a semi-Lagrangian history.
NavierStokes : class
    Navier-Stokes; ``velocity_transport=`` chooses how the momentum is
    carried ("eulerian" or a semi-Lagrangian history) and, for a
    viscoelastic material, ``stress_transport`` how the stress is.
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
from .solvers import SNES_AdvectionDiffusion_Swarm as AdvDiffusionSwarm
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

from .free_surface import FreeSurface

# What solve_report.sub holds — one entry per fieldsplit block (see solver_health).
from .solver_health import SubSolveReport

# are the Lagrangian implementations actually distinct in reality ?
from .ddt import Lagrangian as Lagrangian_DDt
# former names of three of the four schemes ddt.SemiLagrangian selects
from .ddt import BackwardNodesSemiLagrangian as SemiLagragian_DDt
from .ddt import BackwardIntegrationPointsSemiLagrangian as IntegrationPointSemiLagrangian_DDt
from .ddt import ForwardIntegrationPointsSemiLagrangian as ForwardSemiLagrangian_DDt
from .ddt import Lagrangian_Swarm as Lagrangian_Swarm_DDt
from .ddt import Eulerian as Eulerian_DDt
from .ddt import EulerianSUPG as EulerianSUPG_DDt

# δ-continuation driver for hard viscoplastic (Drucker–Prager) yield
from .yield_continuation import yield_continuation, YieldHomotopyControl
from .solve_report import SolveReport


# Former names. One solver per equation, the transport chosen by argument:
# NavierStokes(velocity_transport=...), AdvDiffusion(transport=...).
_FORMER_NAMES = {
    # former name: (the implementation it still returns, the replacement, what differs)
    "NavierStokesSLCN": ("SNES_NavierStokes",
                         "uw.systems.NavierStokes(..., velocity_transport='backward_nodes')",
                         "defaults differ: order 1 (was 2), rho 1 (was 0), p_continuous True (was False), "
                         "no velocity-history smoothing (was 1e-4); estimate_dt returns one number "
                         "(was a (diffusive, advective) pair); the order is fixed at construction "
                         "(solve(order=) is refused) and flux_order is not an argument"),
    "NavierStokesSwarm": ("SNES_NavierStokes",
                          "uw.systems.NavierStokes(..., velocity_transport='backward_nodes')",
                          "as for NavierStokesSLCN; a particle velocity history is velocity_transport='lagrangian'"),
    "AdvDiffusionSLCN": ("SNES_AdvectionDiffusion",
                         "uw.systems.AdvDiffusion(..., transport='backward_nodes')",
                         "order is the order of the value history (BDF2 at order 2, theta 1; the old "
                         "solver kept BDF1 and raised the flux rule's order); the stored-level flux is "
                         "rebuilt from the carried field, not traced; estimate_dt() defaults to the "
                         "cell-crossing time for a semi-Lagrangian transport; the SUPG options are "
                         "refused off the Eulerian path"),
}


def __getattr__(name):
    if name in _FORMER_NAMES:
        import warnings
        from . import solvers
        implementation, instead, differs = _FORMER_NAMES[name]
        warnings.warn(f"uw.systems.{name} is deprecated and still returns the old implementation "
                      f"(solvers.{implementation}); use {instead}. Note: {differs}.",
                      FutureWarning, stacklevel=2)
        return getattr(solvers, implementation)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
