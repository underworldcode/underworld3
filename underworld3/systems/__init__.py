# Underworld systems includes solvers and constitutive equations

from underworld3.cython.generic_solvers import (
    SNES_Scalar,
    SNES_Vector,
    SNES_Stokes_SaddlePt,
)

from .solvers import SNES_Poisson as Poisson
from .solvers import SNES_Darcy as SteadyStateDarcy
from .solvers import SNES_Stokes as Stokes
from .solvers import SNES_Projection as Projection
from .solvers import SNES_Vector_Projection as Vector_Projection
from .solvers import SNES_Tensor_Projection as Tensor_Projection

# from .solvers import SNES_Solenoidal_Vector_Projection as Solenoidal_Vector_Projection  ## WIP / maybe some issues
from .solvers import (
    SNES_AdvectionDiffusion_SLCN as AdvDiffusion,
)  # fix examples then remove this

from .solvers import SNES_AdvectionDiffusion_SLCN as AdvDiffusionSLCN
from .solvers import SNES_AdvectionDiffusion_Swarm as AdvDiffusionSwarm
from .solvers import SNES_NavierStokes_Swarm as NavierStokesSwarm
from .solvers import SNES_NavierStokes_SLCN as NavierStokesSLCN
from .solvers import SNES_NavierStokes_Swarm as NavierStokesSwarm
