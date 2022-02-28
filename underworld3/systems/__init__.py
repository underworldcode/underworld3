# This one is not much good ... needs to be
# swarm based
from .navier_stokes import NavierStokes

from .generic_solvers import SNES_Scalar, SNES_SaddlePoint 

from .solvers import SNES_Poisson                 as Poisson 
from .solvers import SNES_Stokes                  as Stokes
from .solvers import SNES_Projection              as Projection
from .solvers import SNES_AdvectionDiffusion_SLCN as AdvDiffusion