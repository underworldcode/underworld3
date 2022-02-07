from .stokes import Stokes as old_Stokes
from .navier_stokes import NavierStokes
from .poisson import Poisson as old_Poisson
from .adv_diff_poisson import AdvDiffusion
from .projection import Projection as old_Projection
from .generic_solvers import SNES_Scalar, SNES_SaddlePoint 

from .solvers import SNES_Poisson     as Poisson 
from .solvers import SNES_Stokes      as Stokes
from .solvers import SNES_Projection  as Projection
