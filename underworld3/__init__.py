import underworld3.mesh
import underworld3.maths
import underworld3.swarm
import underworld3.systems
import underworld3.tools

from enum import Enum as _Enum
class VarType(_Enum):
    SCALAR=1
    VECTOR=2
    OTHER=3  # add as required 


import numpy as _np
# Info for JIT modules.
# These lists should be populated by submodules
# which define cython/c based classes.
_libfiles = []
_libdirs  = []
_incdirs  = [_np.get_include(),]