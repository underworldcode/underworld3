from petsc4py import PETSc
# pop the default petsc Signal handler to let petsc errors appear in python
# unclear if this is the appropriate way see discussion
# https://gitlab.com/petsc/petsc/-/issues/1066
PETSc.Sys.popErrorHandler()

#PETSc.Log().begin()

import underworld3._api_tools
import underworld3.discretisation
import underworld3.meshing
import underworld3.maths
import underworld3.swarm
import underworld3.systems
import underworld3.systems.tensors
import underworld3.systems.constitutive_models
import underworld3.tools
import underworld3.algorithms
import underworld3.mpi

from enum import Enum as _Enum
class VarType(_Enum):
    SCALAR=1
    VECTOR=2
    OTHER=3  # add as required 


import numpy as _np
# Info for JIT modules.
# These dicts should be populated by submodules
# which define cython/c based classes.
# We use ordered dictionaries because the
# ordering can be important when linking in libraries.
# Note that actually what we want is an ordered set (which Python
# doesn't natively provide). Hence for the key/value pair,
# the value is always set to `None`.

from collections import OrderedDict as _OD
_libfiles = _OD()
_libdirs  = _OD()
_incdirs  = _OD({_np.get_include():None})

def _is_notebook() -> bool:
    """
    Function to determine if the python environment is a Notebook or not.

    Returns 'True' if executing in a notebook, 'False' otherwise

    Script taken from https://stackoverflow.com/a/39662359/8106122
    """

    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

is_notebook = _is_notebook()
