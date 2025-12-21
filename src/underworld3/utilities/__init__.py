r"""
Utility functions and infrastructure for Underworld3.

This module provides supporting utilities including I/O helpers,
geometry tools, array callbacks, and mathematical mixins.

I/O Utilities
-------------
Xdmf, generateXdmf, generate_uw_Xdmf
    XDMF file generation for ParaView visualization.
swarm_h5, swarm_xdmf
    Swarm data I/O in HDF5 and XDMF formats.
read_medit_ascii, create_dmplex_from_medit
    Import meshes from Medit format.

Geometry Tools
--------------
Various geometric helper functions for mesh operations.

Development Utilities
---------------------
CaptureStdout
    Context manager for capturing stdout.
h5_scan
    HDF5 file inspection.
mem_footprint
    Memory usage tracking.
auditor, postHog
    Analytics and debugging tools.

Notes
-----
The units_mixin exports are DEPRECATED. Use the units system in
:mod:`underworld3.function.unit_conversion` instead.

See Also
--------
underworld3.function : Expression evaluation with unit support.
underworld3.discretisation : Mesh I/O functions.
"""
from . import _api_tools


def _append_petsc_path():
    # get/import petsc_gen_xdmf from the original petsc installation
    import sys
    import petsc4py

    conf = petsc4py.get_config()
    petsc_dir = conf["PETSC_DIR"]
    if not petsc_dir + "/lib/petsc/bin" in sys.path:
        sys.path.append(petsc_dir + "/lib/petsc/bin")


_append_petsc_path()

from .uw_petsc_gen_xdmf import Xdmf, generateXdmf, generate_uw_Xdmf
from .uw_swarmIO import swarm_h5, swarm_xdmf
from ._utils import CaptureStdout, h5_scan, mem_footprint, gather_data, auditor, postHog

from .read_medit_ascii import read_medit_ascii, print_medit_mesh_info
from .create_dmplex_from_medit import create_dmplex_from_medit
from .geometry_tools import *
from .nd_array_callback import NDArray_With_Callback
from .mathematical_mixin import MathematicalMixin

# DEPRECATED AND SCHEDULED FOR REMOVAL
# These units_mixin imports are not used and will be removed in a future version.
# DO NOT USE - see underworld3.function.unit_conversion for the active units system.
from .units_mixin import (
    UnitAwareMixin,           # DEPRECATED - REMOVE
    UnitAwareMathematicalMixin,  # DEPRECATED - REMOVE
    UnitsBackend,             # DEPRECATED - REMOVE
    PintBackend,              # DEPRECATED - REMOVE
    make_units_aware,         # DEPRECATED - REMOVE
)

from .unit_aware_array import (
    UnitAwareArray,
    create_unit_aware_array,
    zeros_with_units,
    ones_with_units,
    full_with_units,
)
