# A collection of utilities for the uw3 system

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
