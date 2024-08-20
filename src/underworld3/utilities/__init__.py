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
from ._utils import CaptureStdout, h5_scan, mem_footprint, gather_data

from .read_medit_ascii import read_medit_ascii, print_medit_mesh_info
from .create_dmplex_from_medit import create_dmplex_from_medit

class _uw_record():

    def __init__(self):
        """
        A class to record runtime information about the underworld3 execution environment.
        """
        try: 
            import mpi4py
            comm = mpi4py.MPI.COMM_WORLD
        except ImportError:
            raise ImportError("Can't import mpi4py for runtime information.")


        # rank 0 only builds the data and then broadcasts it
        self._install_data = None
        self._runtime_data = None
        if comm.rank == 0:

            import sys
            import datetime
            import subprocess
            import warnings

            # get the start time of this piece of code
            start_t = datetime.datetime.now().isoformat()

            # get the git version
            try:
                gv = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
            except Exception as e:
                gv = None
                warnings.warn( f"Warning: Underworld can't retrieving commit hash: {e}" )

            # get petsc information
            try:
                import petsc4py as _petsc4py
                from petsc4py import PETSc as _PETSc
                petsc_version = _PETSc.Sys.getVersion()
                petsc_dir = _petsc4py.get_config()['PETSC_DIR']
            except Exception as e:
                petsc_version = None
                petsc_dir = None
                warnings.warn( f"Warning: Underworld can't retrieving petsc installation details: {e}" )

            # get h5py information
            try:
                import h5py as _h5py
                h5py_dir = _h5py.__file__
                h5py_version = _h5py.version.version
                hdf5_version = _h5py.version.hdf5_version
            except Exception as e:
                h5py_dir = None
                h5py_version = None
                hdf5_version = None
                warnings.warn( f"Warning: Underworld can't retrieving h5py installation details: {e}" )

            # get mpi4py information
            try:
                import mpi4py as _mpi4py
                mpi4py_version = _mpi4py.__version__
            except Exception as e:
                mpi4py_version = None
                warnings.warn( f"Warning: Underworld can't retrieving mpi4py installation details: {e}" )

            # get just the version
            from underworld3 import __version__ as uw_version

            self._install_data = {
                "git_version": gv,
                "uw_version": uw_version,
                "python_versions": sys.version,
                "petsc_version": petsc_version,
                "petsc_dir": petsc_dir,
                "h5py_version": h5py_version,
                "hdf5_version": hdf5_version,
                "h5py_dir": h5py_dir,
                "mpi4py_version": mpi4py_version,
            }

            self._runtime_data = {
                "start_time": start_t,
                "uw_object_count": 0,
            }

        # rank 0 broadcast information to other procs
        self._install_data = comm.bcast(self._install_data, root=0)

    @property
    def get_installation_data(self):
        return self._install_data

    @property
    def get_runtime_data(self):
        import datetime
        import mpi4py
        comm = mpi4py.MPI.COMM_WORLD

        if comm.rank == 0:
            now = datetime.datetime.now().isoformat()
            self._runtime_data.update({"current_time": now})

            from underworld3.utilities._api_tools import uw_object
            object_count = uw_object.uw_object_counter
            self._runtime_data.update({"uw_object_count": object_count})

        self._runtime_data = comm.bcast(self._runtime_data, root=0)
        return self._runtime_data

uw_record = _uw_record()

