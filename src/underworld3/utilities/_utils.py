#!/usr/bin/env python

import underworld3 as uw
import numpy as np
import io
import sys
from collections import UserString
from contextlib import redirect_stdout, redirect_stderr

class _uw_record():
    """
    A class to record runtime information about the underworld3 execution environment.
    """

    def __init__(self):
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
                #warnings.warn( f"Warning: Underworld can't retrieving commit hash: {e}" )

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
                "hdf5_version": hdf5_version,
                "h5py_version": h5py_version,
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
        '''
        Get the installation data for the underworld3 installation.
        '''
        return self._install_data

    @property
    def get_runtime_data(self):
        '''
        Get the runtime data for the underworld3 installation.
        Note this requires a MPI broadcast to get the data.
        '''
        import datetime
        import mpi4py
        comm = mpi4py.MPI.COMM_WORLD

        if comm.rank == 0:
            now = datetime.datetime.now().isoformat()
            self._runtime_data.update({"current_time": now})

            from underworld3.utilities._api_tools import uw_object
            object_count = uw_object.uw_object_counter()
            self._runtime_data.update({"uw_object_count": object_count})

        self._runtime_data = comm.bcast(self._runtime_data, root=0)
        return self._runtime_data

auditor = _uw_record()

class CaptureStdout(UserString, redirect_stdout):
    """
    Captures stdout (e.g., from ``print()``) as a variable.

    Based on ``contextlib.redirect_stdout``, but saves the user the trouble of
    defining and reading from an IO stream. Useful for testing the output of functions
    that are supposed to print certain output.

    Citation: https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
    """

    def __init__(self, seq="", split=False, *args, **kwargs):
        self._io = io.StringIO()
        self.split = split
        UserString.__init__(self, seq=seq)
        redirect_stdout.__init__(self, self._io)
        redirect_stderr.__init__(self, self._io)
        return

    def __enter__(self, *args, **kwargs):
        redirect_stdout.__enter__(self, *args, **kwargs)
        redirect_stderr.__enter__(self, *args, **kwargs)
        return self

    def __exit__(self, *args, **kwargs):
        self.data += self._io.getvalue()
        if self.split:
            self.data = self.data.splitlines()
        redirect_stdout.__exit__(self, *args, **kwargs)
        redirect_stderr.__exit__(self, *args, **kwargs)

        return

    def start(self):
        self.__enter__()
        return self

    def stop(self):
        self.__exit__(None, None, None)
        return


# A couple of useful h5 tricks
def h5_scan(filename):
    import h5py

    h5file = h5py.File(filename)
    entities = []
    h5file.visit(entities.append)
    h5file.close()

    return entities


# A rough guide to memory usage
def mem_footprint():
    """Returns resident set size in Mb for this process"""
    import os, psutil

    pid = os.getpid()
    python_process = psutil.Process(pid)

    return python_process.memory_info().rss // 1000000


def gather_data(val, bcast=False, dtype="float64"):

    """
    gather values on root (bcast=False) or all (bcast = True) processors
    Parameters:
        vals : Values to combine into a single array on the root or all processors

    returns:
        val_global : combination of values form all processors

    """

    comm = uw.mpi.comm
    rank = uw.mpi.rank
    size = uw.mpi.size

    ### make sure all data comes in the same order
    with uw.mpi.call_pattern(pattern="sequential"):
        if len(val > 0):
            val_local = np.ascontiguousarray(val.copy())
        else:
            val_local = np.array([np.nan], dtype=dtype)


    comm.barrier()

    ### Collect local array sizes using the high-level mpi4py gather
    sendcounts = np.array(comm.gather(len(val_local), root=0))

    if rank == 0:
        val_global = np.zeros((sum(sendcounts)), dtype=dtype)

    else:
        val_global = None

    comm.barrier()

    ## gather x values, can't do them together
    comm.Gatherv(sendbuf=val_local, recvbuf=(val_global, sendcounts), root=0)

    comm.barrier()

    if uw.mpi.rank == 0:
        ### remove rows with NaN
        val_global = val_global[~np.isnan(val_global)]

    comm.barrier()

    if bcast == True:
        #### make available on all processors
        val_global = comm.bcast(val_global, root=0)

    comm.barrier()

    return val_global
