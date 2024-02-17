#!/usr/bin/env python

import underworld3 as uw
import numpy as np
import io
import sys
from collections import UserString
from contextlib import redirect_stdout, redirect_stderr


# # Capture the stdout to an object
# class CaptureStdout(list):
#     def __enter__(self, split=True):
#         self._stdout = sys.stdout
#         self.split = split
#         sys.stdout = self._stringio = StringIO()
#         return self

#     def __exit__(self, *args):
#         if split:
#             self.extend(self._stringio.getvalue().splitlines())
#         else:
#             self.extend(self._stringio.getvalue()
#         del self._stringio  # free up some memory
#         sys.stdout = self._stdout


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
