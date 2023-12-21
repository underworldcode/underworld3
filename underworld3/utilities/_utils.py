#!/usr/bin/env python

import underworld3 as uw
import numpy as np
import io
import sys
from collections import UserString
from contextlib import redirect_stdout, redirect_stderr
from mpi4py import MPI


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


def gather_data(array, bcast=False):
    """
    gather values on root (bcast=False) or all (bcast = True) processors
    Parameters:
        array : Numpy array to combine into a single array on the root or all processors

    returns:
        val_global : combination of values form all processors

    """

    comm = uw.mpi.comm
    rank = uw.mpi.rank
    

    dtype = array.dtype
    # Determine the total size of the array on the root processor
    total_size = comm.reduce(array.size, op=MPI.SUM, root=0)

    
    # Create the receive buffer on the root processor
    if rank == 0:
        recv_data = np.empty(total_size, dtype=dtype)
    else:
        recv_data = None
    
    # Gather the sizes of the send buffers
    send_sizes = comm.gather(array.size, root=0)

    # Calculate the total size of the array on the root processor
    if rank == 0:
        total_size = np.sum(send_sizes)
    else:
        total_size = None
    
    # Create the receive buffer on the root processor
    if rank == 0:
        val_global = np.empty(total_size, dtype=dtype)
    else:
        val_global = None
    
    # Calculate the displacements for Gatherv
    if rank == 0:
        displacements = np.insert(np.cumsum(send_sizes), 0, 0)[0:-1]
    else:
        displacements = None
    
    # Gather the arrays on the root processor
    comm.Gatherv(sendbuf=array, recvbuf=(val_global, send_sizes, displacements, MPI.DOUBLE), root=0)


    if bcast == True:
        val_global = comm.bcast(val_global, root=0)

    return val_global
