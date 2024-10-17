import underworld3 as uw
import numpy as np
import sympy

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import petsc4py
from petsc4py import PETSc

h5plex = PETSc.DMPlex().createFromFile(".meshes/uw_annulus_ro1.0_ri0.5_csize0.1.msh.h5")

print(f"{rank} - DM created from file - v1", flush=True)

h5plex.distribute()

print(f"{rank} - DM distribution complete - v1", flush=True)


h5plex.destroy()


## v2 - this is what we might also do in uw3

viewer = PETSc.ViewerHDF5().create(
    ".meshes/uw_annulus_ro1.0_ri0.5_csize0.1.msh.h5", "r"
)
h5plex = PETSc.DMPlex().create()
sf0 = h5plex.topologyLoad(viewer)
h5plex.coordinatesLoad(viewer, sf0)
h5plex.labelsLoad(viewer, sf0)

# Do this as well
h5plex.setName("uw_mesh")
h5plex.markBoundaryFaces("All_Boundaries", 1001)

print(f"{rank} - DM created from file - v2", flush=True)

h5plex.distribute()

print(f"{rank} - DM distribution complete - v2", flush=True)

h5plex.destroy()
