import petsc4py
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

h5plex = PETSc.DMPlex().createFromFile("uw_annulus_test_mesh.h5")

print(f"{rank} - DM created from file - v1", flush=True)

h5plex.distribute()

print(f"{rank} - DM distribution complete - v1", flush=True)

swarmdm = PETSc.DMSwarm().create()
swarmdm.setDimension(2)
swarmdm.setCellDM(h5plex)
swarmdm.setType(petsc4py.PETSc.DMSwarm.Type.PIC)

swarmdm.finalizeFieldRegister()

swarmdm.insertPointUsingCellDM(petsc4py.PETSc.DMSwarm.PICLayoutType.LAYOUT_GAUSS, 3)
swarmdm.migrate(remove_sent_points=True)

PIC_coords = swarmdm.getField("DMSwarmPIC_coor").reshape(-1, 2)
PIC_cellid = swarmdm.getField("DMSwarm_cellid")

print(f"{uw.mpi.rank} - {PIC_coords.shape}", flush=True)
print(f"{uw.mpi.rank} - {PIC_coords}", flush=True)
print(f"{uw.mpi.rank} - {PIC_cellid}", flush=True)


indexCoords = PIC_coords
index = uw.kdtree.KDTree(indexCoords)
index.build_index()

swarmdm.restoreField("DMSwarmPIC_coor")
swarmdm.restoreField("DMSwarm_cellid")

swarmdm.view()

swarmdm.destroy()
