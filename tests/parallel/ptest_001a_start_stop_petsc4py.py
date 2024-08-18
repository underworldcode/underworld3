import petsc4py
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy


# from underworld3.cython import petsc_discretisation

dmplex = PETSc.DMPlex().createFromFile("annulus_example.msh")
dmplex.distribute()

dim = dmplex.getDimension()
cdim = dmplex.getCoordinateDim()

print(f"{uw.mpi.rank} - coordinate projection begins", flush=True)

# Set up the coords for dmplex - we should not need petsc_fe
# but petsc4py does not permit a None argument

options = PETSc.Options()
options.setValue("meshproj_{}_petscspace_degree", 1)

petsc_fe = PETSc.FE().createDefault(
    dim,
    cdim,
    True,
    3,
    "meshproj_",
    PETSc.COMM_WORLD,
)

# dmplex.projectCoordinates(petsc_fe)

if (
    PETSc.Sys.getVersion() <= (3, 20, 5)
    and PETSc.Sys.getVersionInfo()["release"] == True
        ):
    dmplex.projectCoordinates(petsc_fe)
else:
    dmplex.setCoordinateDisc(petsc_fe, project=False)

# Coordinate calculations (as mesh does)
dmc = dmplex.getCoordinateDM()
dmc.createDS()
dmnew = dmc.clone()

options = PETSc.Options()
options["coordinterp_petscspace_degree"] = 1
options["coordinterp_petscdualspace_lagrange_continuity"] = False
options["coordinterp_petscdualspace_lagrange_node_endpoints"] = False

dmfe = PETSc.FE().createDefault(dim, cdim, True, 3, "coordinterp_", PETSc.COMM_WORLD)
dmnew.setField(0, dmfe)
dmnew.createDS()

print(f"{uw.mpi.rank} - coordinate interpolation begins", flush=True)


matInterp, vecScale = dmc.createInterpolation(dmnew)
coordsOld = dmplex.getCoordinates()
coordsNewL = dmnew.getLocalVec()
coordsNewG = matInterp * coordsOld
dmnew.globalToLocal(coordsNewG, coordsNewL)

arr = coordsNewL.array
coords = arr.reshape(-1, 2).copy()
print(f"{uw.mpi.rank}:", coords.shape)
