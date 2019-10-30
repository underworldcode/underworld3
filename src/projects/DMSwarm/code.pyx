from petsc4py.PETSc cimport Vec, PetscVec
from petsc4py.PETSc cimport DM, PetscDM

#cimport code
from petsc4py.PETSc import Error

# Make a struct here for everything

cdef extern from "uw_swarm.h":
    int BuildSwarm2( PetscDM dm, int ppcell, PetscDM swarmdm)
    int DMBuildVelocityPressureFields(PetscDM dm, int is_simplex);
    int swarm_metric(PetscDM swarm);
    int BuildSwarm( PetscDM dm, int nFields, const char* fieldnames, int ppcell, PetscDM swarmdm)

def pySwarmMetric(DM swarm):
    cdef int ierr
    ierr = swarm_metric(swarm.dm)
    if ierr != 0: raise Error(ierr)

def pyBuildField(DM dm, int is_simplex):
    cdef int ierr
    ierr = DMBuildVelocityPressureFields(dm.dm, is_simplex)
    if ierr != 0: raise Error(ierr)

"""
def pyBuildSwarmNo(DM dm, int ppcell, tuple fieldnames, DM swarm):
    cdef int ierr
    cdef char* names = [] 
    nfields = len(fieldnames)
    print(fieldnames)
    ierr = BuildSwarm(dm.dm, nfields, fieldnames, ppcell, swarm.dm);
    if ierr != 0: raise Error(ierr)
"""

def pyBuildSwarm(DM dm, int ppcell, tuple fieldnames, DM swarm):
    cdef int ierr
    cdef char* names = [] 
    nfields = len(fieldnames)
    print(fieldnames)
    ierr = BuildSwarm2(dm.dm, ppcell, swarm.dm);
    if ierr != 0: raise Error(ierr)
