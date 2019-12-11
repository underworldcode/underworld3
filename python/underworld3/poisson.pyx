from petsc4py.PETSc cimport DM, DS, PetscDM, PetscDS
from petsc4py.PETSc import Error

cdef extern from * nogil:
    ctypedef long PetscInt
    ctypedef double PetscReal
    ctypedef double PetscScalar
    ctypedef int PetscErrorCode
    ctypedef int PetscBool

cdef extern from "poisson_setup.h":
    ctypedef struct AppCtx:
        PetscBool simplex
        PetscScalar y0
        PetscScalar y1
        PetscScalar T0
        PetscScalar T1
        PetscScalar k
        PetscScalar h

    PetscErrorCode SetupProblem(PetscDM dm, PetscDS ds, void *user);

def PoissonSetup(DM dm, DS ds, user):
    cdef AppCtx cuser
    cuser.simplex = user["simplex"]
    cuser.y0 = user["y0"]
    cuser.y1 = user["y1"]
    cuser.T0 = user["T0"]
    cuser.T1 = user["T1"]
    cuser.k = user["k"]
    cuser.h = user["h"]
    cdef int ierr
    ierr = SetupProblem(dm.dm, ds.ds, &cuser)
    if ierr != 0: raise Error(ierr)
    return
