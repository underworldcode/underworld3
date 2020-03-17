from petsc4py.PETSc cimport DM, PetscDM
from petsc4py.PETSc import Error

cdef extern from * nogil:
    ctypedef long PetscInt
    ctypedef double PetscReal
    ctypedef double PetscScalar
    ctypedef int PetscErrorCode

#cdef extern from * nogil:
#
#    PetscErrorCode DMProjectFunction(DM dm, PetscReal time, PetscErrorCode (**funcs))


cdef extern from "functions.h":
    #void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[],
    #          const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[],
    #          const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants,
    #          const PetscScalar constants[], PetscScalar f1[]) 
    #
    #void f0_p(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[],
    #          const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[],
    #          const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[],
    #          PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]) 
    #
    #void f1_p(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[],
    #          const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[],
    #          const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[],
    #          PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
    #
    #void g1_pu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[],
    #           const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[],
    #           const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
    #           const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
    #
    #void g2_up(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[],
    #           const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[],
    #           const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
    #           const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
    #
    #void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[],
    #           const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[],
    #           const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift,
    #           const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]) 
    #
    #void f0_u_j(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[],
    #                   const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[],
    #                   const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[],
    #                   PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
    
    PetscErrorCode SetupNullSpace(PetscDM dm);
    PetscErrorCode SetupProblem(PetscDM dm, void *user);

def StokesSetup(DM dm):
    cdef int ierr
    ierr = SetupProblem(dm.dm, NULL)
    if ierr != 0: raise Error(ierr)
    return

def NullSpaceSetup(DM dm):
    cdef int ierr
    ierr = SetupNullSpace(dm.dm)
    if ierr != 0: raise Error(ierr)
    return
