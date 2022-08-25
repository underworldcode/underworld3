from petsc4py.PETSc cimport DM,  PetscDM
from petsc4py.PETSc cimport DS,  PetscDS
from petsc4py.PETSc cimport Vec, PetscVec
from petsc4py.PETSc cimport Mat, PetscMat
from petsc4py.PETSc cimport IS,  PetscIS
from petsc4py.PETSc cimport FE,  PetscFE
from petsc4py.PETSc cimport PetscDMLabel
from petsc4py.PETSc cimport PetscQuadrature
from petsc4py.PETSc cimport MPI_Comm, PetscMat, GetCommDefault, PetscViewer

from underworld3.cython.petsc_types cimport PetscBool, PetscInt, PetscReal, PetscScalar
from underworld3.cython.petsc_types cimport PetscErrorCode 
from underworld3.cython.petsc_types cimport DMBoundaryConditionType
from underworld3.cython.petsc_types cimport PetscDSResidualFn, PetscDSJacobianFn
from underworld3.cython.petsc_types cimport PtrContainer
from underworld3.utilities.petsc_gen_xdmf import generateXdmf

ctypedef enum PetscBool:
    PETSC_FALSE
    PETSC_TRUE

cdef CHKERRQ(PetscErrorCode ierr):
    cdef int interr = <int>ierr
    if ierr != 0: raise RuntimeError(f"PETSc error code '{interr}' was encountered.\nhttps://www.mcs.anl.gov/petsc/petsc-current/include/petscerror.h.html")

cdef extern from "petsc_compat.h":
    PetscErrorCode PetscDSAddBoundary_UW( PetscDM, DMBoundaryConditionType, const char[], const char[] , PetscInt, PetscInt, const PetscInt *,                                                      void (*)(), void (*)(), PetscInt, const PetscInt *, void *)
    PetscErrorCode DMSetAuxiliaryVec_UW(PetscDM, PetscDMLabel, PetscInt, PetscInt, PetscVec)

cdef extern from "petsc.h" nogil:
    PetscErrorCode DMPlexSNESComputeBoundaryFEM( PetscDM, void *, void *)
    PetscErrorCode DMPlexSetSNESLocalFEM( PetscDM, void *, void *, void *)
    PetscErrorCode DMPlexComputeGeometryFVM( PetscDM dm, PetscVec *cellgeom, PetscVec *facegeom)
    PetscErrorCode MatInterpolate(PetscMat A, PetscVec x, PetscVec y)
    PetscErrorCode PetscDSSetJacobian( PetscDS, PetscInt, PetscInt, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn)
    PetscErrorCode PetscDSSetJacobianPreconditioner( PetscDS, PetscInt, PetscInt, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn)
    PetscErrorCode PetscDSSetResidual( PetscDS, PetscInt, PetscDSResidualFn, PetscDSResidualFn )

    # These do not appear to be in the 3.17.2 release
    PetscErrorCode DMProjectCoordinates(PetscDM dm, PetscFE disc)
    PetscErrorCode DMCreateSubDM(PetscDM, PetscInt, const PetscInt *, PetscIS *, PetscDM *)
    PetscErrorCode DMDestroy(PetscDM *dm)


