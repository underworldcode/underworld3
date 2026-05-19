

from petsc4py.PETSc cimport DM,  PetscDM
from petsc4py.PETSc cimport DS,  PetscDS
from petsc4py.PETSc cimport Vec, PetscVec
from petsc4py.PETSc cimport Mat, PetscMat
from petsc4py.PETSc cimport IS,  PetscIS
from petsc4py.PETSc cimport SF,  PetscSF
from petsc4py.PETSc cimport FE,  PetscFE
from petsc4py.PETSc cimport DMLabel, PetscDMLabel
from petsc4py.PETSc cimport PetscQuadrature, PetscSection
from petsc4py.PETSc cimport MPI_Comm, PetscMat, GetCommDefault, Viewer, PetscViewer


from underworld3.cython.petsc_types cimport PetscBool, PetscInt, PetscReal, PetscScalar
from underworld3.cython.petsc_types cimport PetscErrorCode
from underworld3.cython.petsc_types cimport DMBoundaryConditionType
from underworld3.cython.petsc_types cimport PetscDMBoundaryConditionType
from underworld3.cython.petsc_types cimport PetscDMBoundaryType
from underworld3.cython.petsc_types cimport PetscDSResidualFn, PetscDSJacobianFn
from underworld3.cython.petsc_types cimport PetscDSBdResidualFn, PetscDSBdJacobianFn
from underworld3.cython.petsc_types cimport PtrContainer

from underworld3.utilities import generateXdmf

ctypedef enum PetscBool:
    PETSC_FALSE
    PETSC_TRUE

cdef CHKERRQ(PetscErrorCode ierr):
    cdef int interr = <int>ierr
    if ierr != 0: raise RuntimeError(f"PETSc error code '{interr}' was encountered.\nhttps://www.mcs.anl.gov/petsc/petsc-current/include/petscerror.h.html")

cdef extern from "petsc_compat.h":

    PetscErrorCode PetscDSAddBoundary_UW( PetscDM, DMBoundaryConditionType, const char[], const char[] , PetscInt, PetscInt, PetscInt *, void (*)(), void (*)(), PetscInt, const PetscInt *, void *)
    PetscErrorCode DMSetAuxiliaryVec_UW(PetscDM, PetscDMLabel, PetscInt, PetscInt, PetscVec)
    # PetscErrorCode UW_PetscDSSetBdResidual(PetscDS, PetscDMLabel, PetscInt, PetscInt, PetscInt, PetscInt, void*, PetscInt, void*)

    PetscErrorCode UW_PetscDSSetBdResidual(PetscDS, PetscDMLabel, PetscInt, PetscInt, PetscInt, PetscInt, void*, void*)
    PetscErrorCode UW_PetscDSSetBdJacobian(PetscDS, PetscDMLabel, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, void*, void*, void*, void*)
    PetscErrorCode UW_PetscDSSetBdJacobianPreconditioner(PetscDS, PetscDMLabel, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, void*, void*, void*, void*)
    PetscErrorCode UW_PetscDSSetBdTerms   (PetscDS, PetscDMLabel, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, void*, void*, void*, void*, void*, void* )
    PetscErrorCode UW_PetscDSViewWF(PetscDS)     
    PetscErrorCode UW_PetscDSViewBdWF(PetscDS, PetscInt)
    PetscErrorCode UW_DMSetTime( PetscDM, PetscReal )
    PetscErrorCode UW_DMPlexSetSNESLocalFEM( PetscDM, PetscBool, void *)
    PetscErrorCode UW_DMForceCoordinateField(PetscDM)
    PetscErrorCode UW_DMPlexComputeBdIntegral( PetscDM, PetscVec, PetscDMLabel, PetscInt, const PetscInt*, void*, PetscScalar*, void*)
    PetscErrorCode UW_DMCreateBdIntegralSandbox(PetscDM, PetscDM*)

cdef extern from "petsc.h" nogil:
    PetscErrorCode PetscDSSetConstants(PetscDS, PetscInt, const PetscScalar[])
    PetscErrorCode DMPlexSNESComputeBoundaryFEM( PetscDM, void *, void *)
    # PetscErrorCode DMPlexSetSNESLocalFEM( PetscDM, void *, void *, void *)
    # PetscErrorCode DMPlexSetSNESLocalFEM( PetscDM, PetscBool, void *)
    PetscErrorCode DMPlexComputeGeometryFVM( PetscDM dm, PetscVec *cellgeom, PetscVec *facegeom)
    PetscErrorCode MatInterpolate(PetscMat A, PetscVec x, PetscVec y)
    PetscErrorCode DMSetLocalSection(PetscDM, PetscSection)
    
    PetscErrorCode PetscDSSetJacobian( PetscDS, PetscInt, PetscInt, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn)
    PetscErrorCode PetscDSSetJacobianPreconditioner( PetscDS, PetscInt, PetscInt, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn)
    PetscErrorCode PetscDSSetResidual( PetscDS, PetscInt, PetscDSResidualFn, PetscDSResidualFn )
    
    PetscErrorCode PetscDSSetBdJacobian( PetscDS, PetscInt, PetscInt, PetscDSBdJacobianFn, PetscDSBdJacobianFn, PetscDSBdJacobianFn, PetscDSBdJacobianFn)
    PetscErrorCode PetscDSSetBdJacobianPreconditioner( PetscDS, PetscInt, PetscInt, PetscDSBdJacobianFn, PetscDSBdJacobianFn, PetscDSBdJacobianFn, PetscDSBdJacobianFn)
    PetscErrorCode PetscDSSetBdResidual( PetscDS, PetscInt, PetscDSBdResidualFn, PetscDSBdResidualFn )
    
    PetscErrorCode PetscDSAddBdJacobian( PetscDS, PetscInt, PetscInt, PetscDSBdJacobianFn, PetscDSBdJacobianFn, PetscDSBdJacobianFn, PetscDSBdJacobianFn)
    PetscErrorCode PetscDSAddBdJacobianPreconditioner( PetscDS, PetscInt, PetscInt, PetscDSBdJacobianFn, PetscDSBdJacobianFn, PetscDSBdJacobianFn, PetscDSBdJacobianFn)
    PetscErrorCode PetscDSAddBdResidual( PetscDS, PetscInt, PetscDSBdResidualFn, PetscDSBdResidualFn )

    PetscErrorCode DMPlexCreateSubmesh(PetscDM, PetscDMLabel label, PetscInt value, PetscBool markedFaces, PetscDM *subdm)
    PetscErrorCode DMPlexSectionLoad(PetscDM, PetscViewer, PetscDM, PetscSF, PetscSF *, PetscSF *)
    PetscErrorCode DMPlexLocalVectorLoad(PetscDM, PetscViewer, PetscDM, PetscSF, PetscVec)
    PetscErrorCode PetscSFDestroy(PetscSF *)
    PetscErrorCode UW_DMPlexFilter(PetscDM, PetscDMLabel, PetscInt, PetscBool, PetscBool, PetscDM *)
    PetscErrorCode DMGetLabel(PetscDM dm, const char name[], PetscDMLabel *label)

    # Region DS — per-cell discrete system dispatch
    PetscErrorCode DMSetRegionDS(PetscDM dm, PetscDMLabel label, PetscIS fields, PetscDS ds, PetscDS dsIn)
    PetscErrorCode DMGetRegionDS(PetscDM dm, PetscDMLabel label, PetscIS *fields, PetscDS *ds, PetscDS *dsIn)
    PetscErrorCode DMGetRegionNumDS(PetscDM dm, PetscInt num, PetscDMLabel *label, PetscIS *fields, PetscDS *ds, PetscDS *dsIn)
    PetscErrorCode DMSetRegionNumDS(PetscDM dm, PetscInt num, PetscDMLabel label, PetscIS fields, PetscDS ds, PetscDS dsIn)
    PetscErrorCode DMGetNumDS(PetscDM dm, PetscInt *num)
    PetscErrorCode DMGetCellDS(PetscDM dm, PetscInt point, PetscDS *ds, PetscDS *dsIn)
    PetscErrorCode PetscDSSetCoordinateDimension(PetscDS ds, PetscInt dim)

    # These do not appear to be in the 3.17.2 release
    PetscErrorCode DMProjectCoordinates(PetscDM dm, PetscFE disc)
    PetscErrorCode DMCreateSubDM(PetscDM, PetscInt, const PetscInt *, PetscIS *, PetscDM *)
    PetscErrorCode DMDestroy(PetscDM *dm)

    # Changed recently: Commit 6858538e
    # PetscErrorCode DMGetPeriodicity(PetscDM dm, PetscReal **maxCell, PetscReal **Lstart, PetscReal **L)
    PetscErrorCode DMSetPeriodicity(PetscDM dm, PetscReal maxCell[], PetscReal Lstart[], PetscReal L[])
    PetscErrorCode DMLocalizeCoordinates(PetscDM dm)

    # Not wrapped at this point
    PetscErrorCode VecConcatenate(PetscInt nx, const PetscVec X[], PetscVec *, PetscIS *)
