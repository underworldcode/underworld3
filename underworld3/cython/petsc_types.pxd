# Underworld Cython declarations to use PetscDS functionality
# The following aren't available via petsc4py - Oct2021

ctypedef int    PetscInt
ctypedef double PetscReal
ctypedef double PetscScalar
ctypedef int    PetscErrorCode
ctypedef int    PetscBool
ctypedef int    DMBoundaryConditionType
ctypedef int    PetscDMBoundaryConditionType
ctypedef int    PetscDMBoundaryType


## The templates for functions passed into pointwise machinery
## NOTE: this currently handles the interior integrals and boundary integrals, plus dirichlet boundary functions

ctypedef void(*PetscDSResidualFn)(PetscInt, PetscInt, PetscInt,
                            const PetscInt*, const PetscInt*, const PetscScalar*, const PetscScalar*, const PetscScalar*,
                            const PetscInt*, const PetscInt*, const PetscScalar*, const PetscScalar*, const PetscScalar*,
                            PetscReal,       const PetscReal*,PetscInt,           const PetscScalar*, PetscScalar*)

ctypedef void (*PetscDSJacobianFn)(PetscInt, PetscInt, PetscInt,
                            const PetscInt*, const PetscInt*, const PetscScalar*, const PetscScalar*, const PetscScalar*,
                            const PetscInt*, const PetscInt*, const PetscScalar*, const PetscScalar*, const PetscScalar*,
                            PetscReal,       PetscReal,       const PetscReal*,   PetscInt,           const PetscScalar*, 
                            PetscScalar*)

ctypedef void(*PetscDSBdResidualFn)(
                            PetscInt,        PetscInt,        PetscInt,
                            const PetscInt*, const PetscInt*, const PetscScalar*, const PetscScalar*, const PetscScalar*,
                            const PetscInt*, const PetscInt*, const PetscScalar*, const PetscScalar*, const PetscScalar*,
                            PetscReal,       const PetscReal*,const PetscReal*,   PetscInt, const     PetscScalar*,       
                            PetscScalar* )

ctypedef void (*PetscDSBdJacobianFn)(
                            PetscInt,           PetscInt,        PetscInt,
                            const PetscInt*,    const PetscInt*, const PetscScalar*, const PetscScalar*, const PetscScalar*,
                            const PetscInt*,    const PetscInt*, const PetscScalar*, const PetscScalar*, const PetscScalar*,
                            PetscReal,          PetscReal,       const PetscReal*,   const PetscReal*,   PetscInt,           
                            const PetscScalar*, PetscScalar* )


cdef class PtrContainer:
    cdef PetscDSResidualFn*    fns_residual
    cdef PetscDSJacobianFn*    fns_jacobian
    cdef PetscDSResidualFn*    fns_bcs
    cdef PetscDSBdResidualFn*  fns_bd_residual
    cdef PetscDSBdJacobianFn*  fns_bd_jacobian


