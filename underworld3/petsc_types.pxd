ctypedef long PetscInt
ctypedef double PetscReal
ctypedef double PetscScalar
ctypedef int PetscErrorCode
ctypedef int PetscBool
ctypedef int DMBoundaryConditionType
ctypedef void(*PetscDSResidualFn)(PetscInt, PetscInt, PetscInt,
                            const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                            const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                            PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[])

ctypedef void (*PetscDSJacobianFn)(PetscInt, PetscInt, PetscInt,
                                                    const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                    const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                    PetscReal, PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[])

cdef class PtrContainer:
    cdef PetscDSResidualFn residual_ptr