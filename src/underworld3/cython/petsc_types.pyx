from libc.stdlib cimport malloc

cdef class PtrContainer:

    cpdef allocate(self, int n_res, int n_bcs, int n_jac, int n_bd_res, int n_bd_jac):
        """Allocate function pointer arrays of the given sizes."""
        self.fns_residual    = <PetscDSResidualFn*>   malloc(n_res    * sizeof(PetscDSResidualFn))
        self.fns_bcs         = <PetscDSResidualFn*>   malloc(n_bcs    * sizeof(PetscDSResidualFn))
        self.fns_jacobian    = <PetscDSJacobianFn*>   malloc(n_jac    * sizeof(PetscDSJacobianFn))
        self.fns_bd_residual = <PetscDSBdResidualFn*> malloc(n_bd_res * sizeof(PetscDSBdResidualFn))
        self.fns_bd_jacobian = <PetscDSBdJacobianFn*> malloc(n_bd_jac * sizeof(PetscDSBdJacobianFn))

    cpdef copy_residual_from(self, int dst, PtrContainer src, int src_idx):
        """Copy a residual function pointer from another container."""
        self.fns_residual[dst] = src.fns_residual[src_idx]

    cpdef copy_bcs_from(self, int dst, PtrContainer src, int src_idx):
        """Copy a BC function pointer from another container."""
        self.fns_bcs[dst] = src.fns_bcs[src_idx]

    cpdef copy_jacobian_from(self, int dst, PtrContainer src, int src_idx):
        """Copy a Jacobian function pointer from another container."""
        self.fns_jacobian[dst] = src.fns_jacobian[src_idx]

    cpdef copy_bd_residual_from(self, int dst, PtrContainer src, int src_idx):
        """Copy a boundary residual function pointer from another container."""
        self.fns_bd_residual[dst] = src.fns_bd_residual[src_idx]

    cpdef copy_bd_jacobian_from(self, int dst, PtrContainer src, int src_idx):
        """Copy a boundary Jacobian function pointer from another container."""
        self.fns_bd_jacobian[dst] = src.fns_bd_jacobian[src_idx]
