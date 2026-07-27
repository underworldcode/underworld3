# cython: language_level=3
"""
Safe Cython wrapper for PETSc DMField C objects.

DMField is a PETSc object that evaluates a field and its spatial derivatives
at arbitrary points using the FE basis functions directly — no L2 projection,
no mass-matrix solve.
"""

import numpy as np
cimport numpy as np

# Import PETSc Cython types
from petsc4py.PETSc cimport DM, PetscDM
from petsc4py.PETSc cimport Vec, PetscVec

# Declare PETSc integer/error types locally
cdef extern from "petsc.h" nogil:
    ctypedef int PetscErrorCode
    ctypedef int PetscInt


cdef class CachedDMField:
    """Python-managed wrapper for a PETSc DMField C object.

    The DMField is created once from a MeshVariable and then reused for
    multiple evaluate() calls, as long as the variable's data is current.
    """

    cdef public object _field_handle  # Python int — DMField pointer as integer
    cdef public object mesh
    cdef public object source_var
    cdef public bint is_valid
    cdef public int nc               # num_components
    cdef public int dim              # spatial dimension
    cdef public object _comm         # MPI communicator (matches mesh DM)

    def __cinit__(self):
        self._field_handle = 0
        self.is_valid = False
        self.nc = 0
        self.dim = 0
        self._comm = None

    def create(self, mesh, source_var):
        """Create the DMField from a MeshVariable via PETSc C API.

        Uses ``ctypes`` to call ``DMFieldCreateDS`` so we never have to
        wrestle with the opaque ``DMField`` pointer type in Cython's type
        system.
        """
        import os
        import ctypes
        from ctypes import c_int, c_void_p, POINTER, byref

        # Locate libpetsc
        petsc_dir = os.environ.get("PETSC_DIR", "")
        petsc_arch = os.environ.get("PETSC_ARCH", "")
        lib_dir = os.path.join(petsc_dir, petsc_arch, "lib")
        if not os.path.isdir(lib_dir):
            lib_dir = os.path.join(petsc_dir, "lib")
        lib_path = os.path.join(lib_dir, "libpetsc.dylib")
        if not os.path.exists(lib_path):
            raise RuntimeError(f"Cannot find libpetsc at {lib_path}")

        lib = ctypes.CDLL(lib_path)
        lib.DMFieldCreateDS.argtypes = [c_void_p, c_int, c_void_p, POINTER(c_void_p)]
        lib.DMFieldCreateDS.restype = c_int

        field_ptr = c_void_p(None)
        ierr = lib.DMFieldCreateDS(
            mesh.dm.handle,
            source_var.field_id,
            mesh.lvec.handle,
            byref(field_ptr),
        )
        if ierr != 0:
            raise RuntimeError(
                f"DMFieldCreateDS failed for field "
                f"'{source_var.clean_name}' (field_id={source_var.field_id}) "
                f"with error {ierr}"
            )

        self._field_handle = field_ptr.value
        self.nc = source_var.num_components
        self.dim = mesh.dim
        self._comm = mesh.dm.comm
        self.mesh = mesh
        self.source_var = source_var
        self.is_valid = True

    def evaluate(self, coords_array, gradient=True, hessian=False):
        """Evaluate the field (and its derivatives) at *coords*.

        Parameters
        ----------
        coords_array : ndarray (n_points, dim)
            Non-dimensional [0-1] coordinates to evaluate at.
        gradient : bool
            If True, return first derivatives D.
        hessian : bool
            If False, return second derivatives H.

        Returns
        -------
        B : ndarray (n_points, nc) or None
            Field values.
        D : ndarray (n_points, dim, nc) or None
            First derivatives: D[k, i, j] = ∂(component j)/∂xᵢ at point k.
        H : ndarray (n_points, dim, dim, nc) or None
            Second derivatives.
        """
        if not self.is_valid:
            raise RuntimeError("Cannot evaluate destroyed CachedDMField")

        # DMFieldEvaluate internally calls DMLocatePoints which needs
        # localized coordinates.  Idempotent — safe on every call.
        self.mesh.dm.localizeCoordinates()

        import ctypes
        from ctypes import c_int, c_void_p
        import os
        from petsc4py import PETSc as PyPETSc

        n_points = coords_array.shape[0]
        dim = self.dim
        nc = self.nc

        # Build points Vec
        pts_flat = np.ascontiguousarray(coords_array.ravel(), dtype=np.float64)
        pts_py = PyPETSc.Vec().createWithArray(pts_flat, comm=self._comm)
        pts_py.setBlockSize(dim)

        # Allocate output arrays
        B = np.empty(n_points * nc, dtype=np.float64) if nc > 0 else None
        D = np.empty(n_points * nc * dim, dtype=np.float64) if gradient and nc > 0 else None
        H = np.empty(n_points * nc * dim * dim, dtype=np.float64) if hessian and nc > 0 else None

        # Call via ctypes
        lib_path = os.path.join(
            os.environ["PETSC_DIR"],
            os.environ.get("PETSC_ARCH", ""),
            "lib", "libpetsc.dylib"
        )
        lib = ctypes.CDLL(lib_path)
        lib.DMFieldEvaluate.argtypes = [c_void_p, c_void_p, c_int, c_void_p, c_void_p, c_void_p]
        lib.DMFieldEvaluate.restype = c_int

        ierr = lib.DMFieldEvaluate(
            c_void_p(self._field_handle),
            pts_py.handle,
            0,  # PETSC_REAL
            B.ctypes.data_as(c_void_p) if B is not None and B.size else ctypes.c_void_p(),
            D.ctypes.data_as(c_void_p) if D is not None and D.size else ctypes.c_void_p(),
            H.ctypes.data_as(c_void_p) if H is not None and H.size else ctypes.c_void_p(),
        )
        pts_py.destroy()
        if ierr != 0:
            raise RuntimeError(f"DMFieldEvaluate failed with error {ierr}")

        # Reshape
        B_out = B.reshape(n_points, nc) if B is not None else None
        D_out = D.reshape(n_points, dim, nc) if D is not None else None
        H_out = H.reshape(n_points, dim, dim, nc) if H is not None else None
        return B_out, D_out, H_out

    def destroy(self):
        if not self.is_valid:
            return
        try:
            import ctypes
            from ctypes import c_int, c_void_p, POINTER
            import os

            lib_path = os.path.join(
                os.environ["PETSC_DIR"],
                os.environ.get("PETSC_ARCH", ""),
                "lib", "libpetsc.dylib"
            )
            lib = ctypes.CDLL(lib_path)
            lib.DMFieldDestroy.argtypes = [POINTER(c_void_p)]
            lib.DMFieldDestroy.restype = c_int

            ptr = c_void_p(self._field_handle)
            lib.DMFieldDestroy(ctypes.byref(ptr))
        except Exception:
            pass  # during interpreter shutdown modules may be gone
        finally:
            self._field_handle = 0
            self.is_valid = False

    def __dealloc__(self):
        self.destroy()

    def __repr__(self):
        status = "valid" if self.is_valid else "destroyed"
        var = self.source_var
        return (f"CachedDMField({status}, "
                f"var='{var.clean_name if var else '?'}', "
                f"nc={self.nc}, dim={self.dim})")
