# cython: language_level=3
"""
Cython wrapper for PETSc DMField — FE-exact field/gradient/Hessian evaluation.

Uses Cython extern declarations (via ``petsc_extras.pxi``) — no ctypes,
no hardcoded ``.dylib`` paths. Portable across macOS, Linux, and CI.

Parallel contract
-----------------
``DMFieldEvaluate`` internally calls ``DMLocatePoints``, which is
**COLLECTIVE on the mesh DM's communicator**.  All ranks must call,
even with zero local points.  Results are rank-local; each rank
evaluates its own coordinate set and receives its own output arrays.

Unlocated points
----------------
Output arrays are initialized to ``NaN``.  PETSc's ``DMFieldEvaluate_DS``
skips writing to slots whose cell index is negative, so unlocated points
retain their ``NaN`` initial value.  Callers detect unlocated points with
``np.isnan()``.

  **Note**: The NaN-survival behavior depends on PETSc's implementation
  of ``DMFieldEvaluate_DS`` (``continue`` on negative cell index).  The
  test ``test_unlocated_points_nan`` guards against regressions.
"""

import numpy as np
cimport numpy as np

from petsc4py import PETSc as PyPETSc
from petsc4py.PETSc cimport DM, PetscDM
from petsc4py.PETSc cimport Vec, PetscVec

include "../cython/petsc_extras.pxi"


cdef class DMFieldEvaluator:
    """Thin wrapper for a PETSc DMField — create, evaluate, destroy.

    No persistent state beyond a single ``evaluate()`` call.  Create a
    fresh instance per evaluation; the overhead (~0.5 us) is negligible
    compared to the evaluate cost (~50-200 us).
    """

    cdef _p_DMField* _dmf

    def __cinit__(self):
        self._dmf = NULL

    def create(self, mesh, source_var):
        """Create DMField from a MeshVariable via ``DMFieldCreateDS``.

        ``mesh.update_lvec()`` must be called (collectively) before this.
        """
        cdef _p_DMField* dmf = NULL
        cdef PetscDM c_dm = (<DM>mesh.dm).dm
        cdef PetscVec c_lvec = (<Vec>mesh.lvec).vec
        cdef PetscErrorCode ierr

        ierr = DMFieldCreateDS(c_dm, source_var.field_id, c_lvec, &dmf)
        CHKERRQ(ierr)
        self._dmf = dmf

    def evaluate(self, coords_array, mesh, source_var,
                 gradient=True, hessian=False):
        """Evaluate at *coords_array*, return ``(B, D, H)``.

        Parameters
        ----------
        coords_array : ndarray (n_points, dim)
            Non-dimensional [0-1] coordinates.
        mesh : Mesh
            The mesh (needed for DM/lvec access).
        source_var : MeshVariable
            The source variable (metadata only — used for nc/dim).
        gradient : bool
            Return first derivatives D (default True).
        hessian : bool
            Return second derivatives H (default False).

        Returns
        -------
        B : ndarray (n_points, nc) or None
            Field values.  Unlocated points are ``NaN``.
        D : ndarray (n_points, nc, dim) or None
            First derivatives.  ``D[k, j, i]`` = partial(component *j*) /
            partial x_i.  ``None`` when ``gradient=False``.
        H : ndarray (n_points, nc, dim, dim) or None
            Second derivatives.  ``H[k, j, i, l]`` = partial^2(component *j*) /
            partial x_i partial x_l.  ``None`` when ``hessian=False``.
        """
        cdef int n_points = coords_array.shape[0]
        cdef int dim = mesh.dim
        cdef int nc = source_var.num_components

        # DMLocatePoints is collective on the mesh DM — all ranks must
        # participate, even those with zero local points.
        mesh.dm.localizeCoordinates()

        # Build points Vec on COMM_SELF — each rank owns its own local
        # coordinates.  DMFieldEvaluate (which calls DMLocatePoints
        # internally) is collective on the DM; the Vec data is rank-local.
        pts_np = np.ascontiguousarray(coords_array.ravel(), dtype=np.float64)
        pts_py = PyPETSc.Vec().createWithArray(pts_np, comm=PyPETSc.COMM_SELF)
        pts_py.setBlockSize(dim)
        cdef PetscVec c_pts = (<Vec>pts_py).vec

        # Allocate outputs — initialised to NaN.
        # If PETSc's evaluator skips unlocated slots (the standard
        # behaviour of DMFieldEvaluate_DS), the NaN survives and
        # callers can detect unlocated points with np.isnan().
        B_arr = (np.full(n_points * nc, np.nan, dtype=np.float64)
                 if nc > 0 else None)
        D_arr = (np.full(n_points * nc * dim, np.nan, dtype=np.float64)
                 if gradient and nc > 0 else None)
        H_arr = (np.full(n_points * nc * dim * dim, np.nan, dtype=np.float64)
                 if hessian and nc > 0 else None)

        cdef PetscScalar* B_cptr = \
            <PetscScalar*>np.PyArray_DATA(B_arr) if B_arr is not None and B_arr.size > 0 else NULL
        cdef PetscScalar* D_cptr = \
            <PetscScalar*>np.PyArray_DATA(D_arr) if D_arr is not None and D_arr.size > 0 else NULL
        cdef PetscScalar* H_cptr = \
            <PetscScalar*>np.PyArray_DATA(H_arr) if H_arr is not None and H_arr.size > 0 else NULL

        cdef PetscErrorCode ierr
        ierr = DMFieldEvaluate(
            self._dmf, c_pts, PETSC_REAL, B_cptr, D_cptr, H_cptr,
        )
        pts_py.destroy()
        CHKERRQ(ierr)

        # ── Reshape to corrected layout ────────────────────────────────
        # PETSc stores point-major, then component, then direction:
        #   rD[(i * nc + j) * dimC + l]
        # So C-order reshape: (n_points, nc, dim) gives D[k, j, i].
        B_out = B_arr.reshape(n_points, nc) if B_arr is not None else None
        D_out = (D_arr.reshape(n_points, nc, dim)
                 if D_arr is not None else None)
        H_out = (H_arr.reshape(n_points, nc, dim, dim)
                 if H_arr is not None else None)

        return B_out, D_out, H_out

    def destroy(self):
        """Release the DMField.  Idempotent — safe to call multiple times."""
        cdef PetscErrorCode ierr
        if self._dmf == NULL:
            return
        try:
            ierr = DMFieldDestroy(&self._dmf)
            CHKERRQ(ierr)
        except Exception:
            # Sanctioned failure: during interpreter shutdown PETSc may
            # have already been finalised; the OS reclaims memory.
            pass
        finally:
            self._dmf = NULL

    def __dealloc__(self):
        self.destroy()

    def __repr__(self):
        status = "valid" if self._dmf != NULL else "destroyed"
        return f"DMFieldEvaluator({status})"
