# cython: language_level=3
"""
Safe Cython wrapper for DMInterpolationInfo C structures.

This extension class manages the lifetime of DMInterpolationInfo structures,
ensuring proper cleanup via Python's reference counting.
"""

from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np

# Import PETSc Cython types
from petsc4py.PETSc cimport DM, PetscDM
from petsc4py.PETSc cimport Vec, PetscVec

# Import PETSc C types and functions
cdef extern from "petsc.h" nogil:
    ctypedef struct DMInterpolationInfo:
        pass

    ctypedef int PetscErrorCode
    ctypedef int PetscInt
    ctypedef double PetscReal
    ctypedef void* MPI_Comm

    MPI_Comm MPI_COMM_SELF

    PetscErrorCode DMInterpolationCreate(MPI_Comm comm, DMInterpolationInfo *ipInfo)
    PetscErrorCode DMInterpolationSetDim(DMInterpolationInfo ipInfo, PetscInt dim)
    PetscErrorCode DMInterpolationSetDof(DMInterpolationInfo ipInfo, PetscInt dof)
    PetscErrorCode DMInterpolationAddPoints(DMInterpolationInfo ipInfo, PetscInt n, PetscReal points[])
    PetscErrorCode DMInterpolationDestroy(DMInterpolationInfo *ipInfo)

# Import custom UW routines
cdef extern from "petsc_tools.h" nogil:
    PetscErrorCode DMInterpolationSetUp_UW(DMInterpolationInfo ipInfo, PetscDM dm, int petscbool, int petscbool, size_t* owning_cell, int petscbool)
    PetscErrorCode DMInterpolationEvaluate_UW(DMInterpolationInfo ipInfo, PetscDM dm, PetscVec x, PetscVec v)
    PetscErrorCode DMInterpolationGetUnlocated_UW(DMInterpolationInfo ipInfo, PetscInt n, signed char* mask)

cdef class CachedDMInterpolationInfo:
    """
    Python-managed wrapper for DMInterpolationInfo C structure.

    This class ensures:
    1. Proper cleanup via __dealloc__ (Python GC calls this)
    2. Keeps referenced arrays alive (coords, cells)
    3. Provides safe access to the C structure

    Attributes
    ----------
    _ipInfo : DMInterpolationInfo
        The actual C structure (opaque pointer)
    coords : ndarray
        Coordinates used to build this structure (kept alive)
    cells : ndarray
        Cell hints used for setup (kept alive)
    dofcount : int
        Total DOF count this structure was built for
    is_valid : bool
        Whether the structure is valid (not yet destroyed)
    """

    cdef DMInterpolationInfo _ipInfo
    cdef public object coords  # numpy array - Python keeps alive
    cdef public object cells   # numpy array - Python keeps alive (may be None)
    cdef public object unlocated_mask  # bool ndarray: points with no owning cell
    cdef public int dofcount
    cdef public int dim
    cdef public bint is_valid
    cdef public double creation_time
    cdef public int use_count

    def __cinit__(self):
        """Initialize - structure not yet created."""
        self.is_valid = False
        self.use_count = 0
        self.unlocated_mask = None
        import time
        self.creation_time = time.time()

    def create_structure(self, mesh, np.ndarray[double, ndim=2] coords,
                        object cells, int dofcount, bint hint_authoritative=False):
        """
        Create and set up the DMInterpolation structure.

        This does the expensive DMInterpolationCreate + SetUp operations.

        Parameters
        ----------
        mesh : Mesh
            The mesh object
        coords : ndarray (n_points, dim)
            Coordinates to interpolate at
        cells : ndarray (n_points,) or None
            Cell hints for each coordinate. With ``hint_authoritative=True``
            the hint bypasses ``DMLocatePoints`` entirely; with ``None`` (or
            ``hint_authoritative=False``) PETSc's search is authoritative and
            unlocated points are reported in :attr:`unlocated_mask` for the
            caller's RBF fallback.
        dofcount : int
            Total number of DOFs to interpolate
        hint_authoritative : bool
            Whether the supplied cells carry geometric authority — decided by
            the mesh's measured capability
            (``mesh._hint_is_authoritative(...)``), the single policy point.
        """
        cdef PetscErrorCode ierr
        cdef int n_points = coords.shape[0]
        cdef int dim = coords.shape[1]

        # Store references (Python keeps these alive)
        self.coords = coords.copy()  # CRITICAL: keep alive!
        self.cells = None if cells is None else np.asarray(cells).copy()
        self.dofcount = dofcount
        self.dim = dim

        # Create DMInterpolation structure
        ierr = DMInterpolationCreate(MPI_COMM_SELF, &self._ipInfo)
        if ierr != 0:
            raise RuntimeError(f"DMInterpolationCreate failed with error {ierr}")

        # Set dimension
        ierr = DMInterpolationSetDim(self._ipInfo, dim)
        if ierr != 0:
            DMInterpolationDestroy(&self._ipInfo)
            raise RuntimeError(f"DMInterpolationSetDim failed with error {ierr}")

        # Set DOF count
        ierr = DMInterpolationSetDof(self._ipInfo, dofcount)
        if ierr != 0:
            DMInterpolationDestroy(&self._ipInfo)
            raise RuntimeError(f"DMInterpolationSetDof failed with error {ierr}")

        # Declare typed memoryviews at function scope (Cython requirement)
        cdef double[:, ::1] coords_view
        cdef long[::1] cells_view
        cdef DM dm_obj = mesh.dm
        cdef PetscDM dm = dm_obj.dm

        # Add interpolation points (guard against empty arrays)
        if n_points > 0:
            coords_view = np.ascontiguousarray(self.coords)
            ierr = DMInterpolationAddPoints(self._ipInfo, n_points, &coords_view[0, 0])
            if ierr != 0:
                DMInterpolationDestroy(&self._ipInfo)
                raise RuntimeError(f"DMInterpolationAddPoints failed with error {ierr}")

        # Set up — calls DMLocatePoints (or, when the hint is passed, the
        # DMLocatePoints-bypass path in petsc_tools.c) which is COLLECTIVE on
        # the mesh DM. All ranks must call this, even with zero local points.
        # ignoreOutsideDomain=1: PETSc silently skips points it cannot locate
        # rather than crashing.
        #
        # Hint policy — decided by the CALLER from the mesh's measured
        # location capability (mesh._hint_is_authoritative, the single policy
        # point) and passed in as hint_authoritative:
        #
        # AUTHORITATIVE (bypass): the cell-wall estimator carries geometric
        # authority — simplex/manifold meshes, and quad/hex meshes whose
        # measured face planarity + convexity qualify ("exact", or
        # "continuous" capability with all-continuous fields). petsc_tools.c
        # skips DMLocatePoints entirely and evaluates in the hinted cell
        # (with reference-coord clamping for on-face queries).
        #
        # NOT AUTHORITATIVE: PETSc DMLocatePoints runs. Points it drops (e.g.
        # queries exactly on the domain's closed upper faces — the ψ*
        # corruption behind the VEP blow-up, #390) get NaN in the interpolant
        # and are reported in unlocated_mask; the caller fills them via the
        # RBF fallback. A supplied non-authoritative hint would prefill the
        # C recovery map instead, but the capability policy passes cells=None
        # here precisely because that hint is not trustworthy on such meshes.
        cdef np.ndarray mask_arr = np.zeros(n_points, dtype=np.int8)
        cdef signed char[::1] mask_view = mask_arr
        if n_points > 0 and self.cells is not None:
            cells_view = np.ascontiguousarray(self.cells)
            ierr = DMInterpolationSetUp_UW(self._ipInfo, dm, 0, 1,
                                           <size_t*> &cells_view[0],
                                           1 if hint_authoritative else 0)
        else:
            # No hint array to pass (no points, or no cells) — but the POLICY
            # still has to be forwarded. Hardcoding 0 here made a rank with zero
            # local points disagree with its peers about whether the hint is
            # authoritative, and petsc_tools.c takes the DMLocatePoints branch
            # when it is not. DMLocatePoints is COLLECTIVE on the mesh DM, so
            # that rank blocked inside DMGetBoundingBox -> MPI_Allreduce while
            # the others bypassed and ran on to DMSwarmMigrate -> MPI_Comm_dup:
            # a deadlock whenever a query set leaves some rank empty (#611).
            # The policy is a mesh capability and already agrees across ranks;
            # it just has to survive the trip.
            ierr = DMInterpolationSetUp_UW(self._ipInfo, dm, 0, 1, NULL,
                                           1 if hint_authoritative else 0)
        if ierr != 0:
            DMInterpolationDestroy(&self._ipInfo)
            raise RuntimeError(f"DMInterpolationSetUp_UW failed with error {ierr}")

        # Which points ended up with no owning cell (dropped by
        # DMLocatePoints, or hinted -1 by the robust locator)? Their
        # interpolant slots will be NaN; the caller fills them via RBF.
        if n_points > 0:
            ierr = DMInterpolationGetUnlocated_UW(self._ipInfo, n_points, &mask_view[0])
            if ierr != 0:
                DMInterpolationDestroy(&self._ipInfo)
                raise RuntimeError(f"DMInterpolationGetUnlocated_UW failed with error {ierr}")
        self.unlocated_mask = mask_arr.astype(bool)

        self.is_valid = True

    def evaluate(self, mesh, np.ndarray[double, ndim=2] outarray):
        """
        Evaluate interpolation using this cached structure.

        This is the FAST operation - just evaluation, no setup!

        Parameters
        ----------
        mesh : Mesh
            The mesh object (must call update_lvec() first!)
        outarray : ndarray (n_points, dofcount)
            Output array to fill with interpolated values

        Returns
        -------
        outarray : ndarray
            Same array, now filled with values
        """
        if not self.is_valid:
            raise RuntimeError("Cannot evaluate with invalid DMInterpolationInfo")

        cdef PetscErrorCode ierr

        # Get mesh DM and lvec as PETSc objects
        cdef DM dm_obj = mesh.dm
        cdef Vec lvec_obj = mesh.lvec
        cdef PetscDM dm = dm_obj.dm
        cdef PetscVec pyfieldvec = lvec_obj.vec

        # Create PETSc vector wrapping the output array
        from petsc4py import PETSc as PyPETSc
        outvec_py = PyPETSc.Vec().createWithArray(outarray.ravel(), comm=PyPETSc.COMM_SELF)
        cdef Vec outvec_obj = outvec_py
        cdef PetscVec outvec = outvec_obj.vec

        # EVALUATE (this is the fast part!)
        ierr = DMInterpolationEvaluate_UW(self._ipInfo, dm, pyfieldvec, outvec)

        # Clean up temporary vector
        outvec_py.destroy()

        if ierr != 0:
            raise RuntimeError(f"DMInterpolationEvaluate_UW failed with error {ierr}")

        self.use_count += 1
        return outarray

    def __dealloc__(self):
        """
        Cleanup when Python GC destroys this object.

        This is called automatically when the cache entry is deleted
        or when the cache is cleared.
        """
        if self.is_valid:
            # Destroy the C structure
            DMInterpolationDestroy(&self._ipInfo)
            self.is_valid = False

    def __repr__(self):
        status = "valid" if self.is_valid else "destroyed"
        return (f"CachedDMInterpolationInfo({status}, "
                f"{self.coords.shape[0]} points, "
                f"{self.dofcount} DOFs, "
                f"used {self.use_count} times)")
