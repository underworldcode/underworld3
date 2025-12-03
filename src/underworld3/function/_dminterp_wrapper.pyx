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
    PetscErrorCode DMInterpolationSetUp_UW(DMInterpolationInfo ipInfo, PetscDM dm, int petscbool, int petscbool, size_t* owning_cell)
    PetscErrorCode DMInterpolationEvaluate_UW(DMInterpolationInfo ipInfo, PetscDM dm, PetscVec x, PetscVec v)

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
    cdef public object cells   # numpy array - Python keeps alive
    cdef public int dofcount
    cdef public int dim
    cdef public bint is_valid
    cdef public double creation_time
    cdef public int use_count

    def __cinit__(self):
        """Initialize - structure not yet created."""
        self.is_valid = False
        self.use_count = 0
        import time
        self.creation_time = time.time()

    def create_structure(self, mesh, np.ndarray[double, ndim=2] coords,
                        np.ndarray[long, ndim=1] cells, int dofcount):
        """
        Create and set up the DMInterpolation structure.

        This does the expensive DMInterpolationCreate + SetUp operations.

        Parameters
        ----------
        mesh : Mesh
            The mesh object
        coords : ndarray (n_points, dim)
            Coordinates to interpolate at
        cells : ndarray (n_points,)
            Cell hints for each coordinate
        dofcount : int
            Total number of DOFs to interpolate
        """
        cdef PetscErrorCode ierr
        cdef int n_points = coords.shape[0]
        cdef int dim = coords.shape[1]

        # Store references (Python keeps these alive)
        self.coords = coords.copy()  # CRITICAL: keep alive!
        self.cells = cells.copy()
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

        # Add interpolation points - use contiguous array's data pointer
        cdef double[:, ::1] coords_view = np.ascontiguousarray(self.coords)
        ierr = DMInterpolationAddPoints(self._ipInfo, n_points, &coords_view[0, 0])
        if ierr != 0:
            DMInterpolationDestroy(&self._ipInfo)
            raise RuntimeError(f"DMInterpolationAddPoints failed with error {ierr}")

        # Set up with cell hints
        # Extract PETSc DM from mesh
        cdef DM dm_obj = mesh.dm
        cdef PetscDM dm = dm_obj.dm

        # Extract cell hints as size_t array
        cdef long[::1] cells_view = np.ascontiguousarray(self.cells)
        ierr = DMInterpolationSetUp_UW(self._ipInfo, dm, 0, 0, <size_t*> &cells_view[0])
        if ierr != 0:
            DMInterpolationDestroy(&self._ipInfo)
            raise RuntimeError(f"DMInterpolationSetUp_UW failed with error {ierr}")

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
