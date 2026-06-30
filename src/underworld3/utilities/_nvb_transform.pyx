# cython: language_level=3
"""Registers the native NVB DMPlexTransform ("uwnvb") into PETSc on import.

Route B of docs/developer/design/NVB_GRADED_ADAPT.md. The C transform lives in
nvb_transform.c; this thin module just calls its registration entry once PETSc is
initialised (importing petsc4py.PETSc guarantees that). After import,
``PETSc.DMPlexTransform().setType("uwnvb")`` and the ``dm_plex_transform_type=uwnvb``
option resolve to the NVB transform.
"""
from petsc4py import PETSc as _PETSc   # ensures PetscInitialize has run

cdef extern int UWNVBTransformRegister()

cdef int _ierr = UWNVBTransformRegister()
if _ierr != 0:
    raise RuntimeError("UWNVBTransformRegister failed with PETSc error %d" % _ierr)

registered = True
