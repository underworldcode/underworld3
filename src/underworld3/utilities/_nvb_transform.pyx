# cython: language_level=3
"""Registers the native NVB DMPlexTransform ("uwnvb") into PETSc on import.

Route B of docs/developer/design/NVB_GRADED_ADAPT.md. The C transform lives in
nvb_transform.c; this thin module just calls its registration entry once PETSc is
initialised (importing petsc4py.PETSc guarantees that). After import,
``PETSc.DMPlexTransform().setType("uwnvb")`` and the ``dm_plex_transform_type=uwnvb``
option resolve to the NVB transform.
"""
from petsc4py import PETSc as _PETSc   # ensures PetscInitialize has run
from petsc4py.PETSc cimport DM, PetscDM

cdef extern from "petsc.h":
    ctypedef int PetscInt

cdef extern int UWNVBTransformRegister()
cdef extern int UWNVBRefine(PetscDM dm, const char *wantName, PetscDM *rdm)
cdef extern int UWNVBPointQueueSelfTest(PetscInt capacity, PetscInt burst, PetscInt rounds, PetscInt *returned, PetscInt *mismatches)

cdef int _ierr = UWNVBTransformRegister()
if _ierr != 0:
    raise RuntimeError("UWNVBTransformRegister failed with PETSc error %d" % _ierr)

registered = True


def refine(dm, want_label="adapt"):
    """Graded NVB refinement of ``dm``.

    Every cell flagged in the ``want_label`` adaptation label (value
    ``DM_ADAPT_REFINE = 1``) is bisected once, with the bounded newest-vertex
    conforming closure, returning a fresh ``PETSc.DMPlex`` that carries an updated
    per-triangle ``uwnvb_refedge`` slot label. Refinement runs as repeated
    conforming single-bisection sub-passes (the ``uwnvb_bisect`` transform) so the
    output stays co-partitioned with the input — the custom-P FMG tail consumes it
    unchanged. Call repeatedly (re-marking) for multi-level grading.

    The whole refine (closure + drain + slot maintenance) runs in C.
    """
    cdef DM c_dm = dm
    cdef DM out = _PETSc.DMPlex()
    cdef bytes name = want_label.encode("utf8")
    cdef int ierr = UWNVBRefine(c_dm.dm, <const char *>name, &out.dm)
    if ierr != 0:
        raise RuntimeError("UWNVBRefine failed with PETSc error %d" % ierr)
    return out


def _point_queue_self_test(capacity, burst, rounds):
    """Exercise the transform's own point queue: starting at ``capacity`` entries,
    ``rounds`` times enqueue ``burst`` consecutive integers and dequeue half of what
    is queued, then drain. Returns ``(returned, mismatches)``: how many values came
    back, and how many of them were not the next value enqueued. See
    ``UWNVBPointQueueSelfTest`` in ``nvb_transform.c`` (#1140)."""
    cdef PetscInt returned = 0
    cdef PetscInt mismatches = 0
    cdef int ierr = UWNVBPointQueueSelfTest(capacity, burst, rounds, &returned, &mismatches)
    if ierr != 0:
        raise RuntimeError("UWNVBPointQueueSelfTest failed with PETSc error %d" % ierr)
    return int(returned), int(mismatches)
