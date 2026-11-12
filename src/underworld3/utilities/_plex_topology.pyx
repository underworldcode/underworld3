# cython: language_level=3, boundscheck=False, wraparound=False
"""Bulk DMPlex topology tables, built in C.

The fault and add-on machinery asks the same questions of every cell, face or
vertex of a mesh — which vertices close this cell, which cells support this
face — and asked them one point at a time through petsc4py, a Python call per
point. These functions answer them for a whole stratum in one C loop and return
numpy tables, so the logic above them can be written as array operations.

All of them are rank-local: they read this rank's chart and call no collective.
"""
import numpy as np
cimport numpy as cnp

from petsc4py.PETSc cimport DM, PetscDM, Section, PetscSection

cnp.import_array()

cdef extern from "petsc.h":
    ctypedef int PetscInt
    ctypedef int PetscErrorCode
    ctypedef enum PetscBool:
        PETSC_FALSE
        PETSC_TRUE
    PetscErrorCode DMPlexGetChart(PetscDM, PetscInt *, PetscInt *)
    PetscErrorCode DMPlexGetDepthStratum(PetscDM, PetscInt, PetscInt *, PetscInt *)
    PetscErrorCode DMPlexGetTransitiveClosure(PetscDM, PetscInt, PetscBool,
                                              PetscInt *, PetscInt **)
    PetscErrorCode DMPlexRestoreTransitiveClosure(PetscDM, PetscInt, PetscBool,
                                                  PetscInt *, PetscInt **)
    PetscErrorCode DMPlexGetSupportSize(PetscDM, PetscInt, PetscInt *)
    PetscErrorCode DMPlexGetSupport(PetscDM, PetscInt, const PetscInt **)
    PetscErrorCode PetscSectionSetChart(PetscSection, PetscInt, PetscInt)
    PetscErrorCode PetscSectionSetDof(PetscSection, PetscInt, PetscInt)
    PetscErrorCode PetscSectionSetUp(PetscSection)
    PetscErrorCode PetscSectionGetOffset(PetscSection, PetscInt, PetscInt *)


cdef inline void _check(PetscErrorCode ierr, str what) except *:
    if ierr != 0:
        raise RuntimeError(f"PETSc error {ierr} in {what}")


def closure_points(DM dm, points, int depth):
    """The points of depth ``depth`` in the transitive closure of each of
    ``points``, in closure order: an ``(n, k)`` int64 table, ``k`` the largest
    count found, short rows padded with -1. ``depth`` 0 gives a cell's or a
    face's vertices, 1 a cell's edges."""
    cdef cnp.int64_t[::1] pts = np.ascontiguousarray(points, dtype=np.int64)
    cdef Py_ssize_t n = pts.shape[0]
    cdef PetscInt dS, dE, size = 0, i, p
    cdef PetscInt *closure = NULL
    cdef Py_ssize_t r, j, cnt, kmax = 0
    _check(DMPlexGetDepthStratum(dm.dm, depth, &dS, &dE), "DMPlexGetDepthStratum")
    # first pass: the row width
    for r in range(n):
        p = <PetscInt>pts[r]
        _check(DMPlexGetTransitiveClosure(dm.dm, p, PETSC_TRUE, &size, &closure),
               "DMPlexGetTransitiveClosure")
        cnt = 0
        for i in range(size):
            if dS <= closure[2 * i] < dE:
                cnt += 1
        _check(DMPlexRestoreTransitiveClosure(dm.dm, p, PETSC_TRUE, &size, &closure),
               "DMPlexRestoreTransitiveClosure")
        if cnt > kmax:
            kmax = cnt
    out = np.full((n, kmax), -1, dtype=np.int64)
    cdef cnp.int64_t[:, ::1] o = out
    for r in range(n):
        p = <PetscInt>pts[r]
        _check(DMPlexGetTransitiveClosure(dm.dm, p, PETSC_TRUE, &size, &closure),
               "DMPlexGetTransitiveClosure")
        j = 0
        for i in range(size):
            if dS <= closure[2 * i] < dE:
                o[r, j] = closure[2 * i]
                j += 1
        _check(DMPlexRestoreTransitiveClosure(dm.dm, p, PETSC_TRUE, &size, &closure),
               "DMPlexRestoreTransitiveClosure")
    return out


def supports(DM dm, points):
    """The support of each of ``points``: an ``(n, k)`` int64 table, ``k`` the
    largest support found (two for the faces of a manifold mesh), short rows
    padded with -1."""
    cdef cnp.int64_t[::1] pts = np.ascontiguousarray(points, dtype=np.int64)
    cdef Py_ssize_t n = pts.shape[0]
    cdef PetscInt size, i, p
    cdef const PetscInt *sup = NULL
    cdef Py_ssize_t r, kmax = 0
    for r in range(n):
        _check(DMPlexGetSupportSize(dm.dm, <PetscInt>pts[r], &size),
               "DMPlexGetSupportSize")
        if size > kmax:
            kmax = size
    out = np.full((n, kmax), -1, dtype=np.int64)
    cdef cnp.int64_t[:, ::1] o = out
    for r in range(n):
        p = <PetscInt>pts[r]
        _check(DMPlexGetSupportSize(dm.dm, p, &size), "DMPlexGetSupportSize")
        _check(DMPlexGetSupport(dm.dm, p, &sup), "DMPlexGetSupport")
        for i in range(size):
            o[r, i] = sup[i]
    return out


def fill_section(Section section, DM dm, points, int ndof):
    """Set ``section`` to the chart of ``dm`` with ``ndof`` dofs on each of
    ``points`` and none elsewhere, and set it up."""
    cdef cnp.int64_t[::1] pts = np.ascontiguousarray(points, dtype=np.int64)
    cdef PetscInt pStart, pEnd
    cdef Py_ssize_t r
    _check(DMPlexGetChart(dm.dm, &pStart, &pEnd), "DMPlexGetChart")
    _check(PetscSectionSetChart(section.sec, pStart, pEnd), "PetscSectionSetChart")
    for r in range(pts.shape[0]):
        _check(PetscSectionSetDof(section.sec, <PetscInt>pts[r], ndof),
               "PetscSectionSetDof")
    _check(PetscSectionSetUp(section.sec), "PetscSectionSetUp")


def section_offsets(Section section, points):
    """The offset of each of ``points`` in ``section``: an int64 array."""
    cdef cnp.int64_t[::1] pts = np.ascontiguousarray(points, dtype=np.int64)
    out = np.empty(pts.shape[0], dtype=np.int64)
    cdef cnp.int64_t[::1] o = out
    cdef PetscInt off
    cdef Py_ssize_t r
    for r in range(pts.shape[0]):
        _check(PetscSectionGetOffset(section.sec, <PetscInt>pts[r], &off),
               "PetscSectionGetOffset")
        o[r] = off
    return out


cdef extern from "petsc.h":
    PetscErrorCode PetscSectionGetDof(PetscSection, PetscInt, PetscInt *)


def closure_dof_rows(DM dm, Section section, points, int block):
    """The rows of each point's closure dofs in a local vector laid out by
    ``section``, in closure order, counted in blocks of ``block`` (the
    coordinate dimension for a coordinate field, so a row is one node): an
    ``(n, k)`` int64 table, ``k`` the largest count, short rows padded with -1.
    A point with ``d`` dofs at offset ``o`` contributes rows ``o / block`` to
    ``(o + d) / block - 1``."""
    cdef cnp.int64_t[::1] pts = np.ascontiguousarray(points, dtype=np.int64)
    cdef Py_ssize_t n = pts.shape[0]
    cdef PetscInt size = 0, i, p, q, dof, off
    cdef PetscInt *closure = NULL
    cdef Py_ssize_t r, j, cnt, kmax = 0, m
    for r in range(n):
        p = <PetscInt>pts[r]
        _check(DMPlexGetTransitiveClosure(dm.dm, p, PETSC_TRUE, &size, &closure),
               "DMPlexGetTransitiveClosure")
        cnt = 0
        for i in range(size):
            _check(PetscSectionGetDof(section.sec, closure[2 * i], &dof),
                   "PetscSectionGetDof")
            cnt += dof // block
        _check(DMPlexRestoreTransitiveClosure(dm.dm, p, PETSC_TRUE, &size, &closure),
               "DMPlexRestoreTransitiveClosure")
        if cnt > kmax:
            kmax = cnt
    out = np.full((n, kmax), -1, dtype=np.int64)
    cdef cnp.int64_t[:, ::1] o = out
    for r in range(n):
        p = <PetscInt>pts[r]
        _check(DMPlexGetTransitiveClosure(dm.dm, p, PETSC_TRUE, &size, &closure),
               "DMPlexGetTransitiveClosure")
        j = 0
        for i in range(size):
            q = closure[2 * i]
            _check(PetscSectionGetDof(section.sec, q, &dof), "PetscSectionGetDof")
            if dof:
                _check(PetscSectionGetOffset(section.sec, q, &off),
                       "PetscSectionGetOffset")
                for m in range(dof // block):
                    o[r, j] = off // block + m
                    j += 1
        _check(DMPlexRestoreTransitiveClosure(dm.dm, p, PETSC_TRUE, &size, &closure),
               "DMPlexRestoreTransitiveClosure")
    return out
