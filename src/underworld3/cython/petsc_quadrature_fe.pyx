# cython: language_level=3
r"""
Quadrature-point finite element (the "delta space").

A ``PetscFE`` whose basis functions are Kronecker deltas at the points of a
quadrature rule and whose dual space is point evaluation at those same
points. Tabulated on its own rule the basis is the identity matrix, so a
field of this type that is read by the assembler as an auxiliary field
(``a[]`` in the pointwise functions) delivers the stored value at each
quadrature point with no interpolation at all. The dofs all sit on the cell
interior, so the local vector is laid out cell-major, point-minor.

Use it for values that are *injected* at the integration points (a
semi-Lagrangian history, a per-point material property reconstructed from a
swarm). It cannot be *sampled* anywhere else: the derivative tabulation is
zero and evaluation at points off the rule returns zeros.

Built from a UW3-registered prime space ``uwdelta`` (``uw_delta_space.h``,
a PETSc plugin type: PETSc's own ``PETSCSPACEPOINT`` cannot be tabulated
anywhere but its own points, which breaks ``PetscFESetUp``, face tabulation
and boundary integrals) and a ``PETSCDUALSPACESIMPLE`` dual space, through
``PetscFECreateFromSpaces``. Works on any PETSc build. petsc4py cannot
construct the one-point delta functionals itself (``Quad`` has no
``setData``), which is why this helper is Cython.

Scalar (one-component) elements only.
"""

from petsc4py import PETSc
from petsc4py.PETSc cimport FE, PetscFE, Quad, PetscQuadrature, DM, PetscDM
from petsc4py.PETSc cimport PetscSpace, PetscDualSpace, PetscObject, MPI_Comm
from petsc4py.PETSc cimport CHKERR as CHKERRQ
from underworld3.cython.petsc_types cimport PetscInt, PetscReal, PetscErrorCode, PetscBool

import numpy as np


cdef extern from "petsc.h" nogil:
    MPI_Comm PETSC_COMM_SELF
    ctypedef int DMPolytopeType

    PetscErrorCode PetscSpaceCreate(MPI_Comm, PetscSpace*)
    PetscErrorCode PetscSpaceSetType(PetscSpace, const char*)
    PetscErrorCode PetscSpaceSetNumVariables(PetscSpace, PetscInt)
    PetscErrorCode PetscSpaceSetNumComponents(PetscSpace, PetscInt)
    PetscErrorCode PetscSpaceSetUp(PetscSpace)

    PetscErrorCode PetscDualSpaceCreate(MPI_Comm, PetscDualSpace*)
    PetscErrorCode PetscDualSpaceSetType(PetscDualSpace, const char*)
    PetscErrorCode PetscDualSpaceSetDM(PetscDualSpace, PetscDM)
    PetscErrorCode PetscDualSpaceSetNumComponents(PetscDualSpace, PetscInt)
    PetscErrorCode PetscDualSpaceSimpleSetDimension(PetscDualSpace, PetscInt)
    PetscErrorCode PetscDualSpaceSimpleSetFunctional(PetscDualSpace, PetscInt, PetscQuadrature)
    PetscErrorCode PetscDualSpaceSetUp(PetscDualSpace)

    PetscErrorCode DMPlexCreateReferenceCell(MPI_Comm, DMPolytopeType, PetscDM*)
    PetscErrorCode DMDestroy(PetscDM*)

    PetscErrorCode PetscQuadratureCreate(MPI_Comm, PetscQuadrature*)
    PetscErrorCode PetscQuadratureSetData(PetscQuadrature, PetscInt, PetscInt, PetscInt, const PetscReal*, const PetscReal*)
    PetscErrorCode PetscQuadratureGetData(PetscQuadrature, PetscInt*, PetscInt*, PetscInt*, const PetscReal**, const PetscReal**)
    PetscErrorCode PetscQuadratureDestroy(PetscQuadrature*)

    PetscErrorCode PetscFECreateFromSpaces(PetscSpace, PetscDualSpace, PetscQuadrature, PetscQuadrature, PetscFE*)
    PetscErrorCode PetscFECreateVector(PetscFE, PetscInt, PetscBool, PetscBool, PetscFE*)
    PetscErrorCode PetscFEDestroy(PetscFE*)
    PetscErrorCode PetscObjectReference(PetscObject)
    PetscErrorCode PetscObjectSetName(PetscObject, const char*)
    PetscErrorCode PetscMalloc(size_t, void**)

    ctypedef struct _n_PetscTabulation:
        PetscInt K
        PetscInt Nr
        PetscInt Np
        PetscInt Nb
        PetscInt Nc
        PetscInt cdim
        PetscReal **T
    ctypedef _n_PetscTabulation* PetscTabulation
    PetscErrorCode PetscFECreateTabulation(PetscFE, PetscInt, PetscInt, const PetscReal*, PetscInt, PetscTabulation*)
    PetscErrorCode PetscTabulationDestroy(PetscTabulation*)
    PetscErrorCode DMPlexComputeCellGeometryFEM(PetscDM, PetscInt, PetscQuadrature, PetscReal*, PetscReal*, PetscReal*, PetscReal*)
    PetscErrorCode DMPlexGetHeightStratum(PetscDM, PetscInt, PetscInt*, PetscInt*)
    PetscErrorCode DMGetCoordinateDim(PetscDM, PetscInt*)

cdef extern from "uw_delta_space.h" nogil:
    PetscErrorCode UWDeltaSpaceRegister()
    PetscErrorCode UWDeltaSpaceSetPoints(PetscSpace, PetscQuadrature)


# Register the plugin space type once, at import.
CHKERRQ(UWDeltaSpaceRegister())


def create_delta_fe(Quad quad, int polytope, name="quadrature_point_fe", int num_components=1):
    r"""Build the quadrature-point element on ``quad``.

    ``num_components > 1`` wraps the scalar element with ``PetscFECreateVector``
    (interleaved basis and components): the dofs of a cell are point-major,
    component-minor, so a local vector reshapes to ``(ncells * Nq, Nc)``.

    Parameters
    ----------
    quad : petsc4py.PETSc.Quad
        The cell rule the element's points coincide with. Take it from an
        existing field, ``fe.getQuadrature()``, so it is the mesh's rule.
    polytope : int
        The reference cell type (``dm.getCellType(cStart)``) for the dual
        space's reference cell.
    name : str
        PETSc object name.

    Returns
    -------
    petsc4py.PETSc.FE
        Element of ``Nq * num_components`` basis functions (``Nq`` points in
        the rule, ``num_components`` interleaved components, one by default),
        with ``quad`` as its cell quadrature and no face quadrature.
    """
    cdef PetscInt qdim = 0, qNc = 0, Nq = 0, i, d
    cdef const PetscReal *points = NULL
    cdef const PetscReal *weights = NULL
    cdef PetscReal *fpts = NULL
    cdef PetscReal *fwts = NULL
    cdef PetscQuadrature functional = NULL
    cdef PetscSpace P = NULL
    cdef PetscDualSpace Q = NULL
    cdef PetscDM refcell = NULL
    cdef PetscFE cfe = NULL
    cdef PetscFE vfe = NULL
    cdef FE pyfe
    if num_components < 1:
        raise ValueError("num_components must be >= 1")

    CHKERRQ(PetscQuadratureGetData(quad.quad, &qdim, &qNc, &Nq, &points, &weights))
    if qNc != 1:
        raise ValueError("create_delta_fe: the rule must have one component")

    # Prime space: deltas at the rule's points.
    CHKERRQ(PetscSpaceCreate(PETSC_COMM_SELF, &P))
    CHKERRQ(PetscSpaceSetType(P, b"uwdelta"))
    CHKERRQ(PetscSpaceSetNumVariables(P, qdim))
    CHKERRQ(PetscSpaceSetNumComponents(P, 1))
    CHKERRQ(UWDeltaSpaceSetPoints(P, quad.quad))
    CHKERRQ(PetscSpaceSetUp(P))

    # Dual space: one point-evaluation functional per rule point, all on the
    # cell interior of the reference cell.
    CHKERRQ(DMPlexCreateReferenceCell(PETSC_COMM_SELF, <DMPolytopeType>polytope, &refcell))
    CHKERRQ(PetscDualSpaceCreate(PETSC_COMM_SELF, &Q))
    CHKERRQ(PetscDualSpaceSetType(Q, b"simple"))
    CHKERRQ(PetscDualSpaceSetDM(Q, refcell))
    CHKERRQ(PetscDualSpaceSetNumComponents(Q, 1))
    CHKERRQ(PetscDualSpaceSimpleSetDimension(Q, Nq))
    for i in range(Nq):
        # PetscQuadratureSetData takes ownership: arrays must be PetscMalloc'd.
        CHKERRQ(PetscMalloc(sizeof(PetscReal) * qdim, <void**>&fpts))
        CHKERRQ(PetscMalloc(sizeof(PetscReal), <void**>&fwts))
        for d in range(qdim):
            fpts[d] = points[i * qdim + d]
        fwts[0] = 1.0
        CHKERRQ(PetscQuadratureCreate(PETSC_COMM_SELF, &functional))
        CHKERRQ(PetscQuadratureSetData(functional, qdim, 1, 1, fpts, fwts))
        # SimpleSetFunctional duplicates; release ours.
        CHKERRQ(PetscDualSpaceSimpleSetFunctional(Q, i, functional))
        CHKERRQ(PetscQuadratureDestroy(&functional))
    CHKERRQ(PetscDualSpaceSetUp(Q))
    CHKERRQ(DMDestroy(&refcell))

    # PetscFECreateFromSpaces consumes P, Q and the quadrature: keep the
    # caller's Quad alive by taking a reference first. No face quadrature.
    CHKERRQ(PetscObjectReference(<PetscObject>quad.quad))
    CHKERRQ(PetscFECreateFromSpaces(P, Q, quad.quad, NULL, &cfe))
    if num_components > 1:
        CHKERRQ(PetscFECreateVector(cfe, num_components, <PetscBool>1, <PetscBool>1, &vfe))
        CHKERRQ(PetscFEDestroy(&cfe))      # the vector element holds its own reference
        cfe = vfe
    CHKERRQ(PetscObjectSetName(<PetscObject>cfe, name.encode()))

    pyfe = FE()
    pyfe.fe = cfe
    return pyfe


def tabulate(FE fe, points, int K=0):
    r"""Tabulate ``fe``'s basis at reference-cell ``points``.

    Returns the value tabulation as an array shaped ``(Np, Nb, Nc)``.
    Exposed for tests: on its own rule the delta element returns the
    identity.
    """
    cdef PetscTabulation T = NULL
    cdef PetscInt Np, Nb, Nc, p, b, c
    pts = np.ascontiguousarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("points must be (Np, dim)")
    cdef double[:, ::1] pv = pts
    Np = pts.shape[0]
    CHKERRQ(PetscFECreateTabulation(fe.fe, 1, Np, &pv[0, 0], K, &T))
    Nb = T.Nb
    Nc = T.Nc
    out = np.empty((Np, Nb, Nc), dtype=np.float64)
    cdef double[:, :, ::1] ov = out
    for p in range(Np):
        for b in range(Nb):
            for c in range(Nc):
                ov[p, b, c] = T.T[0][(p * Nb + b) * Nc + c]
    CHKERRQ(PetscTabulationDestroy(&T))
    return out


def cell_quadrature_points(DM dm, Quad quad):
    r"""Physical coordinates of the rule's points in every local cell.

    Returns an array shaped ``(ncells, Nq, cdim)`` in local cell order, computed
    by ``DMPlexComputeCellGeometryFEM`` - the same map the assembler uses for
    its integration points, so row ``(c, q)`` is exactly where the pointwise
    functions see quadrature point ``q`` of cell ``c``. No locator involved.
    """
    cdef PetscInt cStart = 0, cEnd = 0, cdim = 0, Nq = 0, c, q, d
    cdef PetscReal *v = NULL
    cdef PetscReal *J = NULL
    cdef PetscReal *invJ = NULL
    cdef PetscReal *detJ = NULL
    CHKERRQ(DMPlexGetHeightStratum(dm.dm, 0, &cStart, &cEnd))
    CHKERRQ(DMGetCoordinateDim(dm.dm, &cdim))
    CHKERRQ(PetscQuadratureGetData(quad.quad, NULL, NULL, &Nq, NULL, NULL))
    ncells = cEnd - cStart
    out = np.empty((ncells, Nq, cdim), dtype=np.float64)
    cdef double[:, :, ::1] ov = out
    vbuf = np.empty(Nq * cdim, dtype=np.float64)
    Jbuf = np.empty(Nq * cdim * cdim, dtype=np.float64)
    iJbuf = np.empty(Nq * cdim * cdim, dtype=np.float64)
    dJbuf = np.empty(Nq, dtype=np.float64)
    cdef double[::1] vv = vbuf
    cdef double[::1] Jv = Jbuf
    cdef double[::1] iJv = iJbuf
    cdef double[::1] dJv = dJbuf
    if ncells == 0:
        return out
    v = &vv[0]; J = &Jv[0]; invJ = &iJv[0]; detJ = &dJv[0]
    for c in range(cStart, cEnd):
        CHKERRQ(DMPlexComputeCellGeometryFEM(dm.dm, c, quad.quad, v, J, invJ, detJ))
        for q in range(Nq):
            for d in range(cdim):
                ov[c - cStart, q, d] = v[q * cdim + d]
    return out


def tabulate_with_derivatives(FE fe, points):
    r"""Tabulate ``fe``'s basis and its reference gradient at ``points``.

    Returns ``(B, D)`` shaped ``(Np, Nb, Nc)`` and ``(Np, Nb, Nc, dim)``;
    ``D`` is the gradient with respect to the reference coordinates, so a
    physical gradient is ``invJ^T D``.
    """
    cdef PetscTabulation T = NULL
    cdef PetscInt Np, Nb, Nc, cdim, p, b, c, d
    pts = np.ascontiguousarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("points must be (Np, dim)")
    cdef double[:, ::1] pv = pts
    Np = pts.shape[0]
    CHKERRQ(PetscFECreateTabulation(fe.fe, 1, Np, &pv[0, 0], 1, &T))
    Nb = T.Nb
    Nc = T.Nc
    cdim = T.cdim
    B = np.empty((Np, Nb, Nc), dtype=np.float64)
    D = np.empty((Np, Nb, Nc, cdim), dtype=np.float64)
    cdef double[:, :, ::1] bv = B
    cdef double[:, :, :, ::1] dv = D
    for p in range(Np):
        for b in range(Nb):
            for c in range(Nc):
                bv[p, b, c] = T.T[0][(p * Nb + b) * Nc + c]
                for d in range(cdim):
                    dv[p, b, c, d] = T.T[1][((p * Nb + b) * Nc + c) * cdim + d]
    CHKERRQ(PetscTabulationDestroy(&T))
    return B, D


def cell_affine_maps(DM dm):
    r"""Affine reference map of every local cell.

    Returns ``(v0, invJ, detJ)`` shaped ``(ncells, cdim)``, ``(ncells, cdim,
    cdim)`` and ``(ncells,)`` in local cell order, from
    ``DMPlexComputeCellGeometryFEM`` with no rule (the affine map). The
    reference coordinate of a physical point ``x`` in cell ``c`` is
    ``invJ[c] @ (x - v0[c]) - 1`` in PETSc's ``[-1, 1]`` reference frame
    (``v0`` is the image of the reference corner ``(-1, ..., -1)``), which is
    the frame :func:`tabulate` expects.
    """
    cdef PetscInt cStart = 0, cEnd = 0, cdim = 0, c, d, e
    cdef PetscReal *v = NULL
    cdef PetscReal *J = NULL
    cdef PetscReal *invJ = NULL
    cdef PetscReal detJ = 0.0
    CHKERRQ(DMPlexGetHeightStratum(dm.dm, 0, &cStart, &cEnd))
    CHKERRQ(DMGetCoordinateDim(dm.dm, &cdim))
    ncells = cEnd - cStart
    v0 = np.empty((ncells, cdim), dtype=np.float64)
    iJ = np.empty((ncells, cdim, cdim), dtype=np.float64)
    dJ = np.empty((ncells,), dtype=np.float64)
    cdef double[:, ::1] v0v = v0
    cdef double[:, :, ::1] iJv = iJ
    cdef double[::1] dJv = dJ
    vbuf = np.empty(cdim, dtype=np.float64)
    Jbuf = np.empty(cdim * cdim, dtype=np.float64)
    iJbuf = np.empty(cdim * cdim, dtype=np.float64)
    cdef double[::1] vv = vbuf
    cdef double[::1] Jv = Jbuf
    cdef double[::1] iJvb = iJbuf
    if ncells == 0:
        return v0, iJ, dJ
    v = &vv[0]; J = &Jv[0]; invJ = &iJvb[0]
    for c in range(cStart, cEnd):
        CHKERRQ(DMPlexComputeCellGeometryFEM(dm.dm, c, NULL, v, J, invJ, &detJ))
        dJv[c - cStart] = detJ
        for d in range(cdim):
            v0v[c - cStart, d] = v[d]
            for e in range(cdim):
                iJv[c - cStart, d, e] = invJ[d * cdim + e]
    return v0, iJ, dJ
