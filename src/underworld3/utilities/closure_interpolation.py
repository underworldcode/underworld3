"""Interpolation between two fields on one plex, applied cell by cell (#902).

Two DMs that share a plex (a coordinate DM and a basis DM built from it, a
variable's sub-DM and a scratch clone of the mesh) are interpolated by PETSc's
``DMPlexComputeInterpolatorNested``. Its element matrix comes from the reference
element alone, so it is the same for every cell, and the parallel matrix it
assembles is only that element matrix scattered over the cells' closures. Its
preallocation pass is skipped on a rank with no local rows and no local columns,
while its fill pass still runs over that rank's cells. A shared point belongs to
the highest rank holding it, so a rank whose cells lie wholly inside higher
ranks' regions owns nothing: its entries reach owners that never preallocated
them, the owner raises inside ``MatAssemblyEnd``, and the other ranks wait in
that assembly's collective for ever. Measured on kaiju at np 32 and 52 on a
124-cell coarse level.

Here PETSc builds the element matrix on a single reference cell, on one rank,
and it is applied to each local cell's closure values directly. The values at a
node are those of its owner (the result is sent to the owners and back), as
they were when the matrix was multiplied.
"""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from underworld3.utilities._plex_topology import closure_dof_rows

_REFERENCE_CELLS = {
    # (dim, simplex): vertices of one cell, any valid order (the element
    # matrix depends on the cell's topology, not on its geometry)
    (2, True): [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    (3, True): [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    (2, False): [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    (3, False): [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
}


def _reference_element_matrix(fe_src, fe_dst, dim, cdim, simplex, block):
    """PETSc's element interpolation matrix from ``fe_src`` to ``fe_dst``, in
    closure order of nodes (one row per destination node, one column per source
    node, the same for every component), or ``None`` when a point other than the
    cell carries more than one node: those nodes are ordered by the point's
    orientation, which differs from cell to cell."""
    verts = np.asarray(_REFERENCE_CELLS[(dim, simplex)])
    coords = np.zeros((verts.shape[0], cdim))
    coords[:, :dim] = verts
    cells = np.arange(verts.shape[0], dtype=PETSc.IntType).reshape(1, -1)
    ref = PETSc.DMPlex().createFromCellList(dim, cells, coords, interpolate=True,
                                            comm=PETSc.COMM_SELF)
    pair = []
    for fe in (fe_src, fe_dst):
        rdm = ref.clone()
        rdm.setField(0, fe)
        rdm.createDS()
        pair.append(rdm)
    try:
        cell = ref.getHeightStratum(0)[0]
        points = ref.getTransitiveClosure(cell)[0]
        for rdm in pair:
            sec = rdm.getLocalSection()
            if any(sec.getDof(p) > block for p in points if p != cell):
                return None
        interp, scale = pair[0].createInterpolation(pair[1])
        full = interp.convert("dense").getDenseArray().copy()
        interp.destroy()
        if scale is not None:
            scale.destroy()
        rows_src, rows_dst = (closure_dof_rows(rdm, rdm.getLocalSection(), [cell], block)[0]
                              for rdm in pair)
        return full[np.ix_(rows_dst * block, rows_src * block)]
    finally:
        for rdm in pair:
            rdm.destroy()
        ref.destroy()


def interpolate_within_plex(plex, dm_src, x_src, dm_dst, block, ghosts=True):
    """Field 0 of ``dm_dst`` interpolated from ``x_src``, the local array of
    field 0 of ``dm_src``, where both DMs share the topology of ``plex`` (the
    mesh's DMPlex; a sub-DM is a plain DM in petsc4py, without the plex queries).

    The result is ``dm_dst``'s local array, shaped ``(n, block)`` (``ghosts``
    True) or its owned part (``ghosts`` False), with each node's value computed
    by the rank that owns it. Collective on the DMs' communicator.
    """
    fe_src = dm_src.getField(0)[0]
    fe_dst = dm_dst.getField(0)[0]
    element = _reference_element_matrix(fe_src, fe_dst, plex.getDimension(),
                                        plex.getCoordinateDim(), plex.isSimplex(), block)

    cS, cE = plex.getHeightStratum(0)
    cells = np.arange(cS, cE)
    if element is not None and cells.size:
        rows_src = closure_dof_rows(dm_src, dm_src.getLocalSection(), cells, block)
        rows_dst = closure_dof_rows(dm_dst, dm_dst.getLocalSection(), cells, block)
        uniform = (rows_src.shape[1] == element.shape[1]
                   and rows_dst.shape[1] == element.shape[0]
                   and (rows_src >= 0).all() and (rows_dst >= 0).all())
    else:
        uniform = element is not None
    # one verdict for every rank: the two routes below are collective
    mpi_comm = dm_src.comm.tompi4py()
    if not mpi_comm.allreduce(bool(uniform), op=MPI.LAND):
        return _interpolate_by_matrix(plex, dm_src, x_src, dm_dst, block, ghosts)

    lvec = dm_dst.getLocalVec()
    gvec = dm_dst.getGlobalVec()
    out = lvec.getArray().reshape(-1, block)
    if cells.size:
        x = np.asarray(x_src).reshape(-1, block)
        out[rows_dst.ravel()] = np.einsum("ab,cbk->cak", element,
                                          x[rows_src]).reshape(-1, block)
    # INSERT copies only the points a rank owns, so each node keeps its owner's value
    dm_dst.localToGlobal(lvec, gvec, addv=PETSc.InsertMode.INSERT_VALUES)
    if ghosts:
        dm_dst.globalToLocal(gvec, lvec, addv=PETSc.InsertMode.INSERT_VALUES)
        result = lvec.getArray().reshape(-1, block).copy()
    else:
        result = gvec.getArray().reshape(-1, block).copy()
    dm_dst.restoreGlobalVec(gvec)
    dm_dst.restoreLocalVec(lvec)
    return result


def _interpolate_by_matrix(plex, dm_src, x_src, dm_dst, block, ghosts):
    """PETSc's own interpolation matrix, for the fields whose edge or face nodes
    are ordered by orientation. It fails on a rank that holds cells but owns no
    row and no column, so that partition raises here, on every rank, instead."""
    rows = dm_dst.getGlobalSection().getConstrainedStorageSize()
    cols = dm_src.getGlobalSection().getConstrainedStorageSize()
    cS, cE = plex.getHeightStratum(0)
    mpi_comm = dm_src.comm.tompi4py()
    if mpi_comm.allreduce(bool(cE > cS and rows == 0 and cols == 0), op=MPI.LOR):
        raise NotImplementedError(
            "interpolation between two fields of one mesh: a rank holds cells but "
            "owns none of their degrees of freedom, where PETSc's interpolation "
            "matrix cannot be assembled (#902), and this field has more than one "
            "node on an edge or face. Run on fewer ranks or repartition the mesh.")
    gsrc = dm_src.getGlobalVec()
    lsrc = dm_src.getLocalVec()
    lsrc.getArray()[:] = np.asarray(x_src).ravel()
    dm_src.localToGlobal(lsrc, gsrc, addv=PETSc.InsertMode.INSERT_VALUES)
    interp, scale = dm_src.createInterpolation(dm_dst)
    gdst = dm_dst.getGlobalVec()
    interp.mult(gsrc, gdst)
    if ghosts:
        ldst = dm_dst.getLocalVec()
        dm_dst.globalToLocal(gdst, ldst, addv=PETSc.InsertMode.INSERT_VALUES)
        result = ldst.getArray().reshape(-1, block).copy()
        dm_dst.restoreLocalVec(ldst)
    else:
        result = gdst.getArray().reshape(-1, block).copy()
    dm_dst.restoreGlobalVec(gdst)
    dm_src.restoreLocalVec(lsrc)
    dm_src.restoreGlobalVec(gsrc)
    interp.destroy()
    if scale is not None:
        scale.destroy()
    return result

