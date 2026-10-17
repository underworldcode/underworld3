r"""Split-level geometric multigrid for split-node faults.

A split-node fault lives on the finest mesh. The coarse levels under it come from
the add-on hierarchy: the static background, which is never cut, and the add-on
levels refined from it toward the fault. This module carries the fault through
the add-on levels — every add-on level is split the same way as the finest — so
the coarse spaces contain the discontinuity and the velocity solve stays
mesh-independent as the fault is refined (iterations fall from 10 to 5 over four
halvings, against 9 to 32 with welded levels; see
``docs/developer/design/FAULT_MULTIGRID_HIERARCHIES.md``).

Three pieces make a split level usable as a multigrid level:

* **A side-aware transfer.** For each fine node one fine cell containing it is
  taken and that cell's CENTROID is located in the coarse mesh. A centroid is
  never on the fault, so the coarse cell is on the node's own side, and a node on
  the slit interpolates from its own side. The coarse cell's Lagrange interpolant
  is evaluated at the node by the dual-basis identity :math:`W = BM^{-1}` (as in
  :func:`custom_mg.nested_refine_pair_prolongation`). The levels are conforming
  but not nested near the trace (each cut pulls vertices onto it), so a fine node
  can lie a little outside its located coarse cell: the evaluation point is
  clamped into the cell (barycentric coordinates clipped and renormalised), which
  keeps every weight bounded by the Lagrange basis inside the simplex.
* **A per-level pair frame.** Each split level carries its own pair rotation
  :math:`Q_l` (mean and jump in the fault frame) and its opening rows are removed:
  :math:`\hat P_l = Q_{l+1} P_l Q_l^T` restricted to the admissible unknowns.
* **Unreached coarse unknowns dropped.** A coarse unknown no fine unknown reaches
  (a few near the tips) is a null mode of the Galerkin coarse operator; it is
  removed from the level rather than repaired by injection.

Parallel: each split level is redistributed so its fault star is rank-interior
(the split's own rule), so pair blocks are rank-local on every level. The transfer
locates each owned fine node in the LOCAL coarse mesh (owned plus ghost cells);
the few nodes whose coarse cell lives elsewhere are gathered as query points and
answered by the rank that holds the cell, with global column indices.
"""

import itertools

import numpy as np
from types import SimpleNamespace
from mpi4py import MPI
import scipy.sparse as sp
from petsc4py import PETSc
from scipy.spatial import cKDTree

import underworld3 as uw

# Barycentric slack for "inside": a fine node this far outside (in reference
# coordinates) still counts as located in the cell; beyond it the clamp acts.
_INSIDE_TOL = 1e-8
# Chunk size for the batched weight evaluation (bounds the (n, nb, nb) temporaries).
_CHUNK = 1 << 16
# The last parallel transfer's hit-rate record (see cell_located_transfer_parallel);
# CustomMGHierarchy.build copies it per level into ``level_transfer_stats``.
last_transfer_stats = None


# --------------------------------------------------------------------------- #
#  Geometry
# --------------------------------------------------------------------------- #
def is_split_level(mesh):
    """Does this level carry a split-node fault (a coincident DOF pairing)?"""
    return bool(getattr(mesh, "_fault_point_pairs", None))


def _cell_vertex_table(mesh):
    """Vertex coordinates ``(n_v, cdim)`` and the ``(n_cells, dim + 1)`` table of
    vertex rows per cell — the P1 node table, which the mesh caches."""
    vxy = np.asarray(mesh._get_coords_for_basis(1, True))
    cv = np.asarray(mesh._cell_node_indices(1, True))
    return vxy, cv


def _barycentric(P, V):
    """Barycentric coordinates of points ``P (n, cdim)`` in the simplices
    ``V (n, dim + 1, cdim)``. For an embedded surface (``cdim > dim``) the point is
    projected onto the simplex plane first (normal equations)."""
    A = V[:, 1:] - V[:, :1]                     # (n, dim, cdim)
    b = P - V[:, 0]                             # (n, cdim)
    dim, cdim = A.shape[1], A.shape[2]
    if dim == cdim:
        lam = np.linalg.solve(np.transpose(A, (0, 2, 1)), b[..., None])[..., 0]
    else:
        G = np.einsum("nik,njk->nij", A, A)
        rhs = np.einsum("nik,nk->ni", A, b)
        lam = np.linalg.solve(G, rhs[..., None])[..., 0]
    return np.concatenate([1.0 - lam.sum(axis=1, keepdims=True), lam], axis=1)


def locate_in_cells(points, vxy, cv, k=16):
    """Locate ``points`` among the simplices ``cv`` (vertex rows into ``vxy``).

    Returns ``(cell, lam)``: the best candidate cell per point (the one whose
    minimum barycentric coordinate is largest among the ``k`` nearest by centroid)
    and its barycentric coordinates there. A point is INSIDE its cell when
    ``lam.min() >= -_INSIDE_TOL``; otherwise the caller clamps or asks another rank.
    Empty ``cv`` returns cells of -1."""
    n = points.shape[0]
    dim1 = cv.shape[1]
    best = np.full(n, -1, dtype=np.int64)
    lam_best = np.full((n, dim1), -np.inf)
    if n == 0 or cv.shape[0] == 0:
        return best, lam_best
    cent = vxy[cv].mean(axis=1)
    kk = min(k, cv.shape[0])
    _, cand = cKDTree(cent).query(points, k=kk)
    cand = np.asarray(cand).reshape(n, kk)
    bestmin = np.full(n, -np.inf)
    for j in range(kk):
        c = cand[:, j]
        lam = _barycentric(points, vxy[cv[c]])
        m = lam.min(axis=1)
        better = m > bestmin
        best[better] = c[better]
        lam_best[better] = lam[better]
        bestmin[better] = m[better]
    return best, lam_best


def _clamp(lam):
    """Clip barycentric coordinates into the simplex and renormalise — the
    nearest point of the cell in reference coordinates for a node a little
    outside it (non-nested levels near a pulled tip). Every weight then stays
    bounded by the Lagrange basis inside the simplex."""
    out = lam.copy()
    bad = (out < 0).any(axis=1)
    if bad.any():
        b = np.clip(out[bad], 0.0, None)
        out[bad] = b / b.sum(axis=1, keepdims=True)
    return out


def _monomial_exponents(degree, dim):
    return np.asarray([e for e in itertools.product(range(degree + 1), repeat=dim)
                       if sum(e) <= degree])


def _vander(L, E):
    """Monomials ``E`` of the pulled-back coordinates ``L (n, dim)``: ``(n, nb)``."""
    return np.prod(L[:, None, :] ** E[None, :, :], axis=2)


def _pullback(X, V):
    """Reference coordinates of points ``X (n, cdim)`` in simplices ``V (n, dim + 1,
    cdim)`` — the barycentric vector without its first entry."""
    return _barycentric(X, V)[:, 1:]


def _node_weights(Xf, cells, lam, Xc, cn, vxy_c, cv_c, degree):
    """Weights of the fine nodes ``Xf`` on the nodes of their located coarse
    ``cells`` (barycentric ``lam`` there, clamped): ``(n, nb)``.

    Batched dual-basis evaluation: per distinct coarse cell the inverse nodal
    Vandermonde ``M^{-1}`` (nodes pulled back through the cell's own affine map),
    then ``W = B M^{-1}`` row by row in chunks."""
    dim = cv_c.shape[1] - 1
    E = _monomial_exponents(degree, dim)
    nb = E.shape[0]
    if cn.shape[1] != nb:
        raise RuntimeError(
            f"fault_mg: {cn.shape[1]} nodes per cell against {nb} monomials "
            f"of degree {degree} in {dim}-D")
    used, inv = np.unique(cells, return_inverse=True)
    Vc = vxy_c[cv_c[used]]                                   # (nu, dim+1, cdim)
    # coarse nodes of each used cell, pulled back through that cell's map
    Xn = Xc[cn[used]]                                        # (nu, nb, cdim)
    Lc = _pullback(Xn.reshape(-1, Xn.shape[-1]),
                   np.repeat(Vc, nb, axis=0)).reshape(len(used), nb, dim)
    M = _vander(Lc.reshape(-1, dim), E).reshape(len(used), nb, nb)
    Minv = np.linalg.inv(M)
    Lf = _clamp(lam)[:, 1:]                                  # (n, dim)
    W = np.empty((Xf.shape[0], nb))
    for s in range(0, Xf.shape[0], _CHUNK):
        e = min(s + _CHUNK, Xf.shape[0])
        B = _vander(Lf[s:e], E)                              # (m, nb)
        W[s:e] = np.einsum("tm,tmk->tk", B, Minv[inv[s:e]])
    return W


def _fine_cell_of_node(fn, n_nodes):
    """One fine cell containing each fine node (the last cell listing it). A
    node on the slit is listed only by cells on its own side."""
    owner = np.full(n_nodes, -1, dtype=np.int64)
    for k in range(fn.shape[0]):
        owner[fn[k]] = k
    return owner


# --------------------------------------------------------------------------- #
#  Serial transfer
# --------------------------------------------------------------------------- #
def cell_located_prolongation(coarse, fine, degree, continuous=True,
                              coarse_coords=None, fine_coords=None):
    """Side-aware, cell-located node prolongation ``(n_fine_nodes, n_coarse_nodes)``
    as scipy CSR (serial). Exact for nested pairs; on non-nested pairs it
    reproduces the located coarse cell's polynomial, with the evaluation point
    clamped into the cell."""
    cn = np.asarray(coarse._cell_node_indices(degree, continuous))
    fn = np.asarray(fine._cell_node_indices(degree, continuous))
    Xc = (np.asarray(coarse_coords) if coarse_coords is not None
          else np.asarray(coarse._get_coords_for_basis(degree, continuous)))
    Xf = (np.asarray(fine_coords) if fine_coords is not None
          else np.asarray(fine._get_coords_for_basis(degree, continuous)))
    vxy_c, cv_c = _cell_vertex_table(coarse)
    vxy_f, cv_f = _cell_vertex_table(fine)
    owner = _fine_cell_of_node(fn, Xf.shape[0])
    if (owner < 0).any():
        raise RuntimeError("fault_mg: a fine node belongs to no cell")
    fcent = vxy_f[cv_f].mean(axis=1)
    pcell, _lam_c = locate_in_cells(fcent, vxy_c, cv_c)
    cells = pcell[owner]
    lam = _barycentric(Xf, vxy_c[cv_c[cells]])
    W = _node_weights(Xf, cells, lam, Xc, cn, vxy_c, cv_c, degree)
    nb = W.shape[1]
    rows = np.repeat(np.arange(Xf.shape[0]), nb)
    cols = cn[cells].ravel()
    vals = W.ravel()
    keep = np.abs(vals) > 1e-13
    P = sp.csr_matrix((vals[keep], (rows[keep], cols[keep])),
                      shape=(Xf.shape[0], Xc.shape[0]))
    P.sum_duplicates()
    return P


# --------------------------------------------------------------------------- #
#  Parallel transfer
# --------------------------------------------------------------------------- #
def cell_located_transfer_parallel(coarse, fine, degree, continuous,
                                   coarse_coords, fine_coords, coarse_layout,
                                   fine_layout, ncomp, comm):
    """The same transfer as :func:`cell_located_prolongation`, reduced->reduced,
    as an MPIAIJ matrix.

    Every rank locates its OWNED fine rows in its LOCAL coarse mesh (owned plus
    ghost cells, whose nodes carry the owner's global index through the layout).
    A fine node whose coarse cell is not local is a MISS: the misses are
    all-gathered as (centroid, node position) queries, every rank answers the
    ones it finds in its own cells, and the asker takes the best answer (the
    largest minimum barycentric coordinate). A query nobody finds — a node outside
    every coarse cell, as at a curved boundary — keeps the asker's own clamped
    best. ``coarse_layout`` / ``fine_layout`` are :class:`custom_mg.LevelLayout`.
    """
    m4 = comm.tompi4py()
    cn = np.asarray(coarse._cell_node_indices(degree, continuous))
    fn = np.asarray(fine._cell_node_indices(degree, continuous))
    Xc = np.asarray(coarse_coords)
    Xf = np.asarray(fine_coords)
    vxy_c, cv_c = _cell_vertex_table(coarse)
    vxy_f, cv_f = _cell_vertex_table(fine)
    l2g_c = np.asarray(coarse_layout.l2g, dtype=np.int64).reshape(-1, ncomp)
    l2g_f = np.asarray(fine_layout.l2g, dtype=np.int64).reshape(-1, ncomp)
    fstart, fend = fine_layout.rstart, fine_layout.rend
    cstart, cend = coarse_layout.rstart, coarse_layout.rend
    n_f = Xf.shape[0]
    nb = cn.shape[1]

    owner = _fine_cell_of_node(fn, n_f)
    if (owner < 0).any():
        raise RuntimeError("fault_mg: a fine node belongs to no cell")
    # the fine nodes with at least one owned row (ghost nodes are another rank's)
    mine = np.flatnonzero(((l2g_f >= fstart) & (l2g_f < fend)).any(axis=1))
    fcent = vxy_f[cv_f].mean(axis=1)
    pcell, lam_c = (locate_in_cells(fcent[owner[mine]], vxy_c, cv_c)
                    if cv_c.shape[0] else
                    (np.full(mine.size, -1), np.full((mine.size, cv_c.shape[1]), -np.inf)))
    inside = lam_c.min(axis=1) >= -_INSIDE_TOL

    # --- local hits ------------------------------------------------------- #
    hit = mine[inside]
    cells = pcell[inside]
    gcols = np.empty((0, nb, ncomp), dtype=np.int64)
    W = np.empty((0, nb))
    if hit.size:
        lam = _barycentric(Xf[hit], vxy_c[cv_c[cells]])
        W = _node_weights(Xf[hit], cells, lam, Xc, cn, vxy_c, cv_c, degree)
        gcols = l2g_c[cn[cells]]                             # (nh, nb, ncomp)
    hit_nodes, hit_W, hit_cols = [hit], [W], [gcols]

    # --- misses: ask the other ranks -------------------------------------- #
    miss = mine[~inside]
    # a record of the rank-local hit rate, for the hierarchy's report
    global last_transfer_stats
    last_transfer_stats = {
        "owned_nodes": int(m4.allreduce(int(mine.size))),
        "misses": int(m4.allreduce(int(miss.size))),
        "local_coarse_cells": m4.allgather(int(cv_c.shape[0])),
        "local_fine_cells": m4.allgather(int(cv_f.shape[0])),
    }
    q_cent = fcent[owner[miss]]
    q_node = Xf[miss]
    all_cent = m4.allgather(q_cent)
    all_node = m4.allgather(q_node)
    counts = np.array([a.shape[0] for a in all_cent])
    total = int(counts.sum())
    if total:
        Qc = (np.vstack(all_cent) if total else
              np.zeros((0, Xf.shape[1])))
        Qn = np.vstack(all_node)
        offs = np.concatenate([[0], np.cumsum(counts)])
        if cv_c.shape[0]:
            qcell, qlam = locate_in_cells(Qc, vxy_c, cv_c)
            qin = qlam.min(axis=1) >= -_INSIDE_TOL
        else:
            qcell = np.full(total, -1)
            qlam = np.full((total, cv_c.shape[1]), -np.inf)
            qin = np.zeros(total, dtype=bool)
        found = np.flatnonzero(qin)
        if found.size:
            lamn = _barycentric(Qn[found], vxy_c[cv_c[qcell[found]]])
            Wq = _node_weights(Qn[found], qcell[found], lamn, Xc, cn,
                               vxy_c, cv_c, degree)
            colsq = l2g_c[cn[qcell[found]]]
            score = qlam[found].min(axis=1)
        else:
            Wq = np.zeros((0, nb))
            colsq = np.zeros((0, nb, ncomp), dtype=np.int64)
            score = np.zeros(0)
        # answers travel back to every rank; each keeps those for its own queries
        ans_idx = m4.allgather(found)
        ans_W = m4.allgather(Wq)
        ans_cols = m4.allgather(colsq)
        ans_score = m4.allgather(score)
        lo, hi = offs[m4.rank], offs[m4.rank + 1]
        best_score = np.full(miss.size, -np.inf)
        best_W = np.zeros((miss.size, nb))
        best_cols = np.full((miss.size, nb, ncomp), -1, dtype=np.int64)
        answered = np.zeros(miss.size, dtype=bool)
        for idx, Wa, ca, sa in zip(ans_idx, ans_W, ans_cols, ans_score):
            sel = (idx >= lo) & (idx < hi)
            if not sel.any():
                continue
            local = idx[sel] - lo
            better = sa[sel] > best_score[local]
            local = local[better]
            best_score[local] = sa[sel][better]
            best_W[local] = Wa[sel][better]
            best_cols[local] = ca[sel][better]
            answered[local] = True
        # nobody found it: this rank's own clamped best (if it has coarse cells)
        rest = np.flatnonzero(~answered)
        if rest.size and cv_c.shape[0]:
            cells_r = pcell[~inside][rest]
            lam_r = _barycentric(Xf[miss[rest]], vxy_c[cv_c[cells_r]])
            best_W[rest] = _node_weights(Xf[miss[rest]], cells_r, lam_r, Xc, cn,
                                         vxy_c, cv_c, degree)
            best_cols[rest] = l2g_c[cn[cells_r]]
            answered[rest] = True
        if not answered.all():
            raise RuntimeError(
                "fault_mg: a fine node could not be located in any coarse cell "
                "on any rank")
        hit_nodes.append(miss)
        hit_W.append(best_W)
        hit_cols.append(best_cols)

    nodes = np.concatenate(hit_nodes)
    Wall = np.vstack(hit_W)
    Call = np.concatenate(hit_cols) if len(hit_cols) > 1 else hit_cols[0]

    # --- assemble owned rows ---------------------------------------------- #
    return _assemble_owned_rows(nodes, Wall, Call, l2g_f, fstart, fend, cstart,
                                cend, ncomp, nb, comm)


def locate_in_cells_exhaustive(points, vxy, cv):
    """:func:`locate_in_cells`, widening the search for every point still outside
    its candidate. On a graded mesh a large cell's centroid can be farther from a
    point inside it than many small neighbours' centroids are, so the nearest
    sixteen can all miss: the misses are retried against the nearest 128 and
    1024, and what still misses is tested against every cell, one point at a
    time (memory one cell table, not points times cells). Returns
    ``(cell, lam)``; a point inside no cell keeps its best candidate with
    ``lam.min() < -_INSIDE_TOL``."""
    cell, lam = locate_in_cells(points, vxy, cv)
    if cv.shape[0] == 0:
        return cell, lam
    for k in (128, 1024):
        miss = np.flatnonzero(lam.min(axis=1) < -_INSIDE_TOL)
        if miss.size == 0:
            return cell, lam
        c2, l2 = locate_in_cells(points[miss], vxy, cv, k=k)
        better = l2.min(axis=1) > lam[miss].min(axis=1)
        cell[miss[better]] = c2[better]
        lam[miss[better]] = l2[better]
    V = vxy[cv]
    for k in np.flatnonzero(lam.min(axis=1) < -_INSIDE_TOL):
        lam_all = _barycentric(np.repeat(points[k][None, :], cv.shape[0], axis=0), V)
        best = int(np.argmax(lam_all.min(axis=1)))
        if lam_all[best].min() > lam[k].min():
            cell[k], lam[k] = best, lam_all[best]
    return cell, lam


# --------------------------------------------------------------------------- #
#  The provenance transfer: the one pair the add-on partition moved
# --------------------------------------------------------------------------- #
def provenance_source(coarse, fine):
    """The mesh that records, per cell, where its parent on ``coarse`` lives —
    ``fine`` itself or an ancestor of it (the add-on level as the add-on
    partition left it, before the cut and the split re-represented it; see
    ``addon_partition``). ``None`` when the pair carries no such record."""
    m = fine
    while m is not None:
        if (getattr(m, "_addon_below", None) is coarse
                and getattr(m, "_addon_parent", None) is not None):
            return m
        m = getattr(m, "parent", None)
    return None


def provenance_transfer_parallel(coarse, fine, pre, degree, continuous,
                                 coarse_coords, fine_coords, coarse_layout,
                                 fine_layout, ncomp, comm):
    """The cell-located transfer for a pair the add-on partition moved apart,
    built from the recorded parent relation instead of a search.

    ``pre`` carries the record (``provenance_source``): every cell of ``pre``
    knows the rank and local index of its parent on ``coarse``. It is ``fine``
    itself whenever the record was inherited when ``fine`` was made (the cut,
    the placement and the split call ``addon_partition.inherit_provenance``),
    and then a node's own cell is the recorded cell. Otherwise the fine cell's
    centroid is located in ``pre`` (rank-local, since the surgery was, and a
    centroid is never on the fault). The recorded parent is then fetched — its node columns and vertex and node
    coordinates — through one star forest. Co-resident parents are a local
    copy; the moved units' parents come from their ranks. Nothing is
    all-gathered. ``coarse_layout`` / ``fine_layout`` are
    :class:`custom_mg.LevelLayout`.
    """
    m4 = comm.tompi4py()
    cn = np.asarray(coarse._cell_node_indices(degree, continuous))
    fn = np.asarray(fine._cell_node_indices(degree, continuous))
    Xc = np.asarray(coarse_coords)
    Xf = np.asarray(fine_coords)
    vxy_c, cv_c = _cell_vertex_table(coarse)
    vxy_f, cv_f = _cell_vertex_table(fine)
    vxy_p, cv_p = _cell_vertex_table(pre)
    l2g_c = np.asarray(coarse_layout.l2g, dtype=np.int64).reshape(-1, ncomp)
    l2g_f = np.asarray(fine_layout.l2g, dtype=np.int64).reshape(-1, ncomp)
    fstart, fend = fine_layout.rstart, fine_layout.rend
    cstart, cend = coarse_layout.rstart, coarse_layout.rend
    n_f = Xf.shape[0]
    nb = cn.shape[1]
    nv = cv_c.shape[1]
    dim = Xc.shape[1]

    owner = _fine_cell_of_node(fn, n_f)
    if m4.allreduce(int((owner < 0).sum())):
        raise RuntimeError("fault_mg: a fine node belongs to no cell")
    mine = np.flatnonzero(((l2g_f >= fstart) & (l2g_f < fend)).any(axis=1))
    fcent = vxy_f[cv_f].mean(axis=1)
    if pre is fine:
        # the record is on this very mesh: a node's cell IS the recorded cell
        pcell = owner[mine]
        plam = np.zeros((mine.size, nv))
    elif mine.size:
        pcell, plam = locate_in_cells_exhaustive(fcent[owner[mine]], vxy_p, cv_p)
    else:
        pcell = np.zeros(0, dtype=np.int64)
        plam = np.zeros((0, nv))
    n_out = int(m4.allreduce(int((plam.min(axis=1) < -_INSIDE_TOL).sum())
                             if mine.size else 0))
    if n_out:
        raise RuntimeError(
            f"fault_mg: {n_out} fine node(s) lie in no cell of the level as the "
            f"add-on partition left it; the surgery was not rank-local.")
    parent = np.asarray(pre._addon_parent, dtype=np.int32)[pcell]   # (nm, 2)

    # one fetch per distinct parent: node columns, node coordinates and
    # vertex coordinates, packed as doubles (indices exact below 2**53)
    uniq, inv = np.unique(parent, axis=0, return_inverse=True)
    inv = inv.ravel()
    width = nb * ncomp + nb * dim + nv * dim
    nc_loc = cv_c.shape[0]
    root = np.empty((nc_loc, width))
    root[:, : nb * ncomp] = l2g_c[cn].reshape(nc_loc, -1)
    root[:, nb * ncomp: nb * ncomp + nb * dim] = Xc[cn].reshape(nc_loc, -1)
    root[:, nb * ncomp + nb * dim:] = vxy_c[cv_c].reshape(nc_loc, -1)
    sf = PETSc.SF().create(comm=comm)
    sf.setGraph(nc_loc, None, np.ascontiguousarray(uniq, dtype=np.int32))
    sf.setUp()
    got = np.zeros((uniq.shape[0], width))
    unit = MPI.DOUBLE.Create_contiguous(width)
    unit.Commit()
    sf.bcastBegin(unit, root, got, MPI.REPLACE)
    sf.bcastEnd(unit, root, got, MPI.REPLACE)
    unit.Free()
    sf.destroy()
    cols_u = np.rint(got[:, : nb * ncomp]).astype(np.int64).reshape(-1, nb, ncomp)
    Xn_u = got[:, nb * ncomp: nb * ncomp + nb * dim].reshape(-1, dim)
    V_u = got[:, nb * ncomp + nb * dim:].reshape(-1, dim)
    # the fetched parents as a virtual coarse mesh: cell k has nodes
    # k*nb .. and vertices k*nv ..
    nu = uniq.shape[0]
    cn_u = np.arange(nu * nb).reshape(nu, nb)
    cv_u = np.arange(nu * nv).reshape(nu, nv)
    lam = _barycentric(Xf[mine], V_u[cv_u[inv]]) if mine.size else np.zeros((0, nv))
    W = (_node_weights(Xf[mine], inv, lam, Xn_u, cn_u, V_u, cv_u, degree)
         if mine.size else np.zeros((0, nb)))
    gcols = cols_u[inv]                                        # (nm, nb, ncomp)

    global last_transfer_stats
    last_transfer_stats = {
        "owned_nodes": int(m4.allreduce(int(mine.size))),
        "fetched_remote": int(m4.allreduce(int((uniq[:, 0] != m4.rank).sum()))),
        "local_coarse_cells": m4.allgather(int(nc_loc)),
        "local_fine_cells": m4.allgather(int(cv_f.shape[0])),
    }
    return _assemble_owned_rows(mine, W, gcols, l2g_f, fstart, fend, cstart,
                                cend, ncomp, nb, comm)


def _assemble_owned_rows(nodes, Wall, Call, l2g_f, fstart, fend, cstart, cend,
                         ncomp, nb, comm):
    """The owned rows of a reduced->reduced MPIAIJ transfer from per-node
    weights ``Wall (n, nb)`` on global coarse columns ``Call (n, nb, ncomp)``."""
    m4 = comm.tompi4py()
    nloc_f = fend - fstart
    nloc_c = cend - cstart
    rows_l, cols_l, vals_l = [], [], []
    for c in range(ncomp):
        grow = l2g_f[nodes, c]
        own = (grow >= fstart) & (grow < fend)
        if not own.any():
            continue
        gr = np.repeat(grow[own], nb)
        gc = Call[own][:, :, c].ravel()
        gv = Wall[own].ravel()
        keep = (gc >= 0) & (np.abs(gv) > 1e-13)
        rows_l.append(gr[keep] - fstart)
        cols_l.append(gc[keep])
        vals_l.append(gv[keep])
    if rows_l:
        rows_l = np.concatenate(rows_l)
        cols_l = np.concatenate(cols_l)
        vals_l = np.concatenate(vals_l)
    else:
        rows_l = np.zeros(0, dtype=np.int64)
        cols_l = np.zeros(0, dtype=np.int64)
        vals_l = np.zeros(0)
    loc = sp.csr_matrix((vals_l, (rows_l, cols_l)),
                        shape=(nloc_f, int(m4.allreduce(nloc_c))))
    loc.sum_duplicates()
    loc.sort_indices()
    P = PETSc.Mat().createAIJ(
        size=((nloc_f, None), (nloc_c, None)),
        csr=(loc.indptr.astype(PETSc.IntType), loc.indices.astype(PETSc.IntType),
             loc.data), comm=comm)
    P.assemble()
    return P


# --------------------------------------------------------------------------- #
#  Splitting the add-on levels
# --------------------------------------------------------------------------- #
def split_tail_levels(levels, fault_specs, verbose=False):
    """Replay the finest mesh's faults on each add-on level; return the levels
    (same order), split where the replay succeeds.

    ``fault_specs`` is the list ``fault_split.add_fault`` records on its child:
    ``("line", name, polyline)`` in 2-D, ``("sheet", name, points, triangles,
    placement options)`` in 3-D. Each level is cut and split exactly as the
    finest was, so its slit lies on the same geometry and the centroid rule of
    the transfer assigns sides consistently across levels. A level carries no
    tail of its own.

    The replay runs from the finest level down. A level the faults cannot be
    carried onto — a network whose one-cell ligament closes at the coarser
    spacing, so two labels meet; a sheet the placement refuses — is left WELDED,
    and so is every level beneath it (a split level under a welded one would
    hand the welded level's single trace node one side's value). The hierarchy
    then has split levels down to the finest coarsening the geometry allows,
    and a warning says where the replay stopped and why.
    """
    import warnings

    from underworld3.utilities.fault_split import add_fault

    entries = [tuple(spec[1:4]) for spec in fault_specs]
    kwargs = {}
    for spec in fault_specs:
        if spec[0] == "sheet" and len(spec) > 4:
            kwargs.update(spec[4])
    out = list(levels)
    if not entries:
        return out
    for k in range(len(levels) - 1, -1, -1):
        level = levels[k]
        try:
            # the level arrives partitioned with the finest (the add-on
            # partition ran once, for every level): no second partition
            split = add_fault(level, entries, verbose=verbose, partition=False,
                              **kwargs)
        except (ValueError, RuntimeError, NotImplementedError) as exc:
            warnings.warn(
                f"fault_mg: the fault(s) could not be carried onto add-on "
                f"level {k} of {len(levels)} ({exc}); that level and the "
                f"levels beneath it stay welded, the {len(levels) - 1 - k} "
                f"finer add-on level(s) are split.")
            break
        split._custom_mg_coarse_meshes = None
        out[k] = split
    return out


# --------------------------------------------------------------------------- #
#  Per-level pair rotation
# --------------------------------------------------------------------------- #
def level_pair_rotation(level, solver, field_id, cdm=None, layout=None):
    """The pair rotation of a split coarse level in the level's reduced
    velocity layout: ``(Q, Qt, normal_rows, layout)`` with ``Q`` an AIJ matrix
    (identity except one orthogonal ``2 dim`` block per coincident pair, mean rows
    on the Plus point and jump rows on the Minus point — the same construction as
    ``rotated_bc.build_rotation``) and ``normal_rows`` the global rows constrained
    out of the level (the opening jump; on a surface mesh also the jump and mean
    along the surface normal). The level's DOF layout is derived from the solver's
    discretisation copied onto the level DM (no throwaway solver). Not cached:
    the layout depends on the solver's essential conditions, and the rotated
    solve already keeps the composed prolongations across its Newton steps."""
    from underworld3.utilities import custom_mg, fault_contact

    if cdm is None:
        cdm = custom_mg._clone_dm_with_solver_discretisation(solver, level)
    if layout is None:
        layout = custom_mg._level_dof_layout(cdm, field_id)
    sub = custom_mg._field_subdm(cdm, field_id)
    lsec_sub = sub.getLocalSection()
    lsec = cdm.getLocalSection()
    dim = level.cdim
    manifold = level.dim < level.cdim
    l2g = np.asarray(layout.l2g, dtype=np.int64)
    rstart, rend = layout.rstart, layout.rend
    nloc = rend - rstart
    comm = cdm.getComm()
    N = int(comm.tompi4py().allreduce(nloc))

    # identity rows, replaced block by block
    rows = [np.arange(rstart, rend)]
    cols = [np.arange(rstart, rend)]
    vals = [np.ones(nloc)]
    is_block_row = np.zeros(nloc, dtype=bool)
    normal_rows = []
    s2 = 1.0 / np.sqrt(2.0)
    radial = fault_contact._radial_fn(level) if manifold else None
    problem = None
    try:
        for fname in level._fault_point_pairs:
            pairs = fault_contact._fault_pair_nodes_on(
                level, cdm, lsec, fname, field=field_id or 0)
            for q_plus, q_minus, nrm in pairs:
                frame = fault_contact._pair_frame_on(level, cdm, q_plus, nrm,
                                                     radial)
                op, om = lsec_sub.getOffset(q_plus), lsec_sub.getOffset(q_minus)
                gp = [int(l2g[op + c]) for c in range(dim)]
                gm = [int(l2g[om + c]) for c in range(dim)]
                if any(g < 0 for g in gp + gm):
                    continue
                owned_p = rstart <= gp[0] < rend
                owned_m = rstart <= gm[0] < rend
                if owned_p != owned_m:
                    raise RuntimeError(
                        "fault_mg: a coincident pair straddles ranks on a "
                        "coarse level — the split's redistribution should "
                        "make this impossible.")
                if not owned_p:
                    continue
                is_block_row[np.array(gp + gm) - rstart] = True
                for i, e in enumerate(frame):
                    for row, sgn in ((gp[i], 1.0), (gm[i], -1.0)):
                        rows.append(np.full(2 * dim, row))
                        cols.append(np.array(gp + gm))
                        vals.append(np.concatenate([s2 * e, sgn * s2 * e]))
                normal_rows.append(gm[0])
                if manifold:
                    normal_rows.append(gm[2])
                    normal_rows.append(gp[2])
    except (RuntimeError, ValueError) as err:
        problem = str(err)
    fault_contact._collective_raise(problem)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    # drop the identity entry of every block row (the first nloc entries)
    keep = np.ones(len(rows), dtype=bool)
    keep[:nloc] = ~is_block_row
    loc = sp.csr_matrix((vals[keep], (rows[keep] - rstart, cols[keep])),
                        shape=(nloc, N))
    loc.sum_duplicates()
    loc.sort_indices()
    Q = PETSc.Mat().createAIJ(
        size=((nloc, None), (nloc, None)),
        csr=(loc.indptr.astype(PETSc.IntType), loc.indices.astype(PETSc.IntType),
             loc.data), comm=comm)
    Q.assemble()
    Qt = Q.transpose(PETSc.Mat())
    return Q, Qt, np.array(sorted(set(normal_rows)), dtype=np.int64), layout


class _LevelMeshProxy:
    """The level mesh wearing the solver mesh's coordinate symbols, so the
    solver's analytic wall normals (expressions in ``solver.mesh.X``) evaluate
    on the level's nodes through the finest-level helpers unchanged."""

    def __init__(self, level, X):
        self._level = level
        self.X = X

    def __getattr__(self, name):
        return getattr(self._level, name)


def level_wall_rotation(level, solver, field_id, cdm=None, layout=None):
    """The rotated free-slip wall rotation of a coarse level in the level's
    reduced velocity layout: ``(Q, Qt, normal_rows, layout)`` with one
    orthonormal ``dim x dim`` block per velocity node on each of the solver's
    rotated boundaries (the same frame rule as ``rotated_bc.build_rotation``,
    on the level's own nodes and normals) and ``normal_rows`` the rows of the
    rotated normal components, which the level constrains out. ``None`` when
    the solver has no rotated walls.

    Why the coarse levels need it: with Cartesian coarse levels under a
    rotated finest whose normal rows take no coarse correction, a coarse wall
    node's normal Cartesian component is reached only through the fine
    tangential rows, weighted by the sine of the angle between the two
    nodes' normals. Its Galerkin block is then nearly singular (measured
    1e-3 against 2.5 on an annulus at a node near an axis), the inode SOR
    smoother's 2x2 block inversion reports a zero pivot, and the velocity
    solve fails before its first iteration. Rotating the level and removing
    the normal rows gives the coarse space exactly the constrained motion.
    """
    from underworld3.utilities import custom_mg, rotated_bc

    specs = list(getattr(solver, "_rotated_freeslip_bcs", None) or [])
    if not specs:
        return None
    if cdm is None:
        cdm = custom_mg._clone_dm_with_solver_discretisation(solver, level)
    if layout is None:
        layout = custom_mg._level_dof_layout(cdm, field_id)
    sub = custom_mg._field_subdm(cdm, field_id)
    lsec_sub = sub.getLocalSection()
    dim = level.cdim
    manifold = level.dim < level.cdim
    proxy = SimpleNamespace(dm=cdm, mesh=_LevelMeshProxy(level, solver.mesh.X),
                            snes=None)

    node_normals = {}
    for spec in specs:
        name, normal = rotated_bc._boundary_spec(spec)
        if name == rotated_bc.SHEET_TANGENCY:
            nodes = rotated_bc._manifold_velocity_nodes(proxy, normal)
        else:
            nodes = rotated_bc._boundary_velocity_nodes(proxy, name, normal=normal)
        for q, nrm in nodes:
            node_normals.setdefault(q, []).append(nrm)
    if manifold and is_split_level(level):
        # a paired node takes the sheet tangency through its pair block
        for pairs in level._fault_point_pairs.values():
            for q_plus, q_minus in (pairs.items() if isinstance(pairs, dict)
                                    else pairs):
                node_normals.pop(int(q_plus), None)
                node_normals.pop(int(q_minus), None)

    l2g = np.asarray(layout.l2g, dtype=np.int64)
    rstart, rend = layout.rstart, layout.rend
    nloc = rend - rstart
    comm = cdm.getComm()
    N = int(comm.tompi4py().allreduce(nloc))
    rows = [np.arange(rstart, rend)]
    cols = [np.arange(rstart, rend)]
    vals = [np.ones(nloc)]
    is_block_row = np.zeros(nloc, dtype=bool)
    normal_rows = []
    for q, nrms in node_normals.items():
        off = lsec_sub.getOffset(q)
        g = [int(l2g[off + c]) for c in range(dim)]
        if any(gi < 0 for gi in g):
            # a component pinned by an essential condition on the level: the
            # wall meets an essential wall there; left Cartesian, as the
            # finest's partially constrained nodes are handled separately
            continue
        if not (rstart <= g[0] < rend):
            continue
        M = np.array(nrms, dtype=float)
        _u, sv, Vt = np.linalg.svd(M)
        r = int((sv > 1e-8 * (sv[0] if sv.size else 1.0)).sum())
        is_block_row[np.array(g) - rstart] = True
        for i in range(dim):
            rows.append(np.full(dim, g[i]))
            cols.append(np.array(g))
            vals.append(np.asarray(Vt[i], dtype=float))
        normal_rows.extend(g[:r])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    keep = np.ones(len(rows), dtype=bool)
    keep[:nloc] = ~is_block_row
    loc = sp.csr_matrix((vals[keep], (rows[keep] - rstart, cols[keep])),
                        shape=(nloc, N))
    loc.sum_duplicates()
    loc.sort_indices()
    Q = PETSc.Mat().createAIJ(
        size=((nloc, None), (nloc, None)),
        csr=(loc.indptr.astype(PETSc.IntType), loc.indices.astype(PETSc.IntType),
             loc.data), comm=comm)
    Q.assemble()
    Qt = Q.transpose(PETSc.Mat())
    return Q, Qt, np.array(sorted(set(normal_rows)), dtype=np.int64), layout


def level_rotation(level, solver, field_id):
    """The full rotation of a coarse level: its fault pairs (a split level)
    and its rotated walls, composed. The two act on disjoint node blocks (a
    paired node is never a wall node on a volume mesh; on a surface mesh the
    paired nodes are removed from the wall set), so the product is a single
    orthogonal matrix. Returns ``(Q, Qt, normal_rows)`` or ``None`` when the
    level has neither."""
    from underworld3.utilities import custom_mg

    split = is_split_level(level)
    walls = bool(getattr(solver, "_rotated_freeslip_bcs", None))
    if not split and not walls:
        return None
    cdm = custom_mg._clone_dm_with_solver_discretisation(solver, level)
    layout = custom_mg._level_dof_layout(cdm, field_id)
    parts = []
    if split:
        parts.append(level_pair_rotation(level, solver, field_id, cdm=cdm, layout=layout))
    if walls:
        w = level_wall_rotation(level, solver, field_id, cdm=cdm, layout=layout)
        if w is not None:
            parts.append(w)
    if not parts:
        return None
    if len(parts) == 1:
        Q, Qt, nr, _lay = parts[0]
        return Q, Qt, nr
    Q = parts[0][0].matMult(parts[1][0])
    Qt = Q.transpose(PETSc.Mat())
    nr = np.array(sorted(set(parts[0][2].tolist()) | set(parts[1][2].tolist())),
                  dtype=np.int64)
    return Q, Qt, nr


# --------------------------------------------------------------------------- #
#  Static / add-on levels
# --------------------------------------------------------------------------- #
def addon_level_indices(tail):
    """Indices of the ADD-ON levels of a mesh-owned tail — every level that is
    not a static background wrap (``_mg_static``). These are the levels a fault
    is carried through; the background is never cut."""
    return [k for k, m in enumerate(tail) if not getattr(m, "_mg_static", False)]


# A coarse unknown whose transfer column carries less than this fraction of
# the level's largest column (squared 2-norms) is UNREACHED: its Galerkin
# diagonal is that fraction of the fine operator's, and SOR divides by it.
# Exactly-zero columns are the orphans of a split tip; the near-zero ones
# arise under rotation, where a coarse wall node on an axis has its normal
# Cartesian component reached only through fine tangential rows weighted by
# the sine of a vanishing angle (measured 1e-10 on an annulus, NaN at the
# first sweep).
UNREACHED_REL = 1e-8


def _reached_columns(T, rel=None):
    rel = UNREACHED_REL if rel is None else rel
    n2 = np.asarray(_column_norms_sq(T).array)
    big = T.comm.tompi4py().allreduce(float(n2.max()) if n2.size else 0.0, op=MPI.MAX)
    return n2 > rel * big


def restrict_unreached(Ps, comm):
    """The prolongation list with every coarse unknown no fine unknown reaches
    dropped, fine to coarse, each level's rows restricted to the set the level
    above kept. The unrotated twin of :func:`rotated_split_transfers`, for the
    standard (non-rotated) install of a hierarchy with split levels. "Reaches"
    is relative (``UNREACHED_REL``), so a vanishing column counts as none."""
    nlev = len(Ps) + 1
    kept = [None] * nlev
    out = [None] * (nlev - 1)
    for l in range(nlev - 2, -1, -1):
        T = Ps[l]
        if kept[l + 1] is not None:
            T = T.createSubMatrix(
                _owned_is(kept[l + 1], comm),
                _owned_is(np.arange(*T.getOwnershipRangeColumn()), comm))
        reached = _reached_columns(T)
        cols = np.arange(*T.getOwnershipRangeColumn())
        if comm.tompi4py().allreduce(int((~reached).sum())):
            T = T.createSubMatrix(
                _owned_is(np.arange(*T.getOwnershipRange()), comm),
                _owned_is(cols[reached], comm))
            kept[l] = cols[reached]
        out[l] = T
    return out


# --------------------------------------------------------------------------- #
#  Composition for the rotated solve
# --------------------------------------------------------------------------- #
def _owned_is(indices, comm):
    return PETSc.IS().createGeneral(np.asarray(indices, dtype=PETSc.IntType),
                                    comm=comm)


def _column_norms_sq(P):
    """Squared 2-norm of every column of ``P``, in the column layout (a Vec)."""
    v = P.createVecRight()
    v.set(0.0)
    _indptr, indices, data = P.getValuesCSR()
    if indices.size:
        v.setValues(indices.astype(PETSc.IntType), data * data,
                    addv=PETSc.InsertMode.ADD_VALUES)
    v.assemblyBegin()
    v.assemblyEnd()
    return v


def rotated_split_transfers(solver, hierarchy, Ps, Q, normal_rows):
    r"""The rotated prolongation list for a hierarchy with split levels.

    Level by level from the finest down: :math:`\hat P_l = Q_{l+1} P_l Q_l^T`
    with the finest's opening rows zeroed (its size is the operator's) and every
    coarse level restricted to its admissible unknowns — the level's own opening
    rows removed, then the unknowns no fine unknown reaches dropped. Welded levels
    (the background) have no rotation; their columns are still restricted to the
    reached set. Returns the list ``[P_hat_0, ..., P_hat_{n-2}]`` and records the
    kept index sets on the hierarchy (``hierarchy.level_kept``)."""
    from underworld3.utilities import custom_mg

    levels = hierarchy.level_meshes
    nlev = len(levels)
    comm = solver.dm.comm
    vel_is = solver._subdict["velocity"][0]
    vis = np.asarray(vel_is.getIndices())
    Qv = Q.createSubMatrix(vel_is, vel_is)
    rstart = Qv.getOwnershipRange()[0]
    g2blk = {int(g): rstart + k for k, g in enumerate(vis)}
    fine_normal = sorted({g2blk[g] for g in normal_rows if g in g2blk})

    rot = [None] * nlev
    for l in range(nlev - 1):
        rot[l] = level_rotation(levels[l], solver, hierarchy.field_id)

    kept = [None] * nlev              # owned kept global indices per coarse level
    out = [None] * (nlev - 1)
    for l in range(nlev - 2, -1, -1):
        T = Ps[l]
        if l + 1 == nlev - 1:
            T = Qv.matMult(T)
        elif rot[l + 1] is not None:
            T = rot[l + 1][0].matMult(T)
        if rot[l] is not None:
            T = T.matMult(rot[l][1])
        if l + 1 == nlev - 1:
            T.zeroRows(fine_normal, diag=0.0)
            row_is = None
        else:
            row_is = _owned_is(kept[l + 1], comm)
        # admissible columns at level l: owned, minus the opening rows
        cs, ce = T.getOwnershipRangeColumn()
        adm = np.arange(cs, ce)
        if rot[l] is not None and rot[l][2].size:
            adm = np.setdiff1d(adm, rot[l][2])
        col_is = _owned_is(adm, comm)
        T = T.createSubMatrix(row_is, col_is) if row_is is not None \
            else T.createSubMatrix(_owned_is(np.arange(*T.getOwnershipRange()), comm),
                                   col_is)
        # drop the coarse unknowns no fine unknown reaches
        reached = _reached_columns(T)
        n_drop = int(comm.tompi4py().allreduce(int((~reached).sum())))
        if n_drop:
            adm = adm[reached]
            T = T.createSubMatrix(
                _owned_is(np.arange(*T.getOwnershipRange()), comm),
                _owned_is(np.arange(*T.getOwnershipRangeColumn())[reached], comm))
        kept[l] = adm
        out[l] = T
    hierarchy.level_kept = kept
    # The restriction renumbers every coarse level, so any FAC patch rows the
    # builder recorded for them (keyed to the unrestricted layouts) no longer
    # apply; the finest's layout is untouched and its patch rows stay.
    rows = getattr(hierarchy, "level_patch_rows", None)
    hierarchy.level_patch_rows = ([None] * (nlev - 1) + [rows[-1]]
                                  if rows is not None else [None] * nlev)
    return out
