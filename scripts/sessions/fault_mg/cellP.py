"""Cell-located, side-aware Lagrange prolongation between (possibly split, non-nested) meshes.

For each fine node: one fine cell containing it -> that cell's CENTROID located in the coarse
mesh (a centroid is never on a fault, so the coarse cell is on the node's own side) -> the coarse
cell's interpolant evaluated at the node by the dual-basis identity W = B M^-1 (as in
custom_mg.nested_refine_pair_prolongation). Exact for nested pairs; for non-nested pairs it
reproduces the coarse polynomial of the located cell (global polynomials of the degree exactly).
"""
import itertools
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import os
CLAMP = os.environ.get("CELLP_CLAMP", "1") == "1"


def _cell_vertices(mesh):
    dm = mesh.dm
    dim = dm.getDimension()
    cS, cE = dm.getHeightStratum(0)
    vS, vE = dm.getDepthStratum(0)
    vxy = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dm.getCoordinateDim())
    cv = np.empty((cE - cS, dim + 1), dtype=np.int64)
    for c in range(cS, cE):
        cv[c - cS] = [q - vS for q in dm.getTransitiveClosure(c)[0] if vS <= q < vE]
    return vxy, cv


def _barycentric(P, verts):
    """barycentric coords of points P (n,d) in simplices verts (n,d+1,d)."""
    A = np.transpose(verts[:, 1:] - verts[:, :1], (0, 2, 1))
    lam = np.linalg.solve(A, (P - verts[:, 0])[..., None])[..., 0]
    return np.concatenate([1 - lam.sum(axis=1, keepdims=True), lam], axis=1)


def locate_cells(points, mesh, k=16):
    vxy, cv = _cell_vertices(mesh)
    cent = vxy[cv].mean(axis=1)
    tree = cKDTree(cent)
    _, cand = tree.query(points, k=min(k, len(cent)))
    best = np.full(len(points), -1)
    bestmin = np.full(len(points), -np.inf)
    for j in range(cand.shape[1]):
        c = cand[:, j]
        b = _barycentric(points, vxy[cv[c]])
        m = b.min(axis=1)
        better = m > bestmin
        best[better] = c[better]
        bestmin[better] = m[better]
    return best, bestmin


def cell_located_prolongation(coarse, fine, degree=2, continuous=True):
    cn = coarse._cell_node_indices(degree, continuous)
    fn = fine._cell_node_indices(degree, continuous)
    Xc = np.asarray(coarse._get_coords_for_basis(degree, continuous))
    Xf = np.asarray(fine._get_coords_for_basis(degree, continuous))
    dim = coarse.dm.getDimension()
    vxy_c, cv_c = _cell_vertices(coarse)
    vxy_f, cv_f = _cell_vertices(fine)
    owner = np.full(Xf.shape[0], -1, dtype=np.int64)
    for kk in range(fn.shape[0]):
        owner[fn[kk]] = kk
    assert (owner >= 0).all()
    fcent = vxy_f[cv_f].mean(axis=1)
    parent_cell, inside = locate_cells(fcent, coarse)
    parent_of = parent_cell[owner]
    E = np.asarray([e for e in itertools.product(range(degree + 1), repeat=dim) if sum(e) <= degree])
    vander = lambda L: np.prod(L[:, None, :] ** E[None, :, :], axis=2)
    order = np.argsort(parent_of, kind="stable")
    runs = np.flatnonzero(np.diff(parent_of[order])) + 1
    rows, cols, vals = [], [], []
    for grp in np.split(order, runs):
        c = int(parent_of[grp[0]])
        verts = vxy_c[cv_c[c]]
        Ainv = np.linalg.inv((verts[1:] - verts[0]).T)
        Lc = (Xc[cn[c]] - verts[0]) @ Ainv.T
        Lf = (Xf[grp] - verts[0]) @ Ainv.T
        if CLAMP:
            # evaluate at the nearest point of the located cell: a fine node a little
            # outside it (non-nested levels near a pulled tip) is not extrapolated, so
            # the weights stay bounded (the Lagrange basis inside the simplex)
            b = np.column_stack([1 - Lf.sum(axis=1), Lf])
            if (b < 0).any():
                b = np.clip(b, 0, None); b /= b.sum(axis=1, keepdims=True)
                Lf = b[:, 1:]
        W = np.linalg.solve(vander(Lc).T, vander(Lf).T).T
        for ti, t in enumerate(grp):
            keep = np.flatnonzero(np.abs(W[ti]) > 1e-13)
            rows.extend([int(t)] * keep.size)
            cols.extend(int(cn[c][j]) for j in keep)
            vals.extend(float(W[ti, j]) for j in keep)
    P = sp.csr_matrix((vals, (rows, cols)), shape=(Xf.shape[0], Xc.shape[0]))
    return P, inside.min()
