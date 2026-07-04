"""Sweep-by-sweep drift measurement: how much of the serial-vs-
parallel disagreement happens in iteration 1 vs accumulates over
multiple iterations?

Tells us whether the failure is structural (incomplete adjacency,
drift visible after one sweep) or cumulative (drift compounds over
sweeps, suggesting a stale-coord communication issue).
"""
from __future__ import annotations
import numpy as np
from mpi4py import MPI

import underworld3 as uw
from underworld3.meshing import smooth_mesh_interior, UnstructuredSimplexBox


def _owned_vertex_mask(dm):
    pStart, pEnd = dm.getDepthStratum(0)
    n = pEnd - pStart
    owned = np.ones(n, dtype=bool)
    sf = dm.getPointSF()
    if sf is None:
        return owned
    try:
        _, leaves, _ = sf.getGraph()
    except Exception:
        return owned
    if leaves is None:
        return owned
    for L in leaves:
        if pStart <= L < pEnd:
            owned[L - pStart] = False
    return owned


def _boundary_vertex_mask(mesh):
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    n = pEnd - pStart
    skip = {"All_Boundaries", "Null_Boundary"}
    mask = np.zeros(n, dtype=bool)
    for b in mesh.boundaries:
        nm = getattr(b, "name", None)
        if not nm or nm in skip:
            continue
        lab = dm.getLabel(nm)
        if lab is None:
            continue
        vIS = lab.getValueIS()
        if vIS is None:
            continue
        for v in vIS.getIndices():
            iset = lab.getStratumIS(int(v))
            if iset is None:
                continue
            for idx in iset.getIndices():
                if pStart <= idx < pEnd:
                    mask[idx - pStart] = True
    return mask


def run_with_nsweeps(n_iters):
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=1.0 / 16)

    is_bnd = _boundary_vertex_mask(mesh)
    coords = np.asarray(mesh.X.coords).copy()
    dx = 0.018 * np.sin(7.0 * np.pi * coords[:, 0]) \
        * np.cos(5.0 * np.pi * coords[:, 1])
    dy = 0.018 * np.cos(3.0 * np.pi * coords[:, 0]) \
        * np.sin(11.0 * np.pi * coords[:, 1])
    coords[~is_bnd, 0] += dx[~is_bnd]
    coords[~is_bnd, 1] += dy[~is_bnd]
    mesh._deform_mesh(coords)

    smooth_mesh_interior(mesh, n_iters=n_iters, alpha=0.5)

    final = np.asarray(mesh.X.coords).copy()
    is_owned = _owned_vertex_mask(mesh.dm)
    own = final[is_owned]
    own_rank = np.full(own.shape[0], rank, dtype=np.int32)

    g = comm.gather(own, root=0)
    gr = comm.gather(own_rank, root=0)
    if rank == 0:
        return np.vstack(g), np.concatenate(gr)
    return None, None


if __name__ == "__main__":
    import sys
    from scipy.spatial import cKDTree
    rank = MPI.COMM_WORLD.rank
    sizes = [1, 2, 4, 8]
    out = {}
    for n in sizes:
        c, r = run_with_nsweeps(n)
        if rank == 0:
            out[n] = (c, r)
    if rank == 0:
        size = MPI.COMM_WORLD.size
        np.savez(f"/tmp/winslow_nsweeps_np{size}.npz",
                 **{f"coords_{n}": out[n][0] for n in sizes},
                 **{f"rank_{n}": out[n][1] for n in sizes})
        print(f"[np={size}] dumped {len(sizes)} sweep-counts to "
              f"/tmp/winslow_nsweeps_np{size}.npz")
