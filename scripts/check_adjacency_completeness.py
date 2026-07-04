"""Diagnose: for each owned vertex, is the LOCAL edge stratum
showing every edge incident to it that exists in the global mesh?

Comparison: vertex degree (n_neighbours) from serial run vs sum of
per-vertex degree contributions from parallel run.

We don't have a direct global mesh in parallel, so instead we
* compute owned-vertex degree per rank (from local edge stratum)
* gather to rank 0
* compare to the serial degree at the same (x, y) position
"""
from __future__ import annotations
import numpy as np
from mpi4py import MPI

import underworld3 as uw
from underworld3.meshing import UnstructuredSimplexBox


def owned_mask(dm):
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


def vertex_degrees(dm):
    pStart, pEnd = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)
    n = pEnd - pStart
    deg = np.zeros(n, dtype=np.int64)
    for e in range(eStart, eEnd):
        cone = dm.getCone(e)
        if len(cone) != 2:
            continue
        v0, v1 = cone[0] - pStart, cone[1] - pStart
        if 0 <= v0 < n:
            deg[v0] += 1
        if 0 <= v1 < n:
            deg[v1] += 1
    return deg


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=1.0 / 16)
    dm = mesh.dm
    is_owned = owned_mask(dm)
    deg = vertex_degrees(dm)
    coords = np.asarray(mesh.X.coords).copy()

    own_coords = coords[is_owned]
    own_deg = deg[is_owned]

    gc = comm.gather(own_coords, root=0)
    gd = comm.gather(own_deg, root=0)
    gr = comm.gather(np.full(own_coords.shape[0], rank, dtype=np.int32),
                     root=0)

    if rank == 0:
        all_coords = np.vstack(gc)
        all_deg = np.concatenate(gd)
        all_rank = np.concatenate(gr)
        np.savez(
            f"/tmp/winslow_degree_np{size}.npz",
            coords=all_coords, deg=all_deg, rank=all_rank)
        print(f"[np={size}] dumped {all_coords.shape[0]} owned "
              f"vertices  min_deg={all_deg.min()}  "
              f"max_deg={all_deg.max()}")


if __name__ == "__main__":
    main()
