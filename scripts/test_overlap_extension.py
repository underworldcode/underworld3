"""Test whether DMPlex.distributeOverlap(k) actually extends the
overlap and fixes the vertex-degree deficit observed at rank cuts.

Sequence:
  1. Build mesh as normal (UW3 default = overlap 0)
  2. Print owned-vertex degree summary
  3. Call mesh.dm.distributeOverlap(k) for k=1,2
  4. Re-check degrees
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


def summary(label, dm):
    own = owned_mask(dm)
    deg = vertex_degrees(dm)
    own_deg = deg[own]
    comm = MPI.COMM_WORLD
    local_n = int(own.sum())
    local_sum = int(own_deg.sum())
    local_min = int(own_deg.min()) if local_n else 99
    local_max = int(own_deg.max()) if local_n else 0
    total_n = comm.allreduce(local_n)
    total_sum = comm.allreduce(local_sum)
    glob_min = comm.allreduce(local_min, op=MPI.MIN)
    glob_max = comm.allreduce(local_max, op=MPI.MAX)
    if comm.rank == 0:
        print(f"{label:30s}  n_owned={total_n}  mean_deg="
              f"{total_sum/total_n:.3f}  min={glob_min}  "
              f"max={glob_max}")


def main():
    comm = MPI.COMM_WORLD
    size = comm.size
    if comm.rank == 0:
        print(f"\n=== np={size} ===")

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=1.0 / 16)
    summary("default (overlap=0)", mesh.dm)

    if size > 1:
        try:
            sf1 = mesh.dm.distributeOverlap(1)
            summary("after distributeOverlap(1)", mesh.dm)
        except Exception as e:
            if comm.rank == 0:
                print(f"  distributeOverlap(1) error: {e}")

        try:
            sf2 = mesh.dm.distributeOverlap(1)
            summary("after distributeOverlap(1)+1", mesh.dm)
        except Exception as e:
            if comm.rank == 0:
                print(f"  distributeOverlap(+1) error: {e}")


if __name__ == "__main__":
    main()
