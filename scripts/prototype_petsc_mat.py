"""Prototype: build the vertex-vertex adjacency as a parallel PETSc
Mat. Verify that owned-vertex rows have the SAME degree as serial.

Run with:
    pixi run -e amr-dev python                       scripts/prototype_petsc_mat.py
    pixi run -e amr-dev mpirun -n 2 python           scripts/prototype_petsc_mat.py
    pixi run -e amr-dev mpirun -n 4 python           scripts/prototype_petsc_mat.py
"""
from __future__ import annotations

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw
from underworld3.meshing import UnstructuredSimplexBox


def build_section_1dof_per_vertex(dm):
    """1 dof per vertex; 0 dof on edges/cells. Attached to a clone of
    the topological DM so we can produce a global section without
    disturbing the original."""
    chart_start, chart_end = dm.getChart()
    pStart, pEnd = dm.getDepthStratum(0)

    section = PETSc.Section().create(comm=dm.getComm())
    section.setChart(chart_start, chart_end)
    for p in range(chart_start, chart_end):
        section.setDof(p, 1 if pStart <= p < pEnd else 0)
    section.setUp()

    dm_scalar = dm.clone()
    dm_scalar.setLocalSection(section)
    return dm_scalar


def build_adjacency_mat(mesh):
    """Build the parallel vertex-vertex adjacency Mat. Each rank
    inserts entries for every locally-visible edge using GLOBAL
    indices; mat.assemble() combines cross-rank contributions so
    owned-vertex rows are complete after assembly."""
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)

    dm_scalar = build_section_1dof_per_vertex(dm)
    gsection = dm_scalar.getGlobalSection()

    def gidx(p):
        off = gsection.getOffset(p)
        return off if off >= 0 else -(off + 1)

    A = dm_scalar.createMatrix()
    A.setOption(A.Option.NEW_NONZERO_LOCATION_ERR, False)
    A.setOption(A.Option.IGNORE_OFF_PROC_ENTRIES, False)

    for e in range(eStart, eEnd):
        cone = dm.getCone(e)
        if len(cone) != 2:
            continue
        v0, v1 = cone[0], cone[1]
        if not (pStart <= v0 < pEnd and pStart <= v1 < pEnd):
            continue
        g0, g1 = gidx(v0), gidx(v1)
        A.setValues([g0], [g1], [1.0], PETSc.InsertMode.INSERT)
        A.setValues([g1], [g0], [1.0], PETSc.InsertMode.INSERT)

    A.assemble()
    return A, dm_scalar, gsection


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=1.0 / 16)

    A, dm_scalar, gsection = build_adjacency_mat(mesh)

    rstart, rend = A.getOwnershipRange()
    ones = A.createVecLeft()
    ones.set(1.0)
    deg = A.createVecLeft()
    A.mult(ones, deg)
    deg_local = np.asarray(deg.array).copy()

    local_n = rend - rstart
    total_n = comm.allreduce(local_n)
    if rank == 0:
        print(f"np={size}: global owned-vertex rows = {total_n}")
    # Per-rank degree stats over OWNED rows
    print(f"  rank {rank}: rows {rstart}..{rend-1} "
          f"deg mean={deg_local.mean():.3f} "
          f"min={int(deg_local.min())} max={int(deg_local.max())}",
          flush=True)

    # Global degree summary
    g_min = comm.allreduce(int(deg_local.min()), op=MPI.MIN)
    g_max = comm.allreduce(int(deg_local.max()), op=MPI.MAX)
    g_sum = comm.allreduce(float(deg_local.sum()))
    if rank == 0:
        print(f"np={size}: GLOBAL  mean_deg={g_sum/total_n:.3f}  "
              f"min={g_min}  max={g_max}")


if __name__ == "__main__":
    main()
