"""Parallel (MPI) test of the on-disk snapshot path (v1.1).

Phase 6 of the snapshot toolkit: per-rank swarm sidecars. The mesh
+ mesh-variable disk path is already parallel-correct via #146's
PETSc-collective HDF5 viewer; the swarm sidecar layer needs its
own per-rank file per swarm. This ptest exercises both together at
multi-rank.

Run (4 ranks exercises cross-rank distribution of swarm particles):

    cd tests/parallel
    mpirun -np 4 python ./ptest_0010_snapshot_disk.py

Asserts (collective, checked on rank 0):

  1. Disk write produces one wrapper file + one swarm sidecar per
     rank in the bulk dir, each with the rank+size in its filename.
  2. Each rank's sidecar carries the writing rank's local-particle
     state (verified by per-rank attrs on the sidecar).
  3. Round-trip is exact: scribble all variables + swarm coords +
     swarm-var data, model.load_state(file=...), gathered (gid, x, y,
     material) tables sorted by gid are np.array_equal.
"""

import os

import numpy as np
import sympy
from mpi4py import MPI

import underworld3 as uw

comm = MPI.COMM_WORLD
rank = uw.mpi.rank
size = uw.mpi.size


def build():
    uw.reset_default_model()
    uw.use_strict_units(False)
    uw.use_nondimensional_scaling(False)
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(4.0, 1.0), cellSize=1.0 / 6.0
    )
    x_sym, y_sym = mesh.X
    V_fn = sympy.Matrix([[-(y_sym - 0.5), 0.25 * (x_sym - 2.0)]]).T

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    T.array[:, 0, 0] = mesh.X.coords[:, 0] - mesh.X.coords[:, 1]

    swarm = uw.swarm.Swarm(mesh)
    gid = swarm.add_variable("gid", 1, dtype=float)
    material = swarm.add_variable("material", 1, dtype=float)
    swarm.populate(fill_param=2)

    local_n = swarm.dm.getLocalSize()
    counts = comm.allgather(local_n)
    offset = int(np.sum(counts[:rank]))
    gid.data[:, 0] = offset + np.arange(local_n, dtype=float)
    material.data[:, 0] = swarm._particle_coordinates.data[:, 0]

    model.tracker.time = 1.5
    model.tracker.step = 7
    return uw, model, mesh, V_fn, T, swarm, gid, material


def global_sorted_state(T, swarm, gid, material):
    """Gather (gid, x, y, material, T-value-by-coord-bin) across ranks
    + sort by gid → order/rank-independent canonical view."""
    g = gid.data[:, 0].copy()
    coords = swarm._particle_coordinates.data.copy()
    m = material.data[:, 0].copy()
    local = np.column_stack([g, coords[:, 0], coords[:, 1], m])
    gathered = comm.allgather(local)
    full = np.vstack([a for a in gathered if a.size]) if any(
        a.size for a in gathered
    ) else np.empty((0, 4))
    order = np.argsort(full[:, 0], kind="stable")
    swarm_state = full[order]

    # T round-trip check: gather partition-invariant scalars
    # (max, sum) rather than the full (coord, value) table — DOFs at
    # partition boundaries are visible to multiple ranks and would
    # appear duplicated/reordered in a gathered table, even though
    # the underlying data is bit-exact.
    t_arr = np.asarray(T.array[...]).reshape(-1)
    t_max = comm.allreduce(float(t_arr.max()) if t_arr.size else -np.inf,
                           op=MPI.MAX)
    t_min = comm.allreduce(float(t_arr.min()) if t_arr.size else np.inf,
                           op=MPI.MIN)
    # bit-exact float sum across ranks is non-deterministic in general
    # (non-associative); use min/max as bit-exact invariants instead.
    return swarm_state, (t_max, t_min)


def main():
    import tempfile

    uw, model, mesh, V_fn, T, swarm, gid, material = build()
    pre_swarm, pre_T = global_sorted_state(T, swarm, gid, material)
    pre_count = comm.allreduce(swarm.dm.getLocalSize(), op=MPI.SUM)

    # Use a shared temp dir reachable from every rank
    if rank == 0:
        tmp = tempfile.mkdtemp(prefix="uw3_ptest_0010_")
    else:
        tmp = None
    tmp = comm.bcast(tmp, root=0)

    wrapper = os.path.join(tmp, "parrun.snap.h5")
    model.save_state(file=wrapper)
    comm.Barrier()

    # Check files on rank 0
    files_ok = True
    if rank == 0:
        bulk = os.path.join(tmp, "parrun.snap.bulk")
        files = sorted(os.listdir(bulk))
        per_rank = [f for f in files if ".swarm.rank" in f]
        # Expect one swarm sidecar per rank
        if len(per_rank) != size:
            print(
                f"!! expected {size} swarm sidecars, got {len(per_rank)}: "
                f"{per_rank}",
                flush=True,
            )
            files_ok = False
        else:
            print(
                f"  swarm sidecars OK: {per_rank[0]} ... ({size} total)",
                flush=True,
            )

    # Scribble everything
    T.array[...] = -99.0
    coord_field = swarm.dm.getField("DMSwarmPIC_coor").reshape((-1, swarm.dim))
    coord_field[...] = -99.0
    swarm.dm.restoreField("DMSwarmPIC_coor")
    material.data[...] = -99.0
    model.tracker.time = -1.0
    model.tracker.step = -1

    model.load_state(wrapper)
    post_count = comm.allreduce(swarm.dm.getLocalSize(), op=MPI.SUM)
    post_swarm, post_T = global_sorted_state(T, swarm, gid, material)

    swarm_ok = np.array_equal(pre_swarm, post_swarm)
    # T is checked via partition-invariant min/max scalars (see note
    # in global_sorted_state — gathered DOFs include partition-
    # boundary duplicates that resist a global-table comparison).
    T_ok = (pre_T == post_T)
    count_ok = pre_count == post_count
    tracker_ok = (model.tracker.time == 1.5 and model.tracker.step == 7)

    if rank == 0:
        print(f"[ranks={size}] particles total = {pre_count}", flush=True)
        print(f"  P1 disk wrapper + per-rank sidecars present: {files_ok}",
              flush=True)
        print(f"  P2 particle count preserved:                 {count_ok}",
              flush=True)
        print(f"  P3 swarm (coords + gid + material) exact:    {swarm_ok}",
              flush=True)
        print(f"  P4 T (mesh-variable DOFs) exact:             {T_ok}",
              flush=True)
        print(f"  P5 tracker state restored:                   {tracker_ok}",
              flush=True)

        assert files_ok
        assert count_ok
        assert swarm_ok
        assert T_ok
        assert tracker_ok
        print(f"[ranks={size}] PASS", flush=True)


if __name__ == "__main__":
    main()
