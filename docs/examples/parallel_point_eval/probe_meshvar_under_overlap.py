"""
Probe: how does a MeshVariable behave when the underlying DM has
``distributeOverlap(1)`` applied?

We need to know:
    A. Does T.data on each rank cover only owned DOFs, or owned + ghost?
    B. If we write distinct values to ghost DOFs on different ranks, do
       they STAY distinct (i.e. PETSc treats T.data as the local
       authoritative array and doesn't sync), or does PETSc clobber
       them after sync?
    C. Is there an automatic sync point on solve / evaluate, and what
       does it do?
    D. After a solve, are the ghost DOFs consistent with their owners?

Run on 2 ranks with overlap on:
    UW_OVERLAP_PROBE=1 mpirun -n 2 python probe_meshvar_under_overlap.py

And without overlap as a baseline:
    mpirun -n 2 python probe_meshvar_under_overlap.py
"""

import os
import numpy as np

import underworld3 as uw


def main():
    overlap_on = os.environ.get("UW_OVERLAP_PROBE", "0") == "1"
    uw.pprint(f"=== MeshVariable / overlap probe — overlap={overlap_on}, "
              f"size={uw.mpi.size} ===")

    # 2D volume mesh, parallel, MeshVariable. Volume mesh keeps the
    # probe simple — overlap behaviour is the same regardless of
    # manifold-ness.
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2,
    )
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    rank = uw.mpi.rank

    # --- (A) shape inspection ---------------------------------------
    cstart, cend = mesh.dm.getHeightStratum(0)
    n_local_cells = cend - cstart
    n_dof = T.coords.shape[0]
    print(
        f"[rank {rank}] cells: {n_local_cells}, T.coords.shape: {T.coords.shape}",
        flush=True,
    )

    # PointSF leaves identify ghost POINTS (vertices in the DAG).
    sf = mesh.dm.getPointSF()
    nroots, leaves, _ = sf.getGraph()
    n_leaves = 0 if leaves is None else len(leaves)
    print(f"[rank {rank}] PointSF: nroots={nroots}, n_leaves={n_leaves}",
          flush=True)

    # --- (B) divergent rank writes ----------------------------------
    # Write a distinctive value per rank — e.g. rank R writes
    # 1000*R + i  for each local DOF i.
    sentinel_value = 1000.0 * rank + np.arange(n_dof, dtype=float)
    T.data[:, 0] = sentinel_value

    # Inspect immediately — has the write taken effect locally?
    local_min = float(np.asarray(T.data[:, 0]).min())
    local_max = float(np.asarray(T.data[:, 0]).max())
    print(
        f"[rank {rank}] after local write: T.data min={local_min:.1f} max={local_max:.1f}",
        flush=True,
    )

    # --- (C) explicit Vec-level sync via PETSc's local-to-global ---
    # If PETSc has a sync point (DMLocalToGlobal, DMGlobalToLocal),
    # invoking it should clobber the rank-1 ghost writes with the
    # rank-0 owner values (or vice versa, depending on convention).
    try:
        lvec = T.dm.getLocalVec()
        T.dm.globalToLocal(T.vec, lvec)
        local_after_sync = np.asarray(lvec.getArray()).copy()
        T.dm.restoreLocalVec(lvec)
        after_min = float(local_after_sync.min())
        after_max = float(local_after_sync.max())
        print(
            f"[rank {rank}] after globalToLocal: min={after_min:.1f} max={after_max:.1f}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[rank {rank}] globalToLocal failed: {exc}", flush=True)

    # --- (D) round-trip through solver-style infrastructure --------
    # Call uw.function.evaluate at the local DOF coords — does it
    # reflect this rank's written values, or a synced version?
    # Use a few specific coords to keep output legible.
    local_coords = np.asarray(T.coords)
    if local_coords.shape[0] > 0:
        sample_coords = local_coords[: min(3, local_coords.shape[0])]
        result = uw.function.evaluate(T.sym, sample_coords)
        result_arr = np.asarray(result).reshape(-1)
        sample_dof_values = sentinel_value[: sample_coords.shape[0]]
        print(
            f"[rank {rank}] evaluate at first 3 DOF coords: result={result_arr.tolist()}, "
            f"sentinel written there={sample_dof_values.tolist()}",
            flush=True,
        )

    uw.mpi.barrier()


if __name__ == "__main__":
    main()
