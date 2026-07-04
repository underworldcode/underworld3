"""Run smooth_mesh_interior on a perturbed box; dump final OWNED
vertex coords (gathered to rank 0) so we can overlay serial vs
parallel runs in a single plot.

Usage::

    pixi run python scripts/smoothing_overlay_check.py        # np=1
    mpirun -n 2 pixi run python scripts/smoothing_overlay_check.py
    mpirun -n 4 pixi run python scripts/smoothing_overlay_check.py
    pixi run python scripts/smoothing_overlay_plot.py
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


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=1.0 / 16,
    )

    is_bnd = _boundary_vertex_mask(mesh)
    is_owned = _owned_vertex_mask(mesh.dm)

    coords = np.asarray(mesh.X.coords).copy()
    # Deterministic position-only perturbation — identical between
    # serial and parallel because it's a function of (x,y) alone.
    dx = 0.018 * np.sin(7.0 * np.pi * coords[:, 0]) \
        * np.cos(5.0 * np.pi * coords[:, 1])
    dy = 0.018 * np.cos(3.0 * np.pi * coords[:, 0]) \
        * np.sin(11.0 * np.pi * coords[:, 1])
    coords[~is_bnd, 0] += dx[~is_bnd]
    coords[~is_bnd, 1] += dy[~is_bnd]
    mesh._deform_mesh(coords)

    # Capture POST-perturbation, PRE-smoothing owned coords.
    pre = np.asarray(mesh.X.coords).copy()
    is_owned_pre = _owned_vertex_mask(mesh.dm)
    own_pre = pre[is_owned_pre]

    smooth_mesh_interior(mesh, n_iters=8, alpha=0.5)

    final = np.asarray(mesh.X.coords).copy()
    # Also recompute boundary mask after deformation (labels persist).
    is_bnd_after = _boundary_vertex_mask(mesh)
    is_owned_after = _owned_vertex_mask(mesh.dm)

    own_final = final[is_owned_after]
    own_bnd = is_bnd_after[is_owned_after]
    own_rank = np.full(own_final.shape[0], rank, dtype=np.int32)

    gathered_pre = comm.gather(own_pre, root=0)
    gathered_coords = comm.gather(own_final, root=0)
    gathered_bnd = comm.gather(own_bnd, root=0)
    gathered_rank = comm.gather(own_rank, root=0)

    if rank == 0:
        all_pre = np.vstack(gathered_pre)
        all_coords = np.vstack(gathered_coords)
        all_bnd = np.concatenate(gathered_bnd)
        all_rank = np.concatenate(gathered_rank)
        out = f"/tmp/winslow_overlay_np{size}.npz"
        np.savez(
            out,
            pre=all_pre,
            coords=all_coords,
            is_bnd=all_bnd,
            rank=all_rank,
        )
        print(f"[np={size}] gathered {all_coords.shape[0]} owned "
              f"vertices -> {out}")


if __name__ == "__main__":
    main()
