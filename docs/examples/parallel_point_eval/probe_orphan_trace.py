"""
Trace the lifecycle of orphan points: for each orphan from the
ownership probe, show on EACH rank:
  - closest local cell + its owner status
  - distance from query to that cell's centroid
  - geometric in-cell test result (before owner filter)
  - cell-walk fallback: how many of the top-50 cells contain the
    query (geometric only), and how many of those are owned

This tells us whether the orphan is "no rank's geometric test claims
it" (in-cell test broken) or "owner filter is rejecting a cell that
should be the owned containing cell" (filter wrong) or "owned cell is
not even in the top-50 nearest" (kdtree priorities wrong with overlap).

Run:
    mpirun -n 2 python probe_orphan_trace.py
"""

import numpy as np
from mpi4py import MPI

import underworld3 as uw


def main():
    mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.3)
    rank = uw.mpi.rank
    size = uw.mpi.size

    # Same query batch as probe_ownership_resolution.py — single seed,
    # shared across ranks.
    rng = np.random.default_rng(11)
    n = 60
    theta = rng.uniform(0.05, np.pi - 0.05, size=n)
    phi = rng.uniform(0, 2 * np.pi, size=n)
    query = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])

    # Trigger build of all the indexing structures.
    mesh._mark_local_boundary_faces_inside_and_out()
    mesh._build_kd_tree_index()
    owned = mesh._get_owned_cells_mask()

    # Pre-compute closest local cell per query (rank-local).
    dist, closest_cp = mesh._index.query(query, k=1, sqr_dists=False)
    closest_cell = mesh._indexMap[closest_cp]

    # Top-50 nearest cells (centroids) per query.
    n_top = min(50, mesh._centroids.shape[0])
    _, top_cells = mesh._centroid_index.query(query, k=n_top, sqr_dists=False)

    # Geometric in-cell test result for each query against its closest
    # cell — without the owner filter.
    from underworld3.discretisation.discretisation_mesh import Mesh
    # Re-run the raw geometric test by temporarily inverting the filter
    # — easiest: read result, then check owner separately.
    raw_inside = mesh._test_if_points_in_cells_internal(query, closest_cell)
    # raw_inside already has filter applied; but we want to know if the
    # geometric half-space test would have accepted it. Recompute by
    # forcing owner mask True for this call.
    saved_mask = mesh._owned_cells_mask_cache["mask"].copy()
    mesh._owned_cells_mask_cache["mask"] = np.ones_like(saved_mask, dtype=bool)
    geom_inside = mesh._test_if_points_in_cells_internal(query, closest_cell)
    mesh._owned_cells_mask_cache["mask"] = saved_mask

    # For each query, walk top-50 cells and count how many pass the
    # raw geometric test, and how many of THOSE are owned.
    mesh._owned_cells_mask_cache["mask"] = np.ones_like(saved_mask, dtype=bool)
    geom_hits = np.zeros(n, dtype=int)
    geom_owned_hits = np.zeros(n, dtype=int)
    for j in range(n_top):
        candidate = top_cells[:, j]
        ok = mesh._test_if_points_in_cells_internal(query, candidate)
        geom_hits += ok.astype(int)
        owned_ok = ok & saved_mask[candidate]
        geom_owned_hits += owned_ok.astype(int)
    mesh._owned_cells_mask_cache["mask"] = saved_mask

    # Print per-rank lines, sorted so the 60 queries are visible.
    in_my_domain = mesh.points_in_domain(query).astype(int)
    claim_count = np.zeros_like(in_my_domain)
    MPI.COMM_WORLD.Allreduce(in_my_domain, claim_count, op=MPI.SUM)
    orphans = np.where(claim_count == 0)[0]

    if rank == 0:
        print(f"\n=== Orphan trace ({len(orphans)} orphans) ===\n")
        print(f"{'idx':>4}  {'rank':>4}  {'cell':>5}  {'owned':>5}  "
              f"{'geom':>4}  {'geom_hits':>9}  {'owned_hits':>10}")

    for o in orphans:
        for r in range(size):
            uw.mpi.barrier()
            if rank == r:
                c = int(closest_cell[o])
                print(
                    f"{o:>4}  {r:>4}  {c:>5}  {bool(owned[c])!s:>5}  "
                    f"{bool(geom_inside[o])!s:>4}  {geom_hits[o]:>9}  "
                    f"{geom_owned_hits[o]:>10}",
                    flush=True,
                )
        uw.mpi.barrier()


if __name__ == "__main__":
    main()
