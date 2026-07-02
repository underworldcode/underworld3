"""
Probe: for each query point on the SphericalManifold, what does
EACH rank's `points_in_domain` say?

Classifies every query into one of:
    - "owned": exactly one rank claims it (the correct case)
    - "contested": more than one rank claims it (in-cell test too
      permissive at partition boundary — wrong rank can silently
      evaluate)
    - "orphan": no rank claims it (element identification truly fails;
      the migrate loop's iteration can't recover)

Run:
    mpirun -n 2 python probe_ownership_resolution.py
"""

import numpy as np
from mpi4py import MPI

import underworld3 as uw


def main():
    mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.3)
    rank = uw.mpi.rank
    size = uw.mpi.size

    # Build the same query set as probe_real_uw3_path.py — but identical
    # across ranks this time, so we get a single shared view of which
    # points are owned / contested / orphaned. (We use a single seed
    # so all ranks see the same query batch.)
    rng = np.random.default_rng(11)
    n = 60
    theta = rng.uniform(0.05, np.pi - 0.05, size=n)
    phi = rng.uniform(0, 2 * np.pi, size=n)
    query = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])

    # Ask THIS rank whether each query is in its local domain.
    in_my_domain = mesh.points_in_domain(query)
    my_owned = np.asarray(in_my_domain, dtype=np.int8)

    # Allreduce sum across ranks: 0 = orphan, 1 = unique owner, ≥2 = contested.
    claim_count = np.zeros_like(my_owned, dtype=np.int32)
    MPI.COMM_WORLD.Allreduce(my_owned.astype(np.int32), claim_count, op=MPI.SUM)

    # Per-rank claim mask, so we can identify which rank claims contested
    # points.
    all_claims = np.zeros((size, n), dtype=np.int32)
    MPI.COMM_WORLD.Allgather(my_owned.astype(np.int32), all_claims)

    if rank == 0:
        n_owned = int((claim_count == 1).sum())
        n_contested = int((claim_count > 1).sum())
        n_orphan = int((claim_count == 0).sum())
        print(f"\n=== Ownership classification of {n} surface queries on "
              f"{size} ranks ===")
        print(f"  unique owner:   {n_owned}/{n}")
        print(f"  contested (>1): {n_contested}/{n}")
        print(f"  orphan  (==0):  {n_orphan}/{n}")
        print()

        if n_orphan > 0:
            print(f"Orphan queries (NO rank claims — element ID failure):")
            for i in np.where(claim_count == 0)[0]:
                print(f"  idx={i}  coord={query[i]}  r={np.linalg.norm(query[i]):.4f}")
            print()

        if n_contested > 0:
            print(f"Contested queries (multiple ranks claim — receiver "
                  f"silently extrapolates if wrong rank wins):")
            for i in np.where(claim_count > 1)[0][:8]:  # show up to 8
                claimers = np.where(all_claims[:, i] > 0)[0]
                print(
                    f"  idx={i}  coord={query[i]}  claimed by ranks "
                    f"{claimers.tolist()}"
                )
            print()


if __name__ == "__main__":
    main()
