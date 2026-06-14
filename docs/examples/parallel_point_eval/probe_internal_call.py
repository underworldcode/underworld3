"""
Minimal direct test: call `_get_closest_local_cells_internal` with the
orphan coords. Print result and compare with manually-computed expected.
"""

import numpy as np

import underworld3 as uw


def main():
    mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.3)
    rank = uw.mpi.rank

    # The orphan coords from the previous probe.
    orphan_coords = np.array([
        [ 0.5311747,  -0.84725947,  0.00219645],  # idx 1
        [-0.93145017,  0.20021023, -0.30383622],  # idx 2
        [ 0.31294064, -0.86690124,  0.38800825],  # idx 10
    ])

    # Direct internal call.
    result = mesh._get_closest_local_cells_internal(orphan_coords)

    # Public API.
    in_domain = mesh.points_in_domain(orphan_coords)

    # Compare with what the kdtree alone gives us, and the owner status.
    mesh._build_kd_tree_index()
    owned = mesh._get_owned_cells_mask()
    _, cp = mesh._index.query(orphan_coords, k=1, sqr_dists=False)
    cells_from_kdtree = mesh._indexMap[cp]

    # Also test the FULL 60-point batch via points_in_domain to see if
    # the batch result differs from the small-batch result.
    rng = np.random.default_rng(11)
    n = 60
    theta = rng.uniform(0.05, np.pi - 0.05, size=n)
    phi = rng.uniform(0, 2 * np.pi, size=n)
    full_query = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    full_in_domain = mesh.points_in_domain(full_query)
    # Find positions of orphan_coords in the full query
    orphan_indices_in_full = []
    for oc in orphan_coords:
        # Find the row in full_query that matches oc
        dists = np.linalg.norm(full_query - oc, axis=1)
        orphan_indices_in_full.append(int(np.argmin(dists)))
    full_results = [full_in_domain[i] for i in orphan_indices_in_full]

    for r in range(uw.mpi.size):
        uw.mpi.barrier()
        if rank == r:
            print(f"\n=== Rank {r} ===")
            for i, (rc, kc, c, ind, find) in enumerate(
                zip(result, cells_from_kdtree, orphan_coords, in_domain, full_results)
            ):
                print(
                    f"orphan {i}: "
                    f"_internal->{rc} (owned={bool(owned[rc]) if rc >= 0 else 'N/A'})  "
                    f"kdtree->{kc} (owned={bool(owned[kc])})  "
                    f"in_domain(small_batch)={bool(ind)}  "
                    f"in_domain(full_batch)={bool(find)}"
                )


if __name__ == "__main__":
    main()
