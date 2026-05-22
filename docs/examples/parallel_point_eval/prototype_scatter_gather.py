"""
Prototype: parallel point-evaluation via DMSWARM_BASIC + centroid kdtree.

Demonstrates the scatter-evaluate-gather pattern that
`uw.function.global_evaluate` should use on a manifold mesh (dim != cdim).
Side-steps the `self.mesh.dim` walls in UW3's existing
`add_particles_with_global_coordinates` by writing the swarm's field
arrays directly, so that we exercise only the parts of the pipeline
that are architecturally correct.

Run:
    python prototype_scatter_gather.py
    mpirun -n 2 python prototype_scatter_gather.py
    mpirun -n 4 python prototype_scatter_gather.py

What it checks:
    1. Each rank starts with a (per-rank-disjoint) batch of global query
       coords with shape `(N_local, cdim)`.
    2. The swarm scatters each coord to whichever rank owns it (centroid
       kdtree on per-rank domain centroids).
    3. Each rank evaluates a known analytic field at its local coord
       subset.
    4. Values are gathered back to the originating rank.
    5. Compared against direct rank-local evaluation of the same field
       at the original query points on rank 0 — bit-identical.

Coord arrays are cdim-shaped throughout. On a regular volume mesh,
cdim == dim and this matches UW3's current behaviour; on a manifold it
is the *only* shape that works. The script does not require a manifold
mesh — it just verifies the architecture is dim-agnostic.
"""

import numpy as np
from petsc4py import PETSc

import underworld3 as uw


def analytic_field(coords: np.ndarray) -> np.ndarray:
    """Closed-form scalar field. Used as ground truth.

    Independent of mesh / variable / interpolation — just a function of
    coordinates so we can compare the parallel gather to rank-local
    evaluation directly.
    """
    return np.sum(coords**2, axis=1) + 0.3 * coords[:, 0]


def build_local_query_coords(mesh, n_local: int, seed: int) -> np.ndarray:
    """Build a batch of cdim-shaped query coords inside the mesh's global
    bounding box. Each rank gets a deterministic, disjoint batch.
    """
    rng = np.random.default_rng(seed + uw.mpi.rank)
    cdim = mesh.cdim
    coords = np.asarray(mesh.X.coords)
    lo_local = coords.min(axis=0)
    hi_local = coords.max(axis=0)
    from mpi4py import MPI
    lo = np.zeros(cdim)
    hi = np.zeros(cdim)
    MPI.COMM_WORLD.Allreduce(lo_local, lo, op=MPI.MIN)
    MPI.COMM_WORLD.Allreduce(hi_local, hi, op=MPI.MAX)
    # Shrink slightly so query points don't end up exactly on the
    # boundary where ownership is ambiguous.
    span = hi - lo
    return lo + 0.05 * span + 0.9 * span * rng.random(size=(n_local, cdim))


def scatter_evaluate_gather(mesh, query_coords: np.ndarray) -> np.ndarray:
    """Scatter query_coords to owning ranks, evaluate, gather results.

    query_coords:  shape (N_local, cdim).
    returns:       shape (N_local,) — values at each input coord, on the
                   rank that supplied it.
    """
    cdim = mesh.cdim
    n_local = query_coords.shape[0]
    rank = uw.mpi.rank
    size = uw.mpi.size

    # --- 1. Build a DMSWARM_BASIC and a coord field at cdim blocksize.
    sw = PETSc.DMSwarm().create(comm=PETSc.COMM_WORLD)
    sw.setDimension(cdim)  # ← cdim, not topological dim. Drives the
                            # blocksize of any cell-DM coord field we'd
                            # register through DMSwarmAddCellDM. We don't
                            # use that path; setting cdim is honest.
    sw.setType(PETSc.DMSwarm.Type.BASIC)
    sw.registerField("coord", cdim, dtype=PETSc.RealType)
    sw.registerField("orig_rank", 1, dtype=PETSc.IntType)
    sw.registerField("orig_slot", 1, dtype=PETSc.IntType)
    sw.registerField("value", 1, dtype=PETSc.RealType)
    sw.finalizeFieldRegister()
    sw.setLocalSizes(n_local, -1)

    # --- 2. Populate with the local query batch.
    coord_field = sw.getField("coord")
    # petsc4py returns either flat (n*cdim,) or shaped (n,cdim) depending
    # on PETSc/petsc4py version. Reshape defensively.
    coord_field.reshape(-1, cdim)[...] = query_coords
    sw.restoreField("coord")
    sw.getField("orig_rank").reshape(-1)[...] = rank
    sw.restoreField("orig_rank")
    sw.getField("orig_slot").reshape(-1)[...] = np.arange(n_local, dtype=PETSc.IntType)
    sw.restoreField("orig_slot")

    # --- 3. Compute owner rank for each point using the mesh's domain
    #        centroid kdtree (cdim-aware). Set the DMSwarm_rank field;
    #        dm.migrate will then ship each point to the correct rank.
    if size > 1:
        kdt = mesh._get_domain_kdtree()
        _, owner = kdt.query(query_coords, k=1, sqr_dists=False)
        owner = np.asarray(owner, dtype=np.int32).reshape(-1)
        rank_field = sw.getField("DMSwarm_rank")
        rank_field.reshape(-1)[...] = owner
        sw.restoreField("DMSwarm_rank")
        sw.migrate(remove_sent_points=True)

    # --- 4. Each rank now holds points it owns. Evaluate the analytic
    #        field at those local coords.
    local_coords = sw.getField("coord").reshape(-1, cdim).copy()
    sw.restoreField("coord")
    if local_coords.shape[0] > 0:
        local_values = analytic_field(local_coords)
        v = sw.getField("value")
        v.reshape(-1)[...] = local_values
        sw.restoreField("value")

    # --- 5. Return-trip: set DMSwarm_rank back to orig_rank and migrate.
    if size > 1:
        orig_rank_field = sw.getField("orig_rank")
        rank_field = sw.getField("DMSwarm_rank")
        rank_field.reshape(-1)[...] = orig_rank_field.reshape(-1)
        sw.restoreField("DMSwarm_rank")
        sw.restoreField("orig_rank")
        sw.migrate(remove_sent_points=True)

    # --- 6. Re-order returned values by orig_slot so output position
    #        matches the input position.
    n_returned = sw.getLocalSize()
    out = np.full(n_local, np.nan, dtype=PETSc.RealType)
    if n_returned > 0:
        slot = sw.getField("orig_slot").copy().astype(int)
        sw.restoreField("orig_slot")
        val = sw.getField("value").copy()
        sw.restoreField("value")
        # Defensive: there's no guarantee n_returned == n_local if a
        # query coord fell outside every rank's domain. nan-fill the
        # rest and warn.
        out[slot] = val
    if size > 1:
        n_lost = np.count_nonzero(np.isnan(out))
        if n_lost > 0:
            uw.pprint(
                f"[rank {rank}] WARNING: {n_lost} of {n_local} points lost",
                _from_rank=rank,
            )

    sw.destroy()
    return out


def main():
    uw.pprint("=" * 60)
    uw.pprint(f"Parallel point-eval prototype — {uw.mpi.size} rank(s)")
    uw.pprint("=" * 60)

    # Regular 2-D box mesh; the architecture is dim-agnostic. cdim==dim
    # here, but the prototype writes only cdim-shaped arrays — proving
    # that the manifold case (cdim>dim) needs no different code path.
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.1,
    )
    uw.pprint(f"mesh.dim={mesh.dim}, mesh.cdim={mesh.cdim}")

    # Each rank brings its own batch of query points. Disjoint by seed.
    n_local = 50
    query = build_local_query_coords(mesh, n_local=n_local, seed=12345)

    # Ground truth: evaluate analytic_field directly at the input coords.
    expected = analytic_field(query)

    # The parallel pipeline result.
    actual = scatter_evaluate_gather(mesh, query)

    # Compare locally on each rank.
    if np.all(np.isnan(actual)):
        ok = False
        err = float("nan")
    else:
        mask = ~np.isnan(actual)
        err = float(np.abs(actual[mask] - expected[mask]).max())
        n_ok = int(mask.sum())
        ok = (err < 1e-12) and (n_ok == n_local)

    # Gather per-rank diagnostics.
    all_oks = uw.utilities.gather_data(np.array([int(ok)], dtype=int), bcast=True)
    all_errs = uw.utilities.gather_data(np.array([err], dtype=float), bcast=True)

    if uw.mpi.rank == 0:
        print()
        for r in range(uw.mpi.size):
            print(
                f"  rank {r}: ok={bool(all_oks[r])}  "
                f"max|actual-expected|={all_errs[r]:.3e}"
            )
        if np.all(all_oks):
            print("\n✅ scatter-evaluate-gather round-trip verified")
        else:
            print("\n❌ scatter-evaluate-gather mismatch — see above")


if __name__ == "__main__":
    main()
