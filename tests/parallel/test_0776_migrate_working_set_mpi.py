"""Swarm.migrate's round loop: correctness, and the collective it must not skip.

The loop classifies points once per round. It used to re-offer the whole local
array to `points_in_domain` every round, so the classification work per rank grew
with the round count — a fixed 47k-point set was offered 238k times at np=4. It
now remembers what it has already claimed and only classifies the rest.

The claimed set is keyed by COORDINATE, not by index: `dm.migrate` does not
preserve the local ordering (measured — retained points are not left at the
front), so an index from the previous round names a different particle after the
move.

The second test is here because the first draft of that change deadlocked. It
made the `points_in_domain` call conditional on this rank having undecided
points, and `points_in_domain` is collective — it reaches `get_max_radius()`
before any short-circuit precisely so a rank with nothing to classify still
joins the reduction (the #405 treatment, stated in its own source). At np=4 a
rank ran out of undecided points, skipped the reduction, and the job hung with
every rank inside the call.

Run under MPI::

    mpirun -np 4 python -m pytest --with-mpi \
        tests/parallel/test_0776_migrate_working_set_mpi.py
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [
    pytest.mark.mpi(min_size=2),
    pytest.mark.timeout(180),
    pytest.mark.level_1,
    pytest.mark.tier_a,
]

SEED = 20260819


def _scattered_swarm(cell_size=0.15, n_points=2000):
    """A swarm whose points are spread over the WHOLE domain.

    Populate alone leaves every particle already owned by its own rank, so
    nothing migrates and the round loop never runs — the case that makes this
    file look like it is testing something when it is not.
    """

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cell_size,
        regular=False, qdegree=2)
    swarm = uw.swarm.Swarm(mesh=mesh)
    swarm.populate(fill_param=3)

    rng = np.random.default_rng(SEED)
    everywhere = rng.uniform(0.01, 0.99, size=(n_points, 2))
    mine = np.array_split(everywhere, uw.mpi.size)[uw.mpi.rank]

    coords = swarm.dm.getField("DMSwarmPIC_coor").reshape(-1, mesh.dim)
    n = min(coords.shape[0], mine.shape[0])
    coords[:n] = mine[:n]
    swarm.dm.restoreField("DMSwarmPIC_coor")

    return mesh, swarm


def _local_coords(swarm, dim):
    out = swarm.dm.getField("DMSwarmPIC_coor").reshape(-1, dim).copy()
    swarm.dm.restoreField("DMSwarmPIC_coor")

    return out


def test_premise_the_fixture_actually_migrates():
    """Without points crossing ranks the round loop never runs."""

    mesh, swarm = _scattered_swarm()
    before = _local_coords(swarm, mesh.dim)
    owned_before = int(np.count_nonzero(mesh.points_in_domain(before)))
    strangers = uw.mpi.comm.allreduce(before.shape[0] - owned_before)

    assert strangers > 0, (
        "every point is already owned by the rank holding it, so migration has "
        "nothing to do and neither of the tests below exercises the loop")


def test_migrate_returns_and_every_point_lands_on_a_rank_that_holds_it():
    """The loop terminates, and its answer is right.

    The deadlock this guards against leaves every rank inside `points_in_domain`,
    so the failure is a timeout rather than an assertion.
    """

    mesh, swarm = _scattered_swarm()
    before = _local_coords(swarm, mesh.dim)
    n_before = uw.mpi.comm.allreduce(before.shape[0])

    swarm.migrate()

    after = _local_coords(swarm, mesh.dim)
    n_after = uw.mpi.comm.allreduce(after.shape[0])

    assert n_after == n_before, (
        f"migration changed the global particle count: {n_before} -> {n_after}")

    if after.shape[0]:
        owned = mesh.points_in_domain(after)
        assert bool(owned.all()), (
            f"rank {uw.mpi.rank} holds {int(np.count_nonzero(~owned))} points "
            "its own mesh does not contain")
    else:
        mesh.points_in_domain(after)      # collective: join the reduction


def test_the_partition_is_the_same_at_every_rank_count():
    """The global set of owned coordinates is the input set, whatever np is.

    Keyed on the global multiset rather than per-rank counts, which are a
    partition detail. This is what a change to the claimed-set bookkeeping would
    break: dropping a point, or claiming one twice.
    """

    mesh, swarm = _scattered_swarm()
    before = _local_coords(swarm, mesh.dim)
    swarm.migrate()
    after = _local_coords(swarm, mesh.dim)

    def gathered_rows(a):
        rows = uw.mpi.comm.allgather(np.ascontiguousarray(a))
        stacked = np.concatenate([r for r in rows if r.size], axis=0)
        return stacked[np.lexsort((stacked[:, 1], stacked[:, 0]))]

    assert np.array_equal(gathered_rows(before), gathered_rows(after)), (
        "the multiset of particle coordinates changed across migration — a "
        "point was lost or duplicated")
