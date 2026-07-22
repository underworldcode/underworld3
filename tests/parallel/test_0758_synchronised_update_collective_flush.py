"""MPI regression tests for the ``uw.synchronised_array_update`` rework (#379).

The old machinery queued (callback, array, change_info) events per rank and
replayed them at context exit. A mesh-variable flush is collective (ghost
sync via localToGlobal/globalToLocal), so ranks that queued different events
ran different collectives — a rank-0-only write hung every other rank.

The rework marks touched variables dirty by a creation-order registration
id, allgathers the union of dirty ids at context exit, and flushes the
union in id order on every rank. These tests exercise exactly the shapes
that used to desynchronise: rank-uneven writes, opposite per-rank write
order, variables created in helpers, mixed swarm + mesh updates, and the
zero-size-slice case on an empty rank.

Run with: mpirun -np 2 (or -np 4) python -m pytest --with-mpi <this file>
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [
    pytest.mark.level_2,
    pytest.mark.tier_a,
    pytest.mark.mpi(min_size=2),
    pytest.mark.timeout(120),
]


@pytest.fixture()
def mesh():
    return uw.meshing.StructuredQuadBox(
        elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )


@pytest.mark.mpi(min_size=2)
def test_rank_uneven_write_completes(mesh):
    """Rank 0 writes, the others do not. Pre-rework this hung: only rank 0
    queued the collective pack. Post-rework every rank flushes the agreed
    union, so the context exits everywhere."""
    t = uw.discretisation.MeshVariable("t758a", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)
    t.array[...] = 1.0

    with uw.synchronised_array_update("rank-uneven write"):
        if uw.mpi.rank == 0:
            t.data[...] = 2.0

    # Reaching here on every rank IS the regression check: exit is
    # collective, so a desynchronised flush would hang above, not fail here.
    # After the ghost sync, each dof carries its OWNER's value (rank 0's
    # writes survive on rank-0-owned dofs; rank-1-owned dofs stay 1.0), so
    # the checks must be global reductions, identical on every rank.
    assert float(t.data.global_max()) == pytest.approx(2.0)
    assert float(t.data.global_min()) == pytest.approx(1.0)


@pytest.mark.mpi(min_size=2)
def test_opposite_write_order_agrees(mesh):
    """Two variables written in opposite per-rank order must flush in the
    same (creation) order on every rank — order must not depend on the
    order of writes."""
    a = uw.discretisation.MeshVariable("a758b", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)
    b = uw.discretisation.MeshVariable("b758b", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)

    with uw.synchronised_array_update("opposite order"):
        if uw.mpi.rank % 2 == 0:
            a.data[...] = 1.0
            b.data[...] = 2.0
        else:
            b.data[...] = 2.0
            a.data[...] = 1.0

    assert np.allclose(np.asarray(a.data), 1.0)
    assert np.allclose(np.asarray(b.data), 2.0)
    uw.mpi.barrier()


@pytest.mark.mpi(min_size=2)
def test_helper_created_variable_ids_agree(mesh):
    """Variables created inside helpers (the temp-variable pattern) written
    on ONE rank only: the creation-order ids must line up across ranks so
    the agreed flush resolves the same variable everywhere."""

    def make_scratch(tag):
        return uw.discretisation.MeshVariable(
            f"scratch758_{tag}", mesh, 1, vtype=uw.VarType.SCALAR, degree=1
        )

    s1 = make_scratch("one")
    s2 = make_scratch("two")

    with uw.synchronised_array_update("helper-created"):
        if uw.mpi.rank == uw.mpi.size - 1:
            s2.data[...] = 5.0
        if uw.mpi.rank == 0:
            s1.data[...] = 4.0

    uw.mpi.barrier()


@pytest.mark.mpi(min_size=2)
def test_mixed_swarm_and_mesh_updates(mesh):
    """Swarm-variable packs are rank-local (flushed without agreement);
    mesh-variable packs are collective (flushed by agreed id). Both kinds
    in one context must compose."""
    t = uw.discretisation.MeshVariable("t758d", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)
    swarm = uw.swarm.Swarm(mesh=mesh)
    sv = uw.swarm.SwarmVariable("sv758d", swarm, 1)
    swarm.populate(fill_param=2)

    with uw.synchronised_array_update("mixed swarm + mesh"):
        if uw.mpi.rank == 0:
            t.data[...] = 3.0
        sv.data[...] = float(uw.mpi.rank + 1)

    assert np.allclose(np.asarray(sv.data), float(uw.mpi.rank + 1))
    uw.mpi.barrier()


@pytest.mark.mpi(min_size=2)
def test_zero_size_slice_on_one_rank(mesh):
    """An empty slice write still fires the callback machinery; the
    empty-rank classification (base-chain identity, not may_share_memory)
    must keep every rank in the same flush branch."""
    t = uw.discretisation.MeshVariable("t758e", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)
    t.array[...] = 1.0

    with uw.synchronised_array_update("zero-size slice"):
        if uw.mpi.rank == 0:
            t.data[0:0] = 9.0  # writes nothing, but exercises the marking path
        else:
            mask = np.zeros(t.data.shape[0], dtype=bool)
            mask[::3] = True
            t.data[mask] = 2.0

    # Global reductions: symmetric on every rank. The masked 2.0 writes on
    # ranks != 0 survive on the dofs those ranks own; nothing exceeds 2.0
    # (the zero-size 9.0 write touched no memory).
    assert float(t.data.global_max()) == pytest.approx(2.0)
    assert float(t.data.global_min()) == pytest.approx(1.0)
