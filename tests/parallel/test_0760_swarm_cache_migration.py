"""
Regression test for swarm cache invalidation after migration.

Verifies that SwarmVariable._canonical_data caches are properly invalidated
after Swarm.migrate() and after bare-bones dm.migrate() in global_evaluate.

Bug: SwarmVariable caches were only invalidated inside the delete_lost_points
branch of Swarm.migrate(), so caches became stale when particles moved between
ranks without deletion. This caused shape mismatches in global_evaluate.

See: https://github.com/underworldcode/underworld3/issues/64

Run with:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0760_swarm_cache_migration.py
"""

import pytest
import numpy as np
import underworld3 as uw
from mpi4py import MPI

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(60)]


@pytest.mark.mpi(min_size=2)
@pytest.mark.level_1
@pytest.mark.tier_a
def test_swarm_cache_valid_after_migration():
    """Swarm variable caches must reflect actual particle count after migration."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0))
    swarm = uw.swarm.Swarm(mesh)
    var = uw.swarm.SwarmVariable("test_var", swarm, vtype=uw.VarType.SCALAR, _proxy=False)

    # Add particles at random positions — distribution will be uneven across ranks
    np.random.seed(42 + uw.mpi.rank)
    coords = np.random.random((200, mesh.dim))
    swarm.add_particles_with_global_coordinates(coords, migrate=False)
    var.data[...] = uw.mpi.rank

    pre_count = swarm.dm.getLocalSize()

    # Migrate — particles move to owning rank
    swarm.migrate(remove_sent_points=True, delete_lost_points=False)

    post_count = swarm.dm.getLocalSize()
    coords_cached = swarm._particle_coordinates.data.shape[0]
    var_cached = var.data.shape[0]

    # Cached sizes must match the actual DM particle count
    assert coords_cached == post_count, (
        f"Rank {uw.mpi.rank}: coordinate cache ({coords_cached}) != "
        f"DM count ({post_count}) after migration"
    )
    assert var_cached == post_count, (
        f"Rank {uw.mpi.rank}: variable cache ({var_cached}) != "
        f"DM count ({post_count}) after migration"
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.level_1
@pytest.mark.tier_a
def test_global_evaluate_after_migration():
    """global_evaluate must succeed with coordinates that force heavy migration."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0))
    v = uw.discretisation.MeshVariable("u", mesh, mesh.dim, degree=2)

    # Bias coordinates to one side — forces cross-rank particle movement
    np.random.seed(42)
    N = 300
    coords = np.random.random((N, mesh.dim))
    coords[:, 0] = 0.5 + 0.5 * coords[:, 0]  # all in right half

    result = uw.function.global_evaluate(v.sym, coords)

    assert result.shape[0] == N, (
        f"Rank {uw.mpi.rank}: expected {N} results, got {result.shape[0]}"
    )


@pytest.mark.mpi(min_size=2)
@pytest.mark.level_1
@pytest.mark.tier_a
def test_global_evaluate_displaced_nodes():
    """global_evaluate with displaced node coordinates (DDt/SemiLagrangian path)."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0))
    v = uw.discretisation.MeshVariable("u", mesh, mesh.dim, degree=2)

    # Displace node coordinates — simulates semi-Lagrangian departure points
    node_coords = mesh.X.coords
    np.random.seed(7)
    displacement = np.random.random(node_coords.shape) * 0.3
    mid_pt_coords = node_coords - displacement

    # Clamp to domain
    mid_pt_coords = np.clip(mid_pt_coords, 0.0, 1.0)

    result = uw.function.evaluate(v.sym, mid_pt_coords)

    assert result.shape[0] == node_coords.shape[0], (
        f"Rank {uw.mpi.rank}: expected {node_coords.shape[0]} results, "
        f"got {result.shape[0]}"
    )
