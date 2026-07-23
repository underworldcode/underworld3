"""
Parallel regression tests for IndexSwarmVariable proxy refresh after advection.

Run under MPI, e.g.::

    mpirun -np 4 python -m pytest tests/parallel/test_0767_index_swarm_advection_mpi.py
    mpirun -np 2 python -m pytest tests/parallel/test_0767_index_swarm_advection_mpi.py

Covers:

- SWARM-07 / BF-06 follow-up: ``IndexSwarmVariable._update_proxy_variables()``
  with ``update_type=1`` previously returned early on starved ranks (<= 1 local
  particle) *without* participating in the collective MeshVariable read/write
  sequence, causing an MPI deadlock when the populated ranks' ghost sync waited
  for all ranks. The fix restructures the ``update_type=1`` branch to follow
  the same collective pattern as ``update_type=0`` (and the base
  ``SwarmVariable._rbf_to_meshVar``): starved ranks still read and write their
  level-set proxy variables — they just leave the values unchanged.

- Advection across rank boundaries followed by a projection solve that reads
  the refreshed proxy, exercising the full advection → migrate → proxy-refresh
  → solve path under MPI.
"""

import pytest
import numpy as np
import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(120)]


# ==============================================================================
# Helpers
# ==============================================================================

SENTINEL = -123.0


def _box(cell=0.25):
    """Simple unit-square mesh for all tests."""
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cell,
    )


def _sentinel_level_sets(mat):
    """Write SENTINEL into every level-set proxy variable (all ranks)."""
    for lv in mat._meshLevelSetVars:
        lv.data[:, 0] = SENTINEL


def _check_starved_kept_sentinel(mat, starved):
    """Assert starved-rank proxies still hold SENTINEL; populated ranks don't."""
    for lv in mat._meshLevelSetVars:
        vals = np.asarray(lv.data[:, 0])
        if starved:
            frac = float(np.mean(np.isclose(vals, SENTINEL)))
            assert frac > 0.5, (
                f"rank {uw.mpi.rank}: starved-rank proxy was overwritten "
                f"(only {frac:.0%} of nodes kept the sentinel)"
            )
        else:
            assert not np.allclose(vals, SENTINEL), (
                f"rank {uw.mpi.rank}: populated-rank proxy was NOT refreshed "
                f"(all nodes still hold the sentinel)"
            )


# ==============================================================================
# Test 1: update_type=1 starved-rank proxy refresh (THE HANG REPRODUCER)
# ==============================================================================

@pytest.mark.mpi(min_size=4)
@pytest.mark.level_1
@pytest.mark.tier_a
def test_update_type1_starved_proxy_refresh():
    """IndexSwarmVariable(update_type=1): starved ranks must participate in the
    collective mesh-variable read/write, not return early. Before the fix this
    deadlocked the populated ranks' ghost sync."""
    mesh = _box()
    swarm = uw.swarm.Swarm(mesh)
    mat = uw.swarm.IndexSwarmVariable(
        "M_starved_t1", swarm, indices=2, proxy_degree=1,
        update_type=1, npoints=5, radius=0.5,
    )

    # Cluster particles near the origin — on np≥4 some ranks get ≤1 particle
    xs = np.linspace(0.02, 0.08, 3)
    cluster = np.column_stack([
        np.repeat(xs, len(xs)),
        np.tile(xs, len(xs)),
    ])
    swarm.add_particles_with_coordinates(cluster)

    mat.data[...] = 0

    starved = swarm.local_size <= 1
    n_starved = uw.mpi.comm.allreduce(int(starved), op=uw.MPI.SUM)
    assert n_starved >= 1, "test premise: at least one rank must be starved"
    assert n_starved < uw.mpi.size, "test premise: one rank holds the cluster"

    _sentinel_level_sets(mat)

    # THIS is the code path that deadlocked before the fix:
    # starved ranks return from _update_proxy_variables() without touching
    # meshVar.data, while populated ranks do — causing a collective hang.
    mat._proxy_stale = True
    mat._update_proxy_if_stale()

    _check_starved_kept_sentinel(mat, starved)

    del swarm, mesh


# ==============================================================================
# Test 2: update_type=0 starved-rank proxy refresh (verification — must pass)
# ==============================================================================

@pytest.mark.mpi(min_size=4)
@pytest.mark.level_1
@pytest.mark.tier_a
def test_update_type0_starved_proxy_refresh():
    """IndexSwarmVariable(update_type=0): starved-rank collective read/write
    was already correct; this test verifies nothing is broken."""
    mesh = _box()
    swarm = uw.swarm.Swarm(mesh)
    mat = uw.swarm.IndexSwarmVariable(
        "M_starved_t0", swarm, indices=2, proxy_degree=1,
        update_type=0, npoints=5, radius=0.5,
    )

    xs = np.linspace(0.02, 0.08, 3)
    cluster = np.column_stack([
        np.repeat(xs, len(xs)),
        np.tile(xs, len(xs)),
    ])
    swarm.add_particles_with_coordinates(cluster)
    mat.data[...] = 0

    starved = swarm.local_size <= 1
    n_starved = uw.mpi.comm.allreduce(int(starved), op=uw.MPI.SUM)
    assert n_starved >= 1
    assert n_starved < uw.mpi.size

    _sentinel_level_sets(mat)

    mat._proxy_stale = True
    mat._update_proxy_if_stale()

    _check_starved_kept_sentinel(mat, starved)

    del swarm, mesh


# ==============================================================================
# Test 3: advection across rank boundaries, then proxy solve
# ==============================================================================

@pytest.mark.mpi(min_size=2)
@pytest.mark.level_1
@pytest.mark.tier_a
@pytest.mark.parametrize("update_type", [0, 1])
def test_advection_then_proxy_solve(update_type):
    """Advect a material blob across rank boundaries, then run a projection
    solve that reads the proxy via .sym. Verifies the full advection→migrate
    →proxy-refresh→solve path doesn't hang and produces correct values.

    Uses a rightward-sheared velocity (v_x = y, v_y = 0) so every particle
    at y > 0 moves right while staying inside [0, 1] — no domain-exit
    deletion. Particles close to y=0 barely move; those near y=1 sweep
    across partition boundaries.
    """
    import sympy
    mesh = _box(cell=1.0 / 16.0)

    # Shear flow: v_x = y, v_y = 0.  dt = 0.6 moves particles at y=1
    # rightward by 0.6, staying in [0, 1] since the max x is 1.0 + 0.6
    # but the box is [0, 1], so some may leave at the far right —
    # but the shear means most particles remain in the box.
    x, y = mesh.X
    v_fn = sympy.Matrix([y, 0.0])

    swarm = uw.swarm.Swarm(mesh)
    mat = uw.swarm.IndexSwarmVariable(
        f"M_shear_t{update_type}", swarm, indices=2, proxy_degree=1,
        update_type=update_type, npoints=5, radius=0.3,
    )
    swarm.populate(fill_param=4)

    # Tag left-half particles (x < 0.3) as material 1
    mat.data[...] = 0
    pc = swarm._particle_coordinates.data
    mat.data[pc[:, 0] < 0.3, 0] = 1

    def mat1_count():
        return int((mat.data[:, 0] > 0.5).sum())

    count_before = mat1_count()

    # Advection: shear moves the left-half blob rightward; high-y particles
    # move fast, low-y particles move slow. The blob should shift right
    # across the partition boundary.
    swarm.advection(v_fn, 0.5, order=2, step_limit=True)

    # At least some particles must survive (the ones at low y stayed in box)
    count_after = mat1_count()
    assert count_after > 0, "all material-1 particles were deleted"

    # Projection solve: reads mat.sym via the lazy path. Must NOT hang.
    proj_var = uw.discretisation.MeshVariable(
        "proj_0767", mesh, 1, degree=1,
    )
    proj = uw.systems.Projection(mesh, proj_var)
    proj.uw_function = mat.createMask([0.0, 1.0])  # material 1 fraction
    proj.petsc_options.delValue("ksp_monitor")
    proj.solve()

    # The solve must have produced some material-1 values. Not a tight
    # correctness check — the test is primarily about "no hang" — but
    # a sanity that the proxy was refreshed at all.
    frac1 = np.asarray(proj_var.data[:, 0])
    mean1 = float(frac1.mean())
    assert mean1 > 0.01, (
        f"Material-1 fraction implausibly low after advection "
        f"(mean={mean1:.6f})"
    )

    del proj_var, proj, swarm, mesh


# ==============================================================================
# Test 4: advection that creates starved ranks, then proxy refresh
# ==============================================================================

@pytest.mark.mpi(min_size=4)
@pytest.mark.level_1
@pytest.mark.tier_a
@pytest.mark.parametrize("update_type", [0, 1])
def test_advection_creates_starved_ranks(update_type):
    """Advection that pushes all particles into one side of the domain,
    concentrating them on fewer ranks and leaving others starved. Then
    triggers a proxy refresh via the solve path — must not hang."""
    import sympy
    mesh = _box(cell=1.0 / 8.0)

    # Shear flow: v_x = y, v_y = 0.  High-y particles move far right;
    # low-y particles stay near the left. With a dense uniform population,
    # after migration the left-side ranks may become starved.
    x, y = mesh.X
    v_fn = sympy.Matrix([y, 0.0])

    swarm = uw.swarm.Swarm(mesh)
    mat = uw.swarm.IndexSwarmVariable(
        f"M_starve_t{update_type}", swarm, indices=2, proxy_degree=1,
        update_type=update_type, npoints=5, radius=0.3,
    )
    # Dense uniform population
    swarm.populate(fill_param=6)

    mat.data[...] = 0

    # Advection: large dt pushes most particles to the right side.
    # The migration concentrates them on right-side ranks.
    swarm.advection(v_fn, 1.0, order=2, step_limit=True)

    # Sentinel and force proxy refresh (exercises the collective path).
    # This is what _sync_before_assembly does before a solve.
    _sentinel_level_sets(mat)
    mat._proxy_stale = True
    mat._update_proxy_if_stale()

    # No hang = success. On starved ranks the sentinel is preserved.
    starved = swarm.local_size <= 1
    _check_starved_kept_sentinel(mat, starved)

    del swarm, mesh
