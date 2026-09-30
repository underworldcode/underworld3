"""Regression tests for swarm stale-cache and stale-proxy bugs.

Covers the Track-0 quality-audit findings (docs/reviews/2026-07):

- SWARM-01 / BF-02: ``Swarm.migrate()`` early-returned when no particle needed
  to change rank *without* invalidating the cached canonical ``.data`` arrays,
  so ``add_particles_with_global_coordinates`` (whose only invalidation route
  was migrate) left every variable's cached ``.data`` at the old particle
  count.
- SWARM-02 / BF-02: the same early return left the cached particle kd-tree
  built over a mutated coordinate buffer, silently corrupting
  ``rbf_interpolate`` and every proxy update.
- SWARM-17 / BF-02: ``populate()`` never invalidated caches created before the
  swarm was populated.
- GitHub issue #289: ``swarm.advection()`` left proxy mesh variables frozen at
  the pre-advection particle positions (the no-move migrate early return never
  marked them stale), so time-stepped models silently used stationary
  material fields.
- SWARM-05 / LE-03 / BF-08 (issue #215 Bug 3): solvers read the proxy
  MeshVariable's DM directly, bypassing the lazy ``.sym`` refresh, so a solve
  after a ``material.data`` write consumed stale proxy values.
"""

import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing import UnstructuredSimplexBox

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture
def mesh():
    return UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=1.0 / 8.0,
    )


# --------------------------------------------------------------------------
# SWARM-01: cache size after add_particles_with_global_coordinates
# --------------------------------------------------------------------------

@pytest.mark.parametrize("migrate", [True, False])
def test_add_particles_global_coordinates_invalidates_cache(mesh, migrate):
    swarm = uw.swarm.Swarm(mesh=mesh)
    var = swarm.add_variable("val", 1)
    swarm.populate(fill_param=2)

    _ = var.data.shape  # populate the canonical cache at the old size

    new_pts = np.array([[0.51, 0.51], [0.52, 0.52], [0.11, 0.87]])
    swarm.add_particles_with_global_coordinates(new_pts, migrate=migrate)

    dm_size = swarm.dm.getLocalSize()
    assert var.data.shape[0] == dm_size, (
        f"cached .data rows ({var.data.shape[0]}) out of sync with the DMSwarm "
        f"local size ({dm_size}) after add_particles_with_global_coordinates"
    )
    assert (
        swarm._particle_coordinates.data.shape[0] == dm_size
    ), "coordinate cache out of sync after particle addition"

    del swarm
    del mesh


# --------------------------------------------------------------------------
# SWARM-17: cache created before populate()
# --------------------------------------------------------------------------

def test_populate_invalidates_pre_existing_cache(mesh):
    swarm = uw.swarm.Swarm(mesh=mesh)
    var = swarm.add_variable("val", 1)

    _ = var.data.shape  # create the (0-row) canonical cache before populate

    swarm.populate(fill_param=2)

    dm_size = swarm.dm.getLocalSize()
    assert dm_size > 0
    assert var.data.shape[0] == dm_size, (
        f"cached .data rows ({var.data.shape[0]}) out of sync with the DMSwarm "
        f"local size ({dm_size}) after populate()"
    )

    del swarm
    del mesh


# --------------------------------------------------------------------------
# SWARM-02: kd-tree refreshed after an in-domain coordinate move
# --------------------------------------------------------------------------

def test_rbf_interpolate_uses_current_positions_after_no_move_migrate(mesh):
    swarm = uw.swarm.Swarm(mesh=mesh)
    var = swarm.add_variable("val", 1)
    swarm.populate(fill_param=3)

    # tag every particle with its launch x-coordinate
    var.data[:, 0] = swarm._particle_coordinates.data[:, 0]

    probe = np.array([[0.15, 0.5], [0.85, 0.5], [0.25, 0.3], [0.75, 0.7]])
    _ = var.rbf_interpolate(probe)  # builds and caches the particle kd-tree

    # mirror every particle in-domain: x -> 1 - x. The cached KDTree holds a
    # NO-COPY view of the coordinate buffer, so after this in-place mutation
    # its index topology (built at the old positions) is inconsistent with
    # its stored points — queries return garbage, not merely frozen values.
    coords = swarm._particle_coordinates.data
    swarm._particle_coordinates.data[...] = np.column_stack(
        [1.0 - coords[:, 0], coords[:, 1]]
    )
    swarm.migrate()  # serial / no-rank-change: previously hit the early return

    # the invalidation contract: migrate() must drop the cached tree
    assert swarm._kdtree is None, (
        "migrate() left the cached particle kd-tree in place after an "
        "in-place coordinate mutation (SWARM-02 early-return hole)"
    )

    values = var.rbf_interpolate(probe)

    # with a FRESH tree, the particle now found at probe_x carries its launch
    # coordinate 1 - probe_x.
    expected = 1.0 - probe[:, 0]
    err = np.abs(values[:, 0] - expected).max()
    assert err < 0.1, (
        f"rbf_interpolate is using a stale/poisoned kd-tree after migrate() "
        f"(max error {err:.3f} against fresh-tree expectation)"
    )

    del swarm
    del mesh


# --------------------------------------------------------------------------
# issue #289: proxy tracks the particles through advection
# --------------------------------------------------------------------------

def test_issue289_proxy_tracks_particles_through_advection():
    """Reproducer from GitHub issue #289 (reduced): a blob of material is
    advected upward with a uniform velocity; the lazily-refreshed proxy must
    track the particle centroid, not stay frozen at the launch position."""
    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 0.25), cellSize=1.0 / 24.0, qdegree=3
    )
    v = uw.discretisation.MeshVariable("V", mesh, vtype=uw.VarType.VECTOR, degree=2)
    swarm = uw.swarm.Swarm(mesh=mesh)
    mat = uw.swarm.IndexSwarmVariable(
        "M", swarm, indices=2, proxy_degree=1, proxy_continuous=True
    )
    swarm.populate(fill_param=3)

    mat.data[...] = 0
    pc = swarm._particle_coordinates.data
    blob = (pc[:, 0] - 0.5) ** 2 + (pc[:, 1] - 0.107) ** 2 < 0.03**2
    assert blob.sum() > 0, "test premise: blob must contain particles"
    mat.data[blob, 0] = 1

    v.data[:, 1] = 0.01  # uniform upward velocity

    def particle_cy():
        m = mat.data[:, 0] > 0.5
        return swarm._particle_coordinates.data[m, 1].mean()

    def proxy_cy():
        pv = np.asarray(uw.function.evaluate(mat.sym[1], v.coords)).reshape(-1)
        w = np.clip(pv, 0.0, None)
        return float((w * np.asarray(v.coords)[:, 1]).sum() / w.sum())

    cy_particles_before = particle_cy()
    cy_proxy_before = proxy_cy()
    assert abs(cy_proxy_before - cy_particles_before) < 0.01

    swarm.advection(v.sym, 2.0, order=2, corrector=True)  # blob moves up 0.02

    cy_particles_after = particle_cy()
    moved = cy_particles_after - cy_particles_before
    assert moved > 0.015, "test premise: the particles must actually move"

    # lazy .sym access must now see the refreshed proxy
    cy_proxy_after = proxy_cy()
    tracked = cy_proxy_after - cy_proxy_before
    assert tracked > 0.5 * moved, (
        f"proxy frozen after advection: particle centroid moved {moved:.4f} "
        f"but proxy centroid moved only {tracked:.4f} (issue #289)"
    )

    del swarm
    del mesh


# --------------------------------------------------------------------------
# SWARM-05 / BF-08 (issue #215 Bug 3): solve consumes a fresh proxy
# --------------------------------------------------------------------------

def test_projection_solve_consumes_fresh_proxy(mesh):
    """Write material.data, run a Projection solve WITHOUT touching .sym, and
    assert the solve consumed the new values (the proxy is refreshed eagerly
    at solve entry rather than only via the lazy .sym accessor)."""
    swarm = uw.swarm.Swarm(mesh=mesh)
    var = swarm.add_variable("mat", 1, proxy_degree=1)
    swarm.populate(fill_param=3)

    var.data[:, 0] = 1.0

    proj_var = uw.discretisation.MeshVariable("pv0113", mesh, 1, degree=1)
    proj = uw.systems.Projection(mesh, proj_var)
    proj.uw_function = var.sym[0]  # captured ONCE; refreshes the proxy now
    proj.petsc_options.delValue("ksp_monitor")

    proj.solve()
    assert abs(float(np.mean(proj_var.data)) - 1.0) < 0.05

    # update the particle data; do NOT touch .sym before solving
    var.data[:, 0] = 2.0
    proj.solve()

    mean_after = float(np.mean(proj_var.data))
    assert abs(mean_after - 2.0) < 0.05, (
        f"Projection solve consumed a stale proxy: mean {mean_after:.3f}, "
        "expected ~2.0 (issue #215 Bug 3 / SWARM-05)"
    )

    del swarm
    del mesh


# --------------------------------------------------------------------------
# Fossil-variable lifetime contract
# --------------------------------------------------------------------------

def test_variable_outliving_swarm_is_a_usable_fossil(mesh):
    """A SwarmVariable holds its parent swarm by WEAK reference (strong
    back-references would defer DMSwarm destruction to gc time — the
    transient-evaluation-swarm leak). A variable that outlives its swarm
    must therefore remain usable SYMBOLICALLY — ``.sym`` returns the last
    projection with a warning, never raising from a stale-refresh attempt —
    while particle-data access still raises the lifetime error.
    (Regression: test_0726's fixture drops the swarm and composes with the
    surviving variable; the populate() invalidation fix made the lazy
    refresh dereference the dead parent.)"""
    import gc
    import warnings

    swarm = uw.swarm.Swarm(mesh=mesh)
    var = uw.swarm.SwarmVariable("fossil", swarm, size=1, proxy_degree=1)
    swarm.populate(fill_param=2)  # marks the proxy stale (SWARM-17 fix)

    del swarm
    gc.collect()
    assert var._swarm_ref() is None, "test premise: parent swarm collected"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sym = var.sym  # must WARN, not raise
    assert sym is not None
    assert any("no longer exists" in str(w.message) for w in caught)

    # particle-data access still enforces the lifetime guard
    with pytest.raises(RuntimeError, match="garbage collected"):
        _ = var.swarm

    del mesh
