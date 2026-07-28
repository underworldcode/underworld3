"""Proxy transfer operators are shared across variables and self-invalidate.

The `order=1` weights depend only on geometry — proxy node coordinates and
particle positions — so every proxied variable whose proxy has the same degree
and continuity on the same mesh needs the *same* operator. Building it once per
(geometry, stencil) instead of once per variable is worth ~70% of the refresh
cost on a swarm carrying four proxied variables.

The correctness requirement is that this changes nothing about the values, and
that a cached operator can never outlive the particle positions it was built
from. Validity is tied to the kd-tree instance rather than to a flag, so
`migrate()` — which drops the tree — invalidates the cache structurally.

Known inherited limitation, deliberately not guarded here: the kd-tree is a
no-copy view of the coordinate buffer, so writing coordinates in place without
migrating leaves the tree stale (SWARM-02, `test_0113`). A cached operator is
stale in exactly the same circumstances and no others.
"""

import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing import UnstructuredSimplexBox

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _swarm_with(n_vars, proxy_degrees=None, cell_size=1.0 / 8.0):
    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cell_size
    )
    degrees = proxy_degrees or [2] * n_vars
    swarm = uw.swarm.Swarm(mesh)
    variables = [
        swarm.add_variable(name=f"v{i}", size=1, proxy_degree=d)
        for i, d in enumerate(degrees)
    ]
    swarm.populate(fill_param=3)
    rng = np.random.default_rng(5)
    for var in variables:
        var.data[:, 0] = rng.random(swarm.local_size)
    return mesh, swarm, variables


def test_cached_refresh_matches_the_uncached_result():
    """The cache must be invisible in the values it produces."""
    mesh, swarm, variables = _swarm_with(2)
    var = variables[0]

    swarm._proxy_interpolation_cache.clear()
    var._rbf_to_meshVar(var._meshVar)
    first = np.asarray(var._meshVar.data[:, 0]).copy()

    # Second refresh hits the cache; a third with the cache cleared rebuilds.
    var._rbf_to_meshVar(var._meshVar)
    cached = np.asarray(var._meshVar.data[:, 0]).copy()

    swarm._proxy_interpolation_cache.clear()
    var._rbf_to_meshVar(var._meshVar)
    rebuilt = np.asarray(var._meshVar.data[:, 0]).copy()

    assert np.array_equal(cached, first)
    assert np.array_equal(rebuilt, first)

    del swarm
    del mesh


def test_operator_is_shared_between_variables_of_the_same_proxy_degree():
    mesh, swarm, variables = _swarm_with(3)
    swarm._proxy_interpolation_cache.clear()

    for var in variables:
        var._rbf_to_meshVar(var._meshVar)

    assert len(swarm._proxy_interpolation_cache) == 1, (
        "three variables with the same proxy degree should share one operator, "
        f"got {len(swarm._proxy_interpolation_cache)} cache entries"
    )

    del swarm
    del mesh


def test_different_proxy_degrees_do_not_share_an_operator():
    """Different degrees mean different node coordinates."""
    mesh, swarm, variables = _swarm_with(2, proxy_degrees=[1, 2])
    swarm._proxy_interpolation_cache.clear()

    for var in variables:
        var._rbf_to_meshVar(var._meshVar)

    assert len(swarm._proxy_interpolation_cache) == 2

    coords = [np.asarray(v._meshVar.coords_nd) for v in variables]
    assert coords[0].shape != coords[1].shape

    del swarm
    del mesh


def test_migrate_invalidates_the_cached_operator():
    """A cached operator must not survive the particle positions it used."""
    mesh, swarm, variables = _swarm_with(1)
    var = variables[0]

    var._rbf_to_meshVar(var._meshVar)
    key = next(iter(swarm._proxy_interpolation_cache))
    tree_before, operator_before = swarm._proxy_interpolation_cache[key]

    coords = swarm._particle_coordinates.data.copy()
    coords[:, 0] = np.clip(coords[:, 0] + 0.05, 0.001, 0.999)
    swarm._particle_coordinates.data[...] = coords
    swarm.migrate()

    var._rbf_to_meshVar(var._meshVar)
    tree_after, operator_after = swarm._proxy_interpolation_cache[key]

    assert tree_after is not tree_before, "kd-tree survived a migrate"
    assert operator_after is not operator_before, (
        "the operator was reused across a migrate — it encodes stale particle "
        "positions"
    )

    del swarm
    del mesh


def test_linear_field_still_exact_through_the_cached_path():
    """The guarantee the operator exists to provide survives caching."""
    mesh, swarm, variables = _swarm_with(2)
    var = variables[0]
    proxy = var._meshVar

    particle_coords = swarm._particle_coordinates.data
    var.data[:, 0] = 0.5 + particle_coords @ np.array([1.0, 2.0])

    var._rbf_to_meshVar(proxy)          # miss
    var._rbf_to_meshVar(proxy)          # hit

    node_coords = np.asarray(proxy.coords_nd)
    expected = 0.5 + node_coords @ np.array([1.0, 2.0])
    error = np.abs(np.asarray(proxy.data[:, 0]) - expected).max()

    assert error < 1.0e-12, f"cached path lost linear exactness: {error:.3e}"

    del swarm
    del mesh


def test_monotone_refresh_bypasses_the_cache():
    """The limiter is data-dependent, so it cannot use a geometry-only
    operator; it must still produce a bounded, correct result."""
    mesh, swarm, variables = _swarm_with(1)
    var = variables[0]
    proxy = var._meshVar

    particle_coords = swarm._particle_coordinates.data
    var.data[:, 0] = 0.5 + particle_coords @ np.array([1.0, 2.0])

    swarm._proxy_interpolation_cache.clear()
    var._rbf_to_meshVar(proxy, monotone=True)

    assert len(swarm._proxy_interpolation_cache) == 0, (
        "the monotone path populated the geometry-only cache"
    )

    node_coords = np.asarray(proxy.coords_nd)
    expected = 0.5 + node_coords @ np.array([1.0, 2.0])
    assert np.abs(np.asarray(proxy.data[:, 0]) - expected).max() < 1.0e-12

    del swarm
    del mesh
