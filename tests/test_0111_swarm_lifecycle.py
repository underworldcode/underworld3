"""Lifecycle regression tests for Swarm cleanup.

The prep work for the global-point-routing redesign restores the swarm
lifecycle so that transient swarms (used inside ``global_evaluate_nd``,
checkpoint reads, mesh-adapt transfers) are actually freed on garbage
collection instead of accumulating in the model registry forever.

  - ``Swarm.__del__`` now calls ``self.dm.destroy()`` so the PETSc DMSwarm and
    every registered field are released when the swarm is collected.
  - ``Swarm._invalidate_canonical_data()`` consolidates the cache invalidation
    that previously appeared inline in two places.
  - ``Model._swarms`` is a ``WeakValueDictionary`` so that registration no
    longer pins swarms beyond the user's last strong reference.
  - ``Model._unregister_swarm`` drops the swarm's registered variables when
    ``__del__`` fires.
"""

import gc

import pytest

import underworld3 as uw
from underworld3.meshing import UnstructuredSimplexBox

pytestmark = pytest.mark.level_1


@pytest.fixture
def mesh():
    return UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=1.0 / 16.0,
    )


def test_swarm_del_calls_dm_destroy():
    """The __del__ source contains the dm.destroy() call (§1a).

    petsc4py exposes DM methods as read-only Cython attributes, so we cannot
    monkeypatch ``dm.destroy`` to spy on it. This is a source-level smoke
    test that guards against accidental removal of the destroy line.
    """
    import inspect
    src = inspect.getsource(uw.swarm.Swarm.__del__)
    assert "dm.destroy" in src, (
        "Swarm.__del__ no longer contains dm.destroy() — §1a regression"
    )


def test_swarm_invalidate_canonical_data_method(mesh):
    """The §1b extraction is callable and clears the expected caches."""
    swarm = uw.swarm.Swarm(mesh)
    var = uw.swarm.SwarmVariable(
        "test_var", swarm,
        vtype=uw.VarType.SCALAR, dtype=float, _proxy=False,
    )
    swarm.populate(fill_param=2)
    # Touch caches so they exist
    _ = swarm._particle_coordinates.array
    _ = var.array

    # Set caches manually, then invalidate
    swarm._particle_coordinates._canonical_data = "fake_cache"
    var._canonical_data = "fake_cache"
    swarm._invalidate_canonical_data()
    assert swarm._particle_coordinates._canonical_data is None
    assert var._canonical_data is None

    # Cleanup
    uw.get_default_model()._unregister_swarm(swarm)


def test_model_unregister_swarm_drops_swarm_and_variables(mesh):
    """``Model._unregister_swarm`` releases the swarm registry slot and any
    registered SwarmVariables that belong to it.

    This is the building block consumer recipes use to ensure transient
    swarms (used in checkpoint reads, mesh-adapt transfers, etc.) actually
    get freed instead of accumulating in ``Model._swarms``.
    """
    model = uw.get_default_model()
    n_swarms_before = len(model._swarms)
    n_vars_before = len(model._variables)

    swarm = uw.swarm.Swarm(mesh)
    uw.swarm.SwarmVariable(
        "_test_lifecycle_a", swarm,
        vtype=uw.VarType.MATRIX, size=(1, 3),
        dtype=float, _proxy=False,
    )
    uw.swarm.SwarmVariable(
        "_test_lifecycle_b", swarm,
        vtype=uw.VarType.SCALAR, dtype=float, _proxy=False,
    )

    # Both registries grew
    assert len(model._swarms) == n_swarms_before + 1
    assert len(model._variables) >= n_vars_before + 2

    model._unregister_swarm(swarm)

    # Both registries shrank back
    assert len(model._swarms) == n_swarms_before, (
        "swarm not removed from Model._swarms"
    )
    # The two test variables are gone (other internal vars from the swarm —
    # coord var, _remeshed if any — are also gone).
    assert "_test_lifecycle_a" not in model._variables
    assert "_test_lifecycle_b" not in model._variables


def test_model_unregister_idempotent(mesh):
    """Calling ``_unregister_swarm`` twice is safe (no KeyError)."""
    model = uw.get_default_model()
    swarm = uw.swarm.Swarm(mesh)
    model._unregister_swarm(swarm)
    model._unregister_swarm(swarm)  # must not raise


def test_swarm_del_fires_on_drop(mesh):
    """A swarm with no remaining strong references is collected automatically.

    Before ``Model._swarms`` became a ``WeakValueDictionary`` the registry
    pinned every swarm forever, so ``__del__`` never fired. The proof that
    the cycle is broken: registry size returns to baseline after ``del``.
    """
    model = uw.get_default_model()
    n_before = len(model._swarms)

    swarm = uw.swarm.Swarm(mesh)
    uw.swarm.SwarmVariable(
        "_test_drop", swarm,
        vtype=uw.VarType.SCALAR, dtype=float, _proxy=False,
    )
    swarm.populate(fill_param=2)
    assert len(model._swarms) == n_before + 1

    del swarm
    gc.collect()

    assert len(model._swarms) == n_before, (
        "Swarm not collected after del — the registry is still pinning it"
    )
    assert "_test_drop" not in model._variables


def test_swarm_lifecycle_does_not_leak(mesh):
    """End-to-end: many transient swarms do not grow current RSS unboundedly.

    Uses ``psutil.Process().memory_info().rss`` (current resident memory) so
    OS-level peak tracking does not mask freed memory. The threshold is
    deliberately generous — a real leak grows by tens of MB per 100 swarms.
    """
    psutil = pytest.importorskip("psutil")
    import os
    p = psutil.Process(os.getpid())

    def rss_mb():
        return p.memory_info().rss / (1024 * 1024)

    n_iters = 500
    sample_every = 100

    # Warm-up: first iterations always cost RSS for one-off allocations
    # (caches, JIT tables, lazy initialisation).
    for _ in range(50):
        s = uw.swarm.Swarm(mesh)
        uw.swarm.SwarmVariable(
            "v", s,
            vtype=uw.VarType.MATRIX, size=(1, 3),
            dtype=float, _proxy=False,
        )
        s.populate(fill_param=2)
        del s
    gc.collect()
    rss_samples = [rss_mb()]

    for i in range(n_iters):
        s = uw.swarm.Swarm(mesh)
        for j in range(3):
            uw.swarm.SwarmVariable(
                f"v_{j}", s,
                vtype=uw.VarType.MATRIX, size=(1, 3),
                dtype=float, _proxy=False,
            )
        s.populate(fill_param=2)
        for var in s._vars.values():
            _ = var.array
        del s

        if (i + 1) % sample_every == 0:
            gc.collect()
            rss_samples.append(rss_mb())

    growth_mb = rss_samples[-1] - rss_samples[0]
    growth_per_100 = growth_mb / (n_iters / sample_every)
    print(f"\nRSS samples (MB): {[round(x, 1) for x in rss_samples]}")
    print(f"Growth per 100 swarms: {growth_per_100:.2f} MB")
    # Without the WeakValueDictionary fix this comes back as ~25–30 MB / 100
    # (full DMSwarm + registered fields preserved in the registry).
    assert growth_per_100 < 5.0, (
        f"Suspected swarm leak: {growth_per_100:.2f} MB per 100 swarms "
        f"(samples: {rss_samples})"
    )
