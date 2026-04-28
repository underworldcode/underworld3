"""Lifecycle regression tests for Swarm cleanup.

The prep work for the global-point-routing redesign adds three things:

  - ``Swarm.__del__`` now calls ``self.dm.destroy()`` so the PETSc DMSwarm and
    its registered fields free when ``__del__`` actually runs.
  - ``Swarm._invalidate_canonical_data()`` consolidates the cache invalidation
    that previously appeared inline in two places.
  - ``Model._unregister_swarm(swarm)`` drops a swarm and any of its registered
    variables from the global model registry. Consumer recipes that build a
    transient swarm (read_timestep, load_from_checkpoint, mesh.adapt transfer)
    must call this before the swarm goes out of scope, otherwise the strong
    reference in ``Model._swarms`` keeps it alive and ``__del__`` never fires.

The end-to-end "RSS does not grow" leak-loop is **not** part of this prep
test set: a fully automatic leak-free swarm requires removing the strong ref
in ``Model._swarms`` (probably ``WeakValueDictionary`` + ``weakref.finalize``
to also auto-drop variables), which is a wider lifecycle redesign than this
prep PR scope. Instead, we test the building blocks individually and trust
the consumer recipes to use ``_unregister_swarm`` explicitly.
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
