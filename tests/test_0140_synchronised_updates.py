"""Serial semantics of ``uw.synchronised_array_update`` after the #379 rework.

The old machinery queued every (callback, array, change_info) event per rank
and replayed the queue at context exit. Queue contents were rank-dependent
while the callbacks contain collectives, so rank-uneven writes desynchronised
the collective PETSc sync (the parallel counterpart is tested in
tests/parallel/test_0758_synchronised_update_collective_flush.py).

The rework marks each touched variable dirty and flushes it ONCE at context
exit — same variables, same order, on every rank. These tests pin the serial
contract: writes land immediately in the canonical array, canonical callbacks
stay silent during the context, and each dirty variable gets exactly one
``deferred_flush`` notification at exit.
"""

import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import NDArray_With_Callback

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture()
def mesh():
    return uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )


class _CallRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, array, change_info):
        self.calls.append(change_info["operation"])


def test_values_land_after_exit(mesh):
    u = uw.discretisation.MeshVariable("u140a", mesh, 2, vtype=uw.VarType.VECTOR, degree=2)
    p = uw.discretisation.MeshVariable("p140a", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)

    u_vals = np.random.random(u.array.shape)
    p_vals = np.random.random(p.array.shape)
    with uw.synchronised_array_update("multi-variable update"):
        u.array[...] = u_vals
        p.array[...] = p_vals

    assert np.allclose(np.asarray(u.array), u_vals)
    assert np.allclose(np.asarray(p.array), p_vals)


def test_one_flush_per_variable_at_exit(mesh):
    """N writes to one variable inside the context produce exactly ONE
    canonical-callback firing, at exit, tagged 'deferred_flush'."""
    t = uw.discretisation.MeshVariable("t140b", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)
    probe = _CallRecorder()
    t.data.add_canonical_callback(probe)

    with uw.synchronised_array_update("batched writes"):
        t.array[...] = 1.0
        t.array[:, 0, 0] = 2.0
        t.array[...] = 3.0
        assert probe.calls == []  # silent while the context is open

    assert probe.calls == ["deferred_flush"]
    assert np.allclose(np.asarray(t.array), 3.0)


def test_masked_write_inside_context_flushes_canonically(mesh):
    """The #376 scenario under the delay context: a fancy-index write marks
    the CANONICAL variable dirty (via the parent write-back), never the
    temporary copy."""
    t = uw.discretisation.MeshVariable("t140c", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)
    t.array[...] = 1.0
    probe = _CallRecorder()
    t.data.add_canonical_callback(probe)

    mask = np.zeros(t.data.shape[0], dtype=bool)
    mask[::2] = True
    with uw.synchronised_array_update():
        t.data[mask] += 1.0

    assert probe.calls == ["deferred_flush"]
    flat = np.asarray(t.data)[:, 0]
    assert np.allclose(flat[mask], 2.0) and np.allclose(flat[~mask], 1.0)


def test_nested_contexts_flush_per_level(mesh):
    t_outer = uw.discretisation.MeshVariable("t140d", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)
    t_inner = uw.discretisation.MeshVariable("s140d", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)
    probe_outer, probe_inner = _CallRecorder(), _CallRecorder()
    t_outer.data.add_canonical_callback(probe_outer)
    t_inner.data.add_canonical_callback(probe_inner)

    with uw.synchronised_array_update("outer"):
        t_outer.array[...] = 1.0
        with uw.synchronised_array_update("inner"):
            t_inner.array[...] = 2.0
        assert probe_inner.calls == ["deferred_flush"]  # inner flushed at ITS exit
        assert probe_outer.calls == []
    assert probe_outer.calls == ["deferred_flush"]


def test_exception_skips_flush_but_keeps_values(mesh):
    """Ranks unwind exceptions asymmetrically, so the exit flush (which is
    collective) must not run during unwinding. Values already landed in the
    canonical array; only the deferred synchronisation is skipped."""
    t = uw.discretisation.MeshVariable("t140e", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)
    probe = _CallRecorder()
    t.data.add_canonical_callback(probe)

    with pytest.raises(RuntimeError, match="deliberate"):
        with uw.synchronised_array_update():
            t.array[...] = 7.0
            raise RuntimeError("deliberate")

    assert probe.calls == []
    assert np.allclose(np.asarray(t.array), 7.0)


def test_untagged_callbacks_keep_per_event_replay():
    """Plain add_callback() users (no canonical guard) keep the legacy
    per-event queue. Documented rank-local; must not contain collectives."""
    arr = NDArray_With_Callback(np.zeros(4))
    probe = _CallRecorder()
    arr.add_callback(probe)

    with uw.synchronised_array_update():
        arr[0] = 1.0
        arr[1] = 2.0
        assert probe.calls == []

    assert probe.calls == ["setitem", "setitem"]
