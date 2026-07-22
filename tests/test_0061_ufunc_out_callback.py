"""Regression tests for the ``out=`` ufunc callback bypass (#379 item 4a).

``np.add(x, 1, out=x)`` writes straight into the buffer through numpy's
ufunc machinery — no ``__setitem__``, no in-place operator wrapper — so
before this fix no callback fired: values landed (the canonical array
wraps the PETSc vec memory) but ghost synchronisation and the state
increment silently did not happen.

``__array_ufunc__`` now delegates to numpy unchanged and then notifies
each ``out=`` target that is an ``NDArray_With_Callback``. The canonical
guard composes: an ``out=`` view resolves to the canonical array, an
``out=`` detached copy is skipped, and inside
``uw.synchronised_array_update`` the write dirty-marks for the single
deferred flush.

Known remaining bypasses (documented, not intercepted): ``np.copyto``
and ``ufunc.at`` — neither routes through ``__array_ufunc__``'s ``out=``.
"""

import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import NDArray_With_Callback

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


class _CallRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, array, change_info):
        self.calls.append(change_info["operation"])


def _canonical(shape=(6, 2)):
    arr = NDArray_With_Callback(np.zeros(shape))
    recorder = _CallRecorder()
    arr.add_canonical_callback(recorder)
    return arr, recorder


def test_ufunc_out_on_canonical_fires():
    arr, recorder = _canonical()
    np.add(arr, 1.0, out=arr)
    assert recorder.calls == ["ufunc_out"]
    assert np.allclose(np.asarray(arr), 1.0)


def test_ufunc_out_on_view_resolves_to_canonical():
    arr, recorder = _canonical()
    view = arr[1:4]
    np.multiply(view, 0.0, out=view)
    assert recorder.calls == ["ufunc_out"]


def test_ufunc_out_on_detached_copy_is_skipped():
    arr, recorder = _canonical()
    detached = arr[np.array([0, 2])]  # fancy index → independent copy
    recorder.calls.clear()
    np.add(detached, 5.0, out=detached)
    assert recorder.calls == []
    assert np.allclose(np.asarray(arr), 0.0)


def test_plain_ufunc_without_out_stays_silent():
    arr, recorder = _canonical()
    result = arr + 1.0
    assert recorder.calls == []
    assert not isinstance(result, NDArray_With_Callback)
    assert np.allclose(result, 1.0)


def test_inplace_operator_fires_exactly_once():
    """+= routes through the ufunc machinery with out=self; the explicit
    per-operator triggers are gone, so exactly ONE notification arrives."""
    arr, recorder = _canonical()
    arr += 2.0
    assert recorder.calls == ["ufunc_out"]
    assert np.allclose(np.asarray(arr), 2.0)


def test_masked_inplace_still_fires_exactly_once():
    """arr[mask] += 1: the temporary copy's ufunc_out is skipped (copy),
    the parent write-back fires — net one canonical notification."""
    arr, recorder = _canonical()
    mask = np.zeros(arr.shape[0], dtype=bool)
    mask[1] = True
    arr[mask] += 1.0
    assert recorder.calls == ["setitem"]
    assert arr[1, 0] == 1.0


def test_ufunc_out_defers_inside_synchronised_update():
    arr, recorder = _canonical()
    with uw.synchronised_array_update("ufunc out"):
        np.add(arr, 1.0, out=arr)
        np.add(arr, 1.0, out=arr)
        assert recorder.calls == []
    assert recorder.calls == ["deferred_flush"]
    assert np.allclose(np.asarray(arr), 2.0)


def test_mesh_variable_ufunc_out_reaches_petsc():
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0)
    )
    t = uw.discretisation.MeshVariable("t61", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)
    t.data[...] = 2.0
    np.multiply(t.data, 3.0, out=t.data)
    assert np.allclose(np.asarray(t.data), 6.0)
    assert np.allclose(t.vec.array, 6.0)


def test_out_honours_disable_inplace_operators():
    """np.add(x, 1, out=x) must respect the same contract as x += 1 —
    bypassing the flag re-armed the per-write hazard it exists to prevent
    (#379 review round 1)."""
    arr = NDArray_With_Callback(np.zeros(4), disable_inplace_operators=True)
    with pytest.raises(RuntimeError, match="parallel safety"):
        np.add(arr, 1.0, out=arr)
    with pytest.raises(RuntimeError, match="parallel safety"):
        arr += 1.0
    assert np.allclose(np.asarray(arr), 0.0)
