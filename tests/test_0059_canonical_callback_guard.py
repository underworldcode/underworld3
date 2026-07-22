"""Regression tests for ``NDArray_With_Callback.add_canonical_callback`` (#379).

``__array_finalize__`` hands the callback list to every array derived by
indexing, so a callback written for the canonical storage also fires on
views and temporary fancy-index copies. Firing on a copy is the #376
parallel hang: the copy's contents are partition-dependent, so a PETSc
sync from inside the callback runs its collectives on some ranks only.

``add_canonical_callback`` centralises the fix: the registered callback
only ever receives the canonical array. View-vs-copy is decided by
IDENTITY in numpy's base chain — ``np.may_share_memory`` is False for any
zero-size array, which would re-create the rank asymmetry on ranks whose
local slice is empty.
"""

import gc
import weakref

import numpy as np
import pytest

from underworld3.utilities import NDArray_With_Callback

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


class _CallRecorder:
    """Records every (array, change_info) pair a callback receives."""

    def __init__(self):
        self.calls = []

    def __call__(self, array, change_info):
        self.calls.append((array, change_info))


def _canonical(shape=(6, 2)):
    arr = NDArray_With_Callback(np.zeros(shape))
    recorder = _CallRecorder()
    arr.add_canonical_callback(recorder)
    return arr, recorder


def test_write_to_canonical_fires_with_canonical():
    arr, recorder = _canonical()
    arr[2, 0] = 1.0
    assert len(recorder.calls) == 1
    assert recorder.calls[0][0] is arr


def test_write_through_basic_slice_view_fires_with_full_canonical():
    """A basic slice is a view: the write already landed in canonical
    storage, so the callback must receive the FULL canonical array."""
    arr, recorder = _canonical()
    view = arr[1:4]
    view[0, 0] = 3.0
    assert len(recorder.calls) == 1
    assert recorder.calls[0][0] is arr
    assert arr[1, 0] == 3.0


def test_nested_view_chain_still_resolves_to_canonical():
    arr, recorder = _canonical()
    inner = arr[1:5][1:3]
    inner[0, 0] = 7.0
    assert len(recorder.calls) == 1
    assert recorder.calls[0][0] is arr


def test_zero_size_view_is_still_classified_as_view():
    """The empty-rank case: np.may_share_memory(empty_view, arr) is False,
    but the base-chain walk must still classify it as a view so every rank
    takes the same branch."""
    arr, recorder = _canonical()
    empty_view = arr[3:3]
    empty_view[...] = 9.0  # writes nothing, but fires __setitem__
    assert len(recorder.calls) == 1
    assert recorder.calls[0][0] is arr


def test_fancy_index_write_fires_exactly_once_via_parent_writeback():
    """``arr[mask] += delta`` creates a temporary COPY, fires the inherited
    callback on it (skipped), then numpy writes back through the parent's
    ``__setitem__`` (fires with canonical). Net: exactly one call, on the
    canonical array — this is the #376 scenario."""
    arr, recorder = _canonical()
    mask = np.zeros(arr.shape[0], dtype=bool)
    mask[1] = mask[4] = True
    arr[mask] += 1.0
    assert len(recorder.calls) == 1
    assert recorder.calls[0][0] is arr
    assert arr[1, 0] == 1.0 and arr[4, 0] == 1.0


def test_detached_copy_never_fires():
    """A standalone copy has left the canonical base chain entirely; writes
    to it must not reach the canonical callback."""
    arr, recorder = _canonical()
    detached = arr[np.array([0, 2])]  # fancy index → independent copy
    recorder.calls.clear()  # discard the write-back-free extraction call, if any
    detached[0, 0] = 5.0
    assert recorder.calls == []
    assert arr[0, 0] == 0.0


def test_no_reference_cycle_with_canonical():
    """The dispatch closure must hold the canonical array weakly: the
    callback list lives ON the array, so a strong capture would keep the
    array alive forever."""
    arr = NDArray_With_Callback(np.zeros((4, 1)))
    arr.add_canonical_callback(lambda a, info: None)
    ref = weakref.ref(arr)
    del arr
    gc.collect()
    assert ref() is None
