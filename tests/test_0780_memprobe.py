"""
Smoke tests for the memprobe diagnostic module.

These tests check the *plumbing*: instrumentation enable/disable, snapshot
shape, KDTree live-count tracking, decorator no-op-when-disabled, and
context-manager diff emission. They do not validate that any real leak is
or isn't present — that's what users do with the tool, not what the tool
itself can verify.
"""
import gc
import os
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import memprobe


@pytest.fixture(autouse=True)
def _restore_enabled_flag():
    """Each test starts with memprobe disabled regardless of env."""
    saved = memprobe.ENABLED
    memprobe.disable()
    yield
    memprobe.ENABLED = saved


@pytest.mark.level_1
@pytest.mark.tier_a
def test_snapshot_shape():
    """Snapshot must contain rss_mb (float) and kdtree (dict with two keys)."""
    snap = memprobe.snapshot()

    assert isinstance(snap["rss_mb"], float)
    assert snap["rss_mb"] > 0
    assert set(snap["kdtree"].keys()) == {"live", "total_constructed"}

    # full=False should NOT walk gc
    assert "py_classes" not in snap


@pytest.mark.level_1
@pytest.mark.tier_a
def test_snapshot_full_includes_py_classes():
    snap = memprobe.snapshot(full=True)
    assert "py_classes" in snap
    # Sanity: gc walked something
    assert isinstance(snap["py_classes"], dict)


@pytest.mark.level_1
@pytest.mark.tier_a
def test_kdtree_live_count_tracks_construction_destruction():
    """KDTree __cinit__/__dealloc__ must update the live-instance counter."""
    pts = np.random.random((20, 2))

    before = uw.kdtree.live_count()
    total_before = uw.kdtree.total_constructed()

    tree = uw.kdtree.KDTree(pts)
    assert uw.kdtree.live_count() == before + 1
    assert uw.kdtree.total_constructed() == total_before + 1

    del tree
    gc.collect()
    assert uw.kdtree.live_count() == before
    # total_constructed never decreases
    assert uw.kdtree.total_constructed() == total_before + 1


@pytest.mark.level_1
@pytest.mark.tier_a
def test_diff_reports_kdtree_growth():
    pts = np.random.random((20, 2))

    before = memprobe.snapshot()
    tree = uw.kdtree.KDTree(pts)  # noqa: F841 — kept alive across the diff
    after = memprobe.snapshot()

    delta = memprobe.diff(before, after)
    assert delta["kdtree"]["live"] == 1
    assert delta["kdtree"]["total_constructed"] == 1


@pytest.mark.level_1
@pytest.mark.tier_a
def test_diff_drops_unchanged_keys():
    snap = memprobe.snapshot()
    delta = memprobe.diff(snap, snap)
    # Identical snapshots produce no deltas
    assert "kdtree" not in delta


@pytest.mark.level_1
@pytest.mark.tier_a
def test_probe_context_emits_when_enabled(capsys):
    memprobe.enable()
    with memprobe.probe("test-block"):
        _ = uw.kdtree.KDTree(np.random.random((10, 2)))

    captured = capsys.readouterr()
    assert "[memprobe] test-block" in captured.out
    assert "kdtree" in captured.out


@pytest.mark.level_1
@pytest.mark.tier_a
def test_probe_context_silent_when_disabled(capsys):
    assert memprobe.ENABLED is False
    with memprobe.probe("silent"):
        _ = uw.kdtree.KDTree(np.random.random((10, 2)))

    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.level_1
@pytest.mark.tier_a
def test_instrument_decorator_no_op_when_disabled(capsys):
    """Decorator must not emit (or noticeably slow) when ENABLED is False."""
    @memprobe.instrument("test-fn")
    def f(x):
        return x * 2

    assert f(3) == 6
    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.level_1
@pytest.mark.tier_a
def test_instrument_decorator_emits_when_enabled(capsys):
    @memprobe.instrument("test-fn")
    def f(x):
        return x * 2

    memprobe.enable()
    assert f(3) == 6
    captured = capsys.readouterr()
    assert "[memprobe] test-fn" in captured.out
