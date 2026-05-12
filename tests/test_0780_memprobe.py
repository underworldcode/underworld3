"""
Smoke tests for the memprobe diagnostic module.

These tests check the *plumbing*: instrumentation enable/disable, snapshot
shape, KDTree live-count tracking, decorator no-op-when-disabled, and
context-manager diff emission. They do not validate that any real leak is
or isn't present — that's what users do with the tool, not what the tool
itself can verify.
"""
import gc
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
    """KDTree __cinit__/__dealloc__ must update the live-instance counter.

    The test asserts only on deltas around the operation under test, never
    on absolute counts, because earlier tests in the same pytest session
    can leave KDTree references alive that the cyclic GC may collect at
    any time, shifting the baseline.
    """
    pts = np.random.random((20, 2))

    # Flush any pending cyclic-GC clean-ups so the baseline doesn't shift
    # under us between snapshots.
    gc.collect()
    before_live = uw.kdtree.live_count()
    before_total = uw.kdtree.total_constructed()

    tree = uw.kdtree.KDTree(pts)
    assert uw.kdtree.live_count() - before_live == 1
    assert uw.kdtree.total_constructed() - before_total == 1

    after_create_live = uw.kdtree.live_count()

    del tree
    gc.collect()

    # The only thing we care about: this construction/destruction pair
    # produced a +1/-1 swing relative to its immediate before/after, and
    # total_constructed never went backwards.
    assert uw.kdtree.live_count() - after_create_live == -1
    assert uw.kdtree.total_constructed() >= before_total + 1


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


@pytest.mark.level_1
@pytest.mark.tier_a
def test_mesh_variable_kdtree_caching():
    """MeshVariable._get_kdtree must cache the tree and reuse it."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("v", mesh, 1)

    gc.collect()
    before_total = uw.kdtree.total_constructed()

    # First access builds
    kd1 = v._get_kdtree()
    assert uw.kdtree.total_constructed() - before_total == 1

    # Second access reuses
    kd2 = v._get_kdtree()
    assert uw.kdtree.total_constructed() - before_total == 1
    assert kd1 is kd2

    # Mesh deformation/version change invalidates
    mesh._mesh_version += 1
    kd3 = v._get_kdtree()
    assert uw.kdtree.total_constructed() - before_total == 2
    assert kd3 is not kd1


@pytest.mark.level_1
@pytest.mark.tier_a
def test_swarm_kdtree_caching():
    """Swarm._get_kdtree must cache the tree and reuse it."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=1)

    gc.collect()
    before_total = uw.kdtree.total_constructed()

    # First access builds
    kd1 = swarm._get_kdtree()
    assert uw.kdtree.total_constructed() - before_total == 1

    # Second access reuses
    kd2 = swarm._get_kdtree()
    assert uw.kdtree.total_constructed() - before_total == 1
    assert kd1 is kd2

    # Migration/Invalidation should drop the cache
    swarm._invalidate_canonical_data()
    kd3 = swarm._get_kdtree()
    assert uw.kdtree.total_constructed() - before_total == 2
    assert kd3 is not kd1
