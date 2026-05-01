"""WorkflowRunner — smoke tests on a toy workflow module.

Tests the cache → disk → build resolution chain and produces/requires
matching without depending on UW3 solvers.  A fully synthetic toy
module is built at import time using ``types.ModuleType`` so the
tests are fast and deterministic.
"""

import types

import pytest

from underworld3.workflows import (
    WorkflowConfig,
    WorkflowRunner,
    workflow_step,
)


pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _make_toy_module(call_log):
    """Build a tiny in-memory workflow module.

    Steps form a small DAG::

        a ──┐
            ├── c ── d
        b ──┘
    """
    mod = types.ModuleType("toy_workflow")

    @workflow_step(produces=["a"])
    def make_a(config):
        call_log.append("make_a")
        return config.x + 1

    @workflow_step(produces=["b"])
    def make_b(config):
        call_log.append("make_b")
        return config.x + 2

    @workflow_step(produces=["c"], requires=["a", "b"])
    def make_c(a, b, config):
        call_log.append("make_c")
        return a * b + config.x

    @workflow_step(produces=["d"], requires=["c"])
    def make_d(c, config):
        call_log.append("make_d")
        return c * 10

    mod.make_a = make_a
    mod.make_b = make_b
    mod.make_c = make_c
    mod.make_d = make_d
    return mod


class _ToyConfig(WorkflowConfig):
    x: int = 5


def test_runner_resolves_dependencies():
    """Building a leaf product runs every dependency in order."""
    log = []
    mod = _make_toy_module(log)
    runner = WorkflowRunner(mod, _ToyConfig(), products=None)

    # x = 5 → a = 6, b = 7, c = 6*7+5 = 47, d = 470
    assert runner.build("d") == 470
    # a and b can run in either order; c before d.
    assert log.index("make_c") < log.index("make_d")
    assert "make_a" in log and "make_b" in log
    assert log.index("make_a") < log.index("make_c")
    assert log.index("make_b") < log.index("make_c")


def test_runner_caches_intermediates():
    """A second build call should not re-run any step."""
    log = []
    mod = _make_toy_module(log)
    runner = WorkflowRunner(mod, _ToyConfig(), products=None)

    runner.build("d")
    n_first = len(log)

    runner.build("d")  # everything cached
    assert len(log) == n_first

    runner.build("c")  # already cached
    assert len(log) == n_first


def test_runner_rebuild_invalidates_target_only():
    """rebuild() reruns the target but not its (now-cached) dependencies."""
    log = []
    mod = _make_toy_module(log)
    runner = WorkflowRunner(mod, _ToyConfig(), products=None)

    runner.build("d")
    log.clear()

    runner.rebuild("d")
    # Only d should rerun; a, b, c are still in the cache.
    assert log == ["make_d"]


def test_runner_unknown_product_raises():
    log = []
    mod = _make_toy_module(log)
    runner = WorkflowRunner(mod, _ToyConfig(), products=None)
    with pytest.raises(KeyError):
        runner.build("nonexistent")


def test_runner_build_all_returns_leaves():
    log = []
    mod = _make_toy_module(log)
    runner = WorkflowRunner(mod, _ToyConfig(), products=None)
    leaves = runner.build_all()
    # 'd' is the only product nothing else requires.
    assert leaves == ["d"]


def test_runner_status():
    log = []
    mod = _make_toy_module(log)
    runner = WorkflowRunner(mod, _ToyConfig(), products=None)

    assert runner.status("a") == "missing"
    runner.build("a")
    assert runner.status("a") == "cached"


def test_multi_produce_dict_return():
    """A step returning a dict aligned with produces caches each entry."""
    mod = types.ModuleType("multi")

    @workflow_step(produces=["x", "y"])
    def make_xy(config):
        return {"x": 1, "y": 2}

    mod.make_xy = make_xy
    runner = WorkflowRunner(mod, _ToyConfig(), products=None)
    assert runner.build("x") == 1
    assert runner.build("y") == 2  # cached, no re-run


def test_multi_produce_tuple_return():
    """A step returning a tuple is mapped position-wise to produces."""
    mod = types.ModuleType("multi_tuple")

    @workflow_step(produces=["x", "y"])
    def make_xy(config):
        return (10, 20)

    mod.make_xy = make_xy
    runner = WorkflowRunner(mod, _ToyConfig(), products=None)
    assert runner.build("x") == 10
    assert runner.build("y") == 20


def test_multi_produce_arity_mismatch_raises():
    mod = types.ModuleType("bad_arity")

    @workflow_step(produces=["x", "y"])
    def make_xy(config):
        return (1, 2, 3)  # wrong arity

    mod.make_xy = make_xy
    runner = WorkflowRunner(mod, _ToyConfig(), products=None)
    with pytest.raises(ValueError):
        runner.build("x")


def test_runner_persists_persistable_products(tmp_path):
    """Persistable products (np.ndarray) survive a fresh runner instance."""
    import numpy as np
    from underworld3.workflows import WorkflowProducts

    log = []
    mod = types.ModuleType("persist")

    @workflow_step(produces=["arr"])
    def make_arr(config):
        log.append("make_arr")
        return np.array([1, 2, 3])

    mod.make_arr = make_arr

    config = _ToyConfig(output_dir=str(tmp_path))
    products = WorkflowProducts(config)

    runner1 = WorkflowRunner(mod, config, products=products)
    arr1 = runner1.build("arr")
    assert "make_arr" in log

    # Fresh runner on the same products dir → loads from disk, no rebuild.
    log.clear()
    runner2 = WorkflowRunner(mod, config, products=products)
    arr2 = runner2.build("arr")
    assert log == []
    assert (arr1 == arr2).all()


def test_runner_invalidate_removes_disk_product(tmp_path):
    """invalidate() removes both cache and persisted product."""
    import numpy as np
    from underworld3.workflows import WorkflowProducts

    mod = types.ModuleType("inv")

    @workflow_step(produces=["arr"])
    def make_arr(config):
        return np.array([1, 2, 3])

    mod.make_arr = make_arr

    config = _ToyConfig(output_dir=str(tmp_path))
    products = WorkflowProducts(config)

    runner = WorkflowRunner(mod, config, products=products)
    runner.build("arr")
    assert products.exists("arr")

    runner.invalidate("arr")
    assert not products.exists("arr")
    assert "arr" not in runner.cache


def test_duplicate_producer_is_rejected():
    """Two steps producing the same name should fail at runner construction."""
    mod = types.ModuleType("dup")

    @workflow_step(produces=["x"])
    def first(config):
        return 1

    @workflow_step(produces=["x"])
    def second(config):
        return 2

    mod.first = first
    mod.second = second
    with pytest.raises(ValueError, match="produce"):
        WorkflowRunner(mod, _ToyConfig(), products=None)
