"""The ``uw.selective_ranks()`` contract the parallel guide documents (#837).

Every rank runs the ``with`` body, and the yielded flag selects the ranks. The
documentation once showed the bare ``with uw.selective_ranks(0):`` as
rank-0-only; code that followed it had every rank write the same file. These
tests pin the contract, and the collective guard built on the same selection:
it reports a marked collective when the selection misses a rank, and only then.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

SIZE, RANK = uw.mpi.size, uw.mpi.rank

# selector -> the ranks it must select, written out independently of the library
SELECTORS = {
    "int 0": (0, {0}),
    "numpy int 0": (np.int64(0), {0}),
    "slice": (slice(0, 1), {0}),
    "list": ([0], {0}),
    "tuple": ((0,), {0}),
    "all": ("all", set(range(SIZE))),
    "None": (None, set(range(SIZE))),
    "first": ("first", {0}),
    "last": ("last", {SIZE - 1}),
    "even": ("even", set(range(0, SIZE, 2))),
    "odd": ("odd", set(range(1, SIZE, 2))),
    "100%": ("100%", set(range(SIZE))),
    "callable": (lambda r: r == 0, {0}),
    "bool mask": (np.array([True] + [False] * (SIZE - 1)), {0}),
    "short bool mask": (np.array([True]), {0}),
    "int array": (np.arange(SIZE), set(range(SIZE))),
    "superset list": (list(range(SIZE + 1)), set(range(SIZE))),
    "shifted list": (list(range(1, SIZE + 1)), set(range(1, SIZE))),
    "empty": ([], set()),
}


@pytest.mark.parametrize("name", list(SELECTORS))
def test_the_flag_and_the_guard_agree_on_the_selection(name):
    selector, expected = SELECTORS[name]
    with uw.selective_ranks(selector) as should_execute:
        assert should_execute == (RANK in expected)
        assert uw.mpi._selective_executing_ranks == expected


def _variable(name):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    return uw.discretisation.MeshVariable(name, mesh, 1, degree=1)


@pytest.mark.parametrize("selector", [[], list(range(1, SIZE + 1))],
                         ids=["empty", "shifted, full count"])
def test_a_marked_collective_under_a_partial_selection_raises(selector):
    var = _variable("T_partial")
    with uw.selective_ranks(selector):
        with pytest.raises(uw.CollectiveOperationError):
            var.stats()


@pytest.mark.parametrize("selector", ["all", None, slice(None), list(range(SIZE + 1))],
                         ids=["all", "None", "slice(None)", "superset"])
def test_a_marked_collective_under_a_full_selection_runs(selector):
    var = _variable("T_full")
    with uw.selective_ranks(selector):
        var.stats()


def test_bad_selectors_are_refused():
    with pytest.raises(ValueError):
        with uw.selective_ranks("rank zero"):
            pass
    with pytest.raises(TypeError):
        with uw.selective_ranks(0.5):
            pass


def test_pprint_accepts_a_boolean_mask():
    uw.pprint("mask", proc=np.array([True, False, True, False]))
