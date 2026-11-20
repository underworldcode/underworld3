"""addon_partition.partition_addon agrees on failure (#874).

``partition_addon`` runs inside one ``agreed_build`` context
(``utilities.agreed_build``, #1218): a rank-local exception on one rank is raised
on every rank as the same ``CollectiveBuildError`` (a ``RuntimeError``; not a
builder failure, since this is not a transfer build), at the next agreement point
or at the partition's end. Its body has an agreement point before each
collective it makes, and the array reduction it uses (``_allreduce``) agrees at
its start; the helpers it calls keep their own guards for now.

The add-on of an adapt child is partitioned again with one failure forced on the
last rank just after a helper has returned (so after the helper's own
collectives): the parent location, the region labelling, the region merge, the
balance, the redistribution, the moved mesh's construction and the parent
resolution. Each must be the same error on every rank, with the agreement calls
per rank paired. Before the conversion each hangs the other ranks.
"""
import numpy as np
import pytest

import underworld3 as uw
import underworld3.utilities.fault_split as fault_split
import underworld3.utilities.place_surface as place_surface
from fault_fixtures import TRACE, adapted_2d
from underworld3.utilities import addon_partition as ap
from underworld3.utilities.custom_mg import CollectiveBuildError

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(600, method="thread")]

K = 2


class _CountingVerdicts:
    def __init__(self, comm):
        self._comm = comm
        self.exchanges = 0

    def allreduce(self, *a, **k):
        self.exchanges += 1
        return self._comm.allreduce(*a, **k)

    def allgather(self, *a, **k):
        self.exchanges += 1
        return self._comm.allgather(*a, **k)

    def __getattr__(self, name):
        return getattr(self._comm, name)


def _on_last_rank(message):
    if uw.mpi.rank == uw.mpi.size - 1:
        raise IndexError(f"{message} (forced on one rank)")


def _partition(child):
    return ap.partition_addon(child, ap.mark_polylines([TRACE]), layers=1, tolerance=0.1)


def _raises_everywhere(call, message, monkeypatch):
    counter = _CountingVerdicts(uw.mpi.verdict_comm)
    monkeypatch.setattr(uw.mpi, "verdict_comm", counter)
    caught = None
    try:
        call()
    except RuntimeError as err:
        assert isinstance(err, CollectiveBuildError), type(err)
        assert not err.builder_failure, "a partition failure is not a builder's"
        caught = str(err)
    finally:
        monkeypatch.setattr(uw.mpi, "verdict_comm", counter._comm)
    messages = uw.mpi.comm.allgather(caught)
    counts = uw.mpi.comm.allgather(counter.exchanges)
    assert all(m is not None and message in m for m in messages), messages
    assert len(set(messages)) == 1, messages
    assert len(set(counts)) == 1, counts


def _after(module, name, message, monkeypatch):
    original = getattr(module, name)

    def failing(*a, **k):
        out = original(*a, **k)
        _on_last_rank(message)
        return out

    monkeypatch.setattr(module, name, failing)


def test_the_partition_still_moves_the_add_on(monkeypatch):
    child = adapted_2d(K)
    counter = _CountingVerdicts(uw.mpi.verdict_comm)
    monkeypatch.setattr(uw.mpi, "verdict_comm", counter)
    try:
        out = _partition(child)
    finally:
        monkeypatch.setattr(uw.mpi, "verdict_comm", counter._comm)
    assert out is not child and out._addon_partition["units_moved"] > 0
    counts = uw.mpi.comm.allgather(counter.exchanges)
    assert len(set(counts)) == 1 and counts[0] > 0, counts


@pytest.mark.parametrize("module, name", [
    (ap, "_parent_cells_local"),
    (place_surface, "_region_cells"),
    (place_surface, "_merge_regions"),
    (ap, "_peel_balance"),
    (fault_split, "redistribute_by_assignment"),
    (ap, "_moved_wrap"),
    (ap, "_resolve_parents"),
], ids=["parent-location", "region-cells", "region-merge", "balance",
        "redistribution", "moved-mesh", "parent-resolution"])
def test_a_failure_after_a_helper_is_raised_on_every_rank(module, name, monkeypatch):
    child = adapted_2d(K)
    message = f"After {name}"
    _after(module, name, message, monkeypatch)
    _raises_everywhere(lambda: _partition(child), message, monkeypatch)
