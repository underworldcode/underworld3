"""Rank-local failures in the provenance transfer and the add-on partition's
parent relation are collective (#874, the remainder).

* ``fault_mg.provenance_transfer_parallel`` weights each owned fine node in its
  recorded parent (``_node_weights``) and, when the record is on another mesh,
  locates the fine cells in it. A failure on one rank raised there alone while
  the other ranks waited in the next allreduce.
* ``addon_partition._parent_cells_local`` locates each fine cell's centroid in
  the level below. A failure on one rank raised there alone while the other
  ranks waited in the allreduce of the unlocated count.

Each is now raised on every rank, the same error, through
``fault_contact._collective_raise``. Each test forces the failure on the last
rank only. Before the guard, the other ranks hang until the timeout.
"""
import numpy as np
import pytest

import underworld3 as uw
from fault_fixtures import TRACE, adapted_2d, shear_fault_solve
from underworld3.utilities import addon_partition as ap
from underworld3.utilities import fault_mg
from underworld3.utilities.fault_mg import addon_level_indices

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(600, method="thread")]

K = 2


def _fail_on_last_rank(name, message):
    """Replace ``fault_mg.<name>`` with a version that raises on the last rank."""
    original = getattr(fault_mg, name)

    def failing(*a, **k):
        if uw.mpi.rank == uw.mpi.size - 1:
            raise np.linalg.LinAlgError(f"{message} (forced on one rank)")
        return original(*a, **k)

    setattr(fault_mg, name, failing)
    return original


def _same_error_everywhere(call, name, message):
    original = _fail_on_last_rank(name, message)
    try:
        caught = None
        try:
            call()
        except RuntimeError as err:
            caught = str(err)
    finally:
        setattr(fault_mg, name, original)
    messages = uw.mpi.comm.allgather(caught)
    assert all(m is not None and message in m for m in messages), messages
    assert len(set(messages)) == 1, messages


def test_parent_location_failure_is_raised_on_every_rank():
    child = adapted_2d(K)
    tail = child._custom_mg_coarse_meshes
    levels = [tail[i] for i in addon_level_indices(tail)] + [child]
    assert len(levels) >= 2
    _same_error_everywhere(lambda: ap._parent_cells_local(levels[0], levels[1]),
                           "locate_in_cells_exhaustive", "No parent located")


def test_provenance_weight_failure_is_raised_on_every_rank():
    calls = []
    original = fault_mg.provenance_transfer_parallel

    def spy(*args, **kwargs):
        if not calls:
            calls.append((args, kwargs))
        return original(*args, **kwargs)

    fault_mg.provenance_transfer_parallel = spy
    try:
        mesh = adapted_2d(K).add_fault([("Flt", TRACE)])
        shear_fault_solve(mesh)
    finally:
        fault_mg.provenance_transfer_parallel = original
    assert uw.mpi.comm.allreduce(len(calls)) == uw.mpi.size, "no provenance transfer built"
    args, kwargs = calls[0]
    _same_error_everywhere(lambda: fault_mg.provenance_transfer_parallel(*args, **kwargs),
                           "_node_weights", "Singular matrix")
