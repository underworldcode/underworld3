"""``uw.selective_ranks()`` in parallel (#837): the flag differs by rank, the
body runs on every rank, and the collective guard reports a partial selection
on every rank that reaches the call. The serial contract is in
tests/test_0076_selective_ranks_contract.py."""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [
    pytest.mark.level_1,
    pytest.mark.tier_a,
    pytest.mark.mpi(min_size=2),
    pytest.mark.timeout(60),
]


def test_every_rank_runs_the_body_and_only_rank_0_is_flagged():
    with uw.selective_ranks(0) as should_execute:
        flags = uw.mpi.comm.allgather(bool(should_execute))   # every rank is here
    assert flags == [True] + [False] * (uw.mpi.size - 1)


def test_a_mask_and_its_guard_set_agree_on_every_rank():
    mask = np.array([r % 2 == 1 for r in range(uw.mpi.size)])
    with uw.selective_ranks(mask) as should_execute:
        assert should_execute == bool(mask[uw.mpi.rank])
        assert uw.mpi._selective_executing_ranks == set(np.flatnonzero(mask))


def test_the_guard_reports_a_partial_selection_and_passes_a_full_one():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
    var = uw.discretisation.MeshVariable("T_mpi", mesh, 1, degree=1)
    # every rank calls inside the block, so every rank raises: no rank is left
    # waiting in the collective
    with uw.selective_ranks(list(range(1, uw.mpi.size + 1))):
        with pytest.raises(uw.CollectiveOperationError):
            var.stats()
    with uw.selective_ranks(list(range(uw.mpi.size + 1))):
        var.stats()
