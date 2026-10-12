"""``cell_size()`` must survive a rank that owns none of the submesh.

A region submesh keeps only the cells carrying its label, so a partition whose
share of the parent lies outside that region legitimately owns nothing. That is
routine for submeshes and is the case this guards -- not an over-decomposed
full mesh, which is a different (and degenerate) situation.

The defect (#698): ``_assemble_cell_size`` returned early on ``radii.size == 0``,
in front of TWO collectives --- the first ``var.data`` access, which allocates
through ``dm.createSubDM`` and ``createGlobalVector``, and the assignment into
it, which fires ``pack_raw_data_to_petsc``. Starved ranks skipped both while
their populated peers made them, and the job never finished. Measured at np=8
with cells per rank ``[12, 11, 0, 19, 0, 0, 0, 0]``.
"""

import pytest

import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_1, pytest.mark.tier_a]


def _thin_slab_parent():
    """A box split near the top, so the outer region is a few cells deep."""
    return uw.meshing.BoxInternalBoundary(
        elementRes=(12, 12), zelementRes=(10, 2), zintCoord=0.85,
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), simplex=True, qdegree=2)


@pytest.mark.mpi(min_size=2)
def test_cell_size_completes_when_ranks_own_none_of_the_submesh():
    """The whole test is that this returns. A regression hangs rather than fails.

    The starved-rank count is reported, not asserted: how the parent is split is
    the partitioner's business, and at np=2 the slab may well reach both ranks.
    What must hold at every rank count is that every rank comes back.
    """
    submesh = _thin_slab_parent().extract_region("Outer")
    start, end = submesh.dm.getHeightStratum(0)
    counts = uw.mpi.comm.allgather(end - start)

    submesh.cell_size()
    uw.mpi.comm.barrier()

    uw.mpi.pprint(f"SUBMESH_CELL_SIZE ranks={uw.mpi.size} cells={counts} "
                  f"starved={counts.count(0)}")
    assert sum(counts) > 0, "the region submesh is empty everywhere; test is vacuous"


@pytest.mark.mpi(min_size=2)
def test_the_size_field_is_filled_where_there_are_cells():
    """A starved rank writing nothing must not stop the others writing.

    Without this, the fix above could be satisfied by never filling the field
    at all.
    """
    import numpy as np

    submesh = _thin_slab_parent().extract_region("Outer")
    start, end = submesh.dm.getHeightStratum(0)
    submesh.cell_size()

    local = np.asarray(submesh._cell_size_variable.array[:, 0, 0])
    assert local.shape[0] == end - start
    if local.size:
        assert np.all(local > 0.0), "populated rank has non-positive cell sizes"
    from mpi4py import MPI
    filled = uw.mpi.comm.allreduce(int(local.size), op=MPI.SUM)
    assert filled > 0, "no rank filled the field at all"
