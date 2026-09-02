"""The adapt level loop must stop on a collective fact, not a rank-local one (#512).

Each level of the marking loop ends with a DM refinement, which is collective.
Deciding to leave that loop is therefore a decision every rank has to take
together. The sbr path used to break on ``refine.size == 0`` — "this rank's own
cells are all fine enough" — which says nothing about anyone else's. The rank
that finished first left for good and its peers went round again into the
refinement and waited.

It is not latent. Measured at np=2 on `development` before the fix: a callable
metric demanding h = 0.01 within r < 0.15 of one corner and nothing elsewhere,
on a 132-cell-per-rank base, hung with neither rank returning; `mpirun` reached
its timeout.

Run under MPI::

    mpirun -np 2 python -m pytest --with-mpi \
        tests/parallel/test_0873_adapt_collective_stop_mpi.py

A callable metric is used deliberately, so that nothing here depends on how
``global_evaluate`` behaves — it is evaluated rank-locally. #512 attributes the
hazard to ``eval_metric`` being collective; measured at np=2 that is not so,
with one rank asleep for 10 s the other's ``global_evaluate`` returned in
0.06 s, for in-domain and for stranded points alike. The collective in this
loop is the refinement, not the metric.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [
    pytest.mark.mpi(min_size=2),
    pytest.mark.timeout(300),
    pytest.mark.level_2,
    pytest.mark.tier_a,
]

ENGINES = ["sbr", "edge_split", "nvb"]


def _corner_metric(points):
    """Fine in one corner, already satisfied everywhere else.

    This is what splits the ranks: whichever rank holds no corner cells has
    nothing to refine at the first level, while its peer has work for several.
    """

    p = np.asarray(points)
    r = np.sqrt((p[:, 0] - 0.02) ** 2 + (p[:, 1] - 0.02) ** 2)

    return 1.0 / np.where(r < 0.15, 0.01, 1.0) ** 2


def _base_mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.2,
        regular=False,
        qdegree=2,
        refinement=1,
    )


def test_premise_the_metric_splits_the_ranks():
    """Test premise: without a rank that runs out of work, nothing here is tested."""

    mesh = _base_mesh()
    cs, ce = mesh.dm.getHeightStratum(0)

    centroid = np.asarray(
        [mesh.dm.computeCellGeometryFVM(c)[1] for c in range(cs, ce)]
    ).reshape(ce - cs, -1)[:, : mesh.dim]

    wants_refining = int((_corner_metric(centroid) > 1.0).sum())
    per_rank = uw.mpi.comm.allgather(wants_refining)

    assert sum(per_rank) > 0, f"no rank has anything to refine: {per_rank}"
    assert min(per_rank) == 0, (
        f"every rank has work at np={uw.mpi.size} ({per_rank}) — this test "
        "cannot detect the rank-local stop it exists for"
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_adapt_returns_when_one_rank_runs_out_of_work(engine):
    """The whole assertion is that this returns. The failure mode is a hang."""

    mesh = _base_mesh()

    child = mesh.adapt(_corner_metric, max_levels=4, engine=engine)

    c0, c1 = child.dm.getHeightStratum(0)
    total = uw.mpi.comm.allreduce(c1 - c0)
    base = uw.mpi.comm.allreduce(
        mesh.dm.getHeightStratum(0)[1] - mesh.dm.getHeightStratum(0)[0]
    )

    assert total > base, (
        f"{engine}: the corner metric demands refinement, but the child has "
        f"{total} cells against the base's {base}"
    )
