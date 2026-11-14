"""adapt(engine='nvb') runs on every rank when a rank holds no cells.

``DMPlexIsSimplex`` is rank-local: it returns False on a rank with no cells. The
NVB adapt refused on ``not dm.isSimplex()``, so the empty rank alone raised
NotImplementedError while the others went on into ``marking_metric``'s
collectives and hung. The mesh's ``isSimplex``, reconciled over ranks at
construction, decides now.

A unit box of cell size 1 with two refinements has 64 cells; at np 3 and np 4 the
partition leaves a rank without cells (asserted; skipped where it does not).

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 3 python -m pytest --with-mpi tests/parallel/test_1189_adapt_nvb_rank_without_cells_parallel.py
"""
import numpy as np
import pytest

import underworld3 as uw

from empty_rank_fixtures import cells_per_rank

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(300)]


def test_nvb_adapt_with_a_rank_without_cells():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=1.0,
                                             refinement=2)
    counts = cells_per_rank(mesh)
    if 0 not in counts:
        pytest.skip(f"np {uw.mpi.size}: no rank is left without cells ({counts})")
    assert mesh.isSimplex

    metric = uw.discretisation.MeshVariable("M1189", mesh, 1, degree=1)
    X = np.asarray(metric.coords)
    metric.array[:, 0, 0] = np.where(np.abs(X[:, 1] - 0.5) < 0.2, 16.0, 4.0) ** 2

    message = ""
    child = None
    try:
        child = mesh.adapt(metric, max_levels=1, engine="nvb")
    except NotImplementedError as error:
        message = str(error)
    messages = uw.mpi.comm.allgather(message)
    assert not any(messages), messages
    s, e = child.dm.getHeightStratum(0)
    assert uw.mpi.comm.allreduce(e - s) > sum(counts)
