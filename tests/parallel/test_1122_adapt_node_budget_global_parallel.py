"""adapt's node_budget is one budget for the whole mesh, not one per rank (#1122).

``node_budget=N`` caps the cells marked in each refinement pass at N, highest
metric first. Each rank cut its own candidates at N, so on P ranks up to N x P
cells were marked and which they were depended on the partition. The cut is now
taken over all ranks, with ties broken by cell centroid, so the marked set, and
the refined mesh, do not depend on the rank count.

The fixture is the issue's: a guide surface along (0.3, 0.45)-(0.7, 0.55) in a
unit box of cell size 0.1 with one refinement, where a budget of 50 bites. The
cell counts of every multigrid level must equal the serial ones.

    python scripts/mpi_supervisor.py --silence 600 -- mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_1122_adapt_node_budget_global_parallel.py
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(600)]

# Cells of each level with node_budget=50, the same at np 1, 2 and 4. On
# development (aa3c49df) np 1 gave these but for edge_split's last level (2956,
# where argsort chose among cells of equal metric); np 2 and 4 gave
# nvb 1208/1360, edge_split 1174/2898 and 1413/2884, sbr 1348/3025 and 1374/3123
# for the last two levels.
SERIAL = {
    "nvb": [242, 968, 1112, 1282],
    "edge_split": [242, 968, 1138, 2690],
    "sbr": [242, 968, 1176, 2036],
}


def _owned_cells(dm):
    cs, ce = dm.getHeightStratum(0)
    try:
        _, ilocal, _ = dm.getPointSF().getGraph()
    except (ValueError, TypeError):      # serial: no graph
        ilocal = None
    leaves = set() if ilocal is None else {int(x) for x in ilocal}
    return sum(1 for c in range(cs, ce) if c not in leaves)


@pytest.mark.parametrize("engine", ["nvb", "edge_split", "sbr"])
def test_node_budget_is_global(engine):
    if engine == "nvb":
        pytest.importorskip("underworld3.utilities._nvb_transform",
                            reason="native uwnvb transform not built")
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=0.1, refinement=1, qdegree=2)
    pts = np.column_stack([np.linspace(0.3, 0.7, 9), np.linspace(0.45, 0.55, 9),
                           np.zeros(9)])
    guide = uw.meshing.Surface(f"G1122{engine}", base, pts)
    guide.discretize()
    metric = guide.refinement_metric_function(h_near=0.0125, h_far=0.05, width=0.08,
                                              profile="linear")
    child = base.adapt(metric, max_levels=2, engine=engine, node_budget=50)
    counts = [uw.mpi.comm.allreduce(_owned_cells(m.dm)) for m in child.multigrid_levels]
    assert counts == SERIAL[engine], f"np {uw.mpi.size}: {counts}"


def test_the_cut_does_not_depend_on_the_deal():
    """40 candidates, half of them tied on the metric, dealt to the ranks in
    three ways, one leaving rank 0 with none: the 15 kept are the same set."""
    from underworld3.utilities.edge_split import top_over_ranks

    rng = np.random.default_rng(1122)
    centroids = rng.random((40, 2))
    metric = np.where(np.arange(40) < 20, 7.0, rng.random(40))
    # the truth: by metric, then x, then y
    truth = set(np.lexsort((centroids[:, 1], centroids[:, 0], -metric))[:15].tolist())
    size, rank = uw.mpi.size, uw.mpi.rank
    for deal in (np.arange(40) % size,
                 rng.permutation(40) % size,
                 np.where(np.arange(40) < 20, size - 1, (np.arange(40) % max(size - 1, 1)) + (size > 1))):
        mine = np.where(deal == rank)[0]
        kept = mine[top_over_ranks(metric[mine], centroids[mine], 15)]
        pooled = set(np.concatenate(uw.mpi.comm.allgather(kept)).tolist())
        assert pooled == truth


@pytest.mark.parametrize("engine", ["nvb", "edge_split", "sbr"])
def test_a_budget_of_zero_refines_nothing(engine):
    """node_budget=0 marks nothing and returns the mesh unrefined, as before the
    global cut (which first raised IndexError here)."""
    if engine == "nvb":
        pytest.importorskip("underworld3.utilities._nvb_transform",
                            reason="native uwnvb transform not built")
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=0.1, refinement=1, qdegree=2)
    child = base.adapt(lambda X: np.full(len(X), 1.0e4), max_levels=1, engine=engine,
                       node_budget=0)
    counts = [uw.mpi.comm.allreduce(_owned_cells(m.dm)) for m in child.multigrid_levels]
    assert counts[-1] == 968, f"np {uw.mpi.size}: {counts}"
