"""An adapted mesh keeps the same multigrid levels at any rank count (#865).

``Mesh.adapt`` keeps one refinement generation per halving of the cell size it
resolves with. That size was the smallest of the ranks' own 5th-percentile
cell diameters, so it moved with how the cells were dealt out. The size is now
one percentile over every rank's cells together.

Which fixture shows the old rule depends on the rank count. ``two_levels``
kept two add-on levels at 1 to 6 ranks and one at 7 and 8.
``h_far_above_the_base`` kept 1144 cells between its base and its child on one
rank, 1040 on 2, and 1040 and 1230 on 4. On one rank the old rule and the new
give the same levels, so a serial run of this file pins the serial hierarchy
and the percentile, not the rank independence.

The same assertions hold on one rank, so the file carries no ``min_size`` mark
and also runs serially.

    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_1177_mg_levels_same_at_any_rank_count_parallel.py
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import edge_split

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(300)]

TRACE = np.array([[0.3, 0.45], [0.7, 0.55]])


def _pooled(size, kind):
    """A fixed set of values, the same on every rank."""
    rng = np.random.default_rng(1177 + size)
    if kind == "spread":
        return rng.lognormal(-3.0, 2.0, size)
    if kind == "tied":
        return np.round(rng.random(size), 1)
    return rng.random(size) * 1e-12 + 1.0          # a few float steps apart


@pytest.mark.parametrize("pct", [5, 0, 100, 37.5, 12.3])
@pytest.mark.parametrize("kind", ["spread", "tied", "packed"])
@pytest.mark.parametrize("size", [1, 2, 4, 6, 7, 60, 1000])
def test_a_percentile_over_ranks_is_that_of_the_pooled_values(size, kind, pct):
    """Equal as floats, however the values are dealt out. With fewer values
    than ranks some ranks hold none. At 12.3 the spread values of sizes 4 and
    6 separate numpy's two interpolation formulas, one on each side."""
    pooled = _pooled(size, kind)
    mine = pooled[uw.mpi.rank::uw.mpi.size]
    assert edge_split.percentile_over_ranks(mine, pct) == float(np.percentile(pooled, pct))


def test_a_percentile_of_no_values_is_nan():
    assert np.isnan(edge_split.percentile_over_ranks(np.zeros(0), 5))


def _base():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, refinement=1,
        qdegree=2)


def test_the_resolved_cell_size_is_the_serial_one_at_any_rank_count():
    """0.0485807 is the 5th percentile of the 968 cell diameters. The smallest
    of the ranks' own percentiles was 0.04592 at 5 ranks and 0.04356 at 8."""
    mesh = _base()
    h = edge_split.resolved_cell_size(mesh.dm)
    assert abs(h - 0.04858069196873683) < 1e-9
    # handed this rank's diameters, as the adapt's marking pass hands them
    assert edge_split.resolved_cell_size(
        mesh.dm, diameters=edge_split.cell_diameters(mesh.dm)) == h


def _global_cells(dm):
    s, e = dm.getHeightStratum(0)
    return uw.mpi.comm.allreduce(e - s)


# The metric along the trace, the adapt's level cap and engine, and the cells
# of every multigrid level from the coarsest to the child, as one rank gives
# them. The ``edge_split`` engine hands the selection the sizes its marking
# passes measured; ``nvb`` leaves the selection to measure each generation.
FIXTURES = {
    "two_levels": (dict(h_near=0.05 / 4, h_far=0.05, width=0.08), 2, "nvb",
                   [242, 968, 1208, 1360]),
    "h_far_above_the_base": (dict(h_near=0.00625, h_far=0.1, width=0.05), 4, "nvb",
                             [242, 968, 1144, 1436]),
    "edge_split": (dict(h_near=0.00625, h_far=0.1, width=0.05), 4, "edge_split",
                   [242, 968, 1088, 1242, 1494]),
}


@pytest.mark.parametrize("name", list(FIXTURES))
def test_an_adapted_mesh_keeps_the_same_levels_at_any_rank_count(name):
    metric, max_levels, engine, cells = FIXTURES[name]
    base = _base()
    pts = np.column_stack([np.linspace(*TRACE[:, 0], 9), np.linspace(*TRACE[:, 1], 9),
                           np.zeros(9)])
    guide = uw.meshing.Surface(f"guide_1177_{name}", base, pts)
    guide.discretize()
    child = base.adapt(guide.refinement_metric_function(profile="linear", **metric),
                       max_levels=max_levels, engine=engine)
    levels = [m.dm for m in child._custom_mg_coarse_meshes] + [child.dm]
    assert [_global_cells(dm) for dm in levels] == cells
