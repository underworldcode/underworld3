"""Higher-order native rendering in parallel: ghost DOFs (#1057).

Each rank draws its own cells. Their closures reach shared vertices and
edges, whose DOFs the rank holds as ghosts, and those must be found through
the local section like any other. Every rank's grid keeps all of its cells,
has the variable's local DOFs as its points, and the sub-cells of all ranks
together tile the unit square exactly once.
"""

import numpy as np
import pytest
import underworld3 as uw
from mpi4py import MPI

pv = pytest.importorskip("pyvista")
vis = pytest.importorskip("underworld3.visualisation")

pytestmark = [pytest.mark.level_2, pytest.mark.mpi(min_size=2), pytest.mark.timeout(120)]


@pytest.mark.tier_b
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_native_grid_per_rank(degree, simplex):
    if simplex:
        mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                                 cellSize=0.2)
    else:
        mesh = uw.meshing.StructuredQuadBox(elementRes=(6, 6))
    T = uw.discretisation.MeshVariable(f"T{degree}{int(simplex)}", mesh, 1, degree=degree)
    grid = vis.meshVariable_to_native_pv_mesh(T)
    ok = grid is not None
    assert uw.mpi.comm.allreduce(ok, op=MPI.LAND)

    cs, ce = mesh.dm.getHeightStratum(0)
    assert grid.n_cells == (ce - cs) * degree**2
    np.testing.assert_array_equal(np.asarray(grid.points)[:, :2], np.asarray(T.coords)[:, :2])
    area = np.asarray(grid.compute_cell_sizes(length=False, area=True, volume=False)
                      .cell_data["Area"])
    assert area.min() > 0.0
    total = uw.mpi.comm.allreduce(float(area.sum()), op=MPI.SUM)
    assert total == pytest.approx(1.0, rel=1e-12)
