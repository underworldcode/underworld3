"""Empty partitions must not change the cell family or poison later mesh loads.

The original failure appeared in SUPG test 1077 after the empty-partition
test: rank-local DMPlexIsSimplex returned False on empty ranks, so the
coordinate FE consumed different COMM_WORLD tags. The next HDF5 label load
then hung in PetscSFSetUp_Basic. No transport solve is needed to reproduce it.
"""

import numpy as np
import pytest
from petsc4py import PETSc

import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2,
              pytest.mark.tier_b, pytest.mark.timeout(120)]


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False])
def test_cell_family_and_next_mesh_load_on_empty_ranks(dim, simplex):
    if simplex:
        coords = np.vstack([np.zeros(dim), np.eye(dim)])
        cells = np.arange(dim + 1, dtype=PETSc.IntType).reshape(1, -1)
        dm = PETSc.DMPlex().createFromCellList(dim, cells, coords)
    else:
        dm = PETSc.DMPlex().createBoxMesh([1] * dim, simplex=False)
    first = uw.discretisation.Mesh(dm, simplex=simplex, qdegree=3)
    start, end = first.dm.getHeightStratum(0)
    counts = uw.mpi.comm.allgather(end - start)
    assert min(counts) == 0 and sum(counts) == 1, counts

    families = uw.mpi.comm.allgather(first.isSimplex)
    assert families == [simplex] * uw.mpi.size, families
    expected = { (2, True): "triangle", (3, True): "tetrahedron",
                 (2, False): "quadrilateral", (3, False): "hexahedron" }
    elements = uw.mpi.comm.allgather(first._element.type)
    assert elements == [expected[dim, simplex]] * uw.mpi.size, elements

    # This public constructor exercises the HDF5 label SF exchange that hung.
    second = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim,
        cellSize=0.25, qdegree=3, regular=False,
    )
    field = uw.discretisation.MeshVariable("T", second, 1, degree=2)
    field.array[:, 0, 0] = 1.0
    volume = uw.maths.Integral(second, field.sym[0]).evaluate()
    assert np.isclose(volume, 1.0, rtol=1e-12, atol=1e-12), volume
    for boundary in second.boundaries:
        if boundary.name in ("Null_Boundary", "All_Boundaries"):
            continue
        area = uw.maths.BdIntegral(second, field.sym[0], boundary=boundary.name).evaluate()
        assert np.isclose(area, 1.0, rtol=1e-12, atol=1e-12), (boundary.name, area)
