"""Issue #687: cell_size is an own-cell geometric quantity, including after deform.

The independent oracle reads vertex coordinates through the coordinate section;
it does not use the mesh's cached radii or centroid kd-tree. Run serial and MPI.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _vertex_rms(mesh):
    dm = mesh.dm
    section = dm.getCoordinateDM().getLocalSection()
    coordinates = dm.getCoordinatesLocal().array
    start, end = dm.getHeightStratum(0)
    first_vertex, last_vertex = dm.getDepthStratum(0)
    radii = []
    for cell in range(start, end):
        vertices = [int(point) for point in dm.getTransitiveClosure(cell)[0]
                    if first_vertex <= point < last_vertex]
        points = np.array([coordinates[section.getOffset(v):section.getOffset(v) + mesh.cdim]
                           for v in vertices])
        radii.append(np.sqrt(np.mean(np.sum((points - points.mean(axis=0)) ** 2, axis=1))))
    return np.asarray(radii)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [True, False], ids=["simplex", "tensor"])
def test_cell_size_matches_own_vertices_and_tracks_deform(dim, simplex):
    geometry = dict(minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim, qdegree=3)
    mesh = (uw.meshing.UnstructuredSimplexBox(**geometry, cellSize=0.25, regular=False)
            if simplex else uw.meshing.StructuredQuadBox(**geometry, elementRes=(4,) * dim))
    mesh.cell_size()
    field = mesh._cell_size_variable
    errors = []
    for phase in ("initial", "deformed"):
        if phase == "deformed":
            coordinates = np.array(mesh.X.coords)
            coordinates[:, 0] = 1.7 * coordinates[:, 0] + 0.2 * coordinates[:, 1]
            mesh.deform(coordinates)
        expected = _vertex_rms(mesh)
        actual = np.asarray(field.array[:, 0, 0])
        shapes_match = actual.shape == expected.shape
        assert all(uw.mpi.comm.allgather(shapes_match)), (actual.shape, expected.shape)
        local_error = float(np.abs(actual - expected).max(initial=0.0))
        error = max(uw.mpi.comm.allgather(local_error))
        errors.append(error)
        uw.pprint(f"CELL_SIZE_GEOMETRY dim={dim} simplex={simplex} phase={phase} "
                  f"ranks={uw.mpi.size} max_error={error:.12g}")
    assert max(errors) < 1e-12, errors


def test_regular_square_cell_size_keeps_global_radius():
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), qdegree=2)
    legacy = np.array(mesh._radii)
    global_radius = mesh.get_min_radius()
    mesh.cell_size()
    expected = np.sqrt(2.0) / 8.0
    error = float(np.abs(np.asarray(mesh._cell_size_variable.array) - expected).max(initial=0.0))
    assert max(uw.mpi.comm.allgather(error)) < 1e-12
    assert global_radius == pytest.approx(expected, rel=1e-12)
    assert all(uw.mpi.comm.allgather(np.array_equal(mesh._radii, legacy)))
