"""``mesh.cell_size()`` is each cell's ``volume**(1/dim)``, and it tracks deformation.

The oracle here is computed from the vertex coordinates, not from PETSc, so this
is a check and not a restatement of the implementation. For simplices that is
the determinant volume of the cell; for the structured boxes it is the analytic
cell volume, scaled by the determinant of the affine map when the mesh is
deformed.

Partition independence is NOT tested here -- it cannot be, at one rank count.
``tests/parallel/test_1077`` compares the field cell by cell against its own
serial answer, and ``test_1078`` does the same for the three radius accessors.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

# The deformation applied below, and its Jacobian determinant. Volumes scale by
# |det|, so lengths scale by |det|**(1/dim).
AFFINE = np.array([[1.7, 0.2], [0.0, 1.0]])
AFFINE_DET = 1.7


def _simplex_volume_from_vertices(mesh):
    """Each simplex's volume from its own vertex coordinates.

    Triangle: ``|det[v1-v0, v2-v0]| / 2``. Tetrahedron: ``|det[...]| / 6``.
    """
    dim = mesh.dim
    cell_start, cell_end = mesh.dm.getHeightStratum(0)
    point_start, _point_end = mesh.dm.getDepthStratum(0)

    volumes = np.empty(cell_end - cell_start)
    factorial = 2.0 if dim == 2 else 6.0
    for cell in range(cell_end - cell_start):
        corners = mesh.dm.getTransitiveClosure(cell)[0][-(dim + 1):]
        coords = mesh._coords[corners - point_start]
        edges = coords[1:] - coords[0]
        volumes[cell] = abs(np.linalg.det(edges)) / factorial
    return volumes


@pytest.mark.parametrize("dim", [2, 3])
def test_simplex_cell_size_is_the_cube_root_of_its_own_volume(dim):
    """Against a determinant volume computed from the cell's vertices."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim, qdegree=3,
        cellSize=0.25, regular=False)
    mesh.cell_size()
    actual = np.asarray(mesh._cell_size_variable.array[:, 0, 0])
    expected = _simplex_volume_from_vertices(mesh) ** (1.0 / dim)

    error = float(np.abs(actual - expected).max(initial=0.0))
    uw.pprint(f"CELL_SIZE_GEOMETRY simplex dim={dim} max_error={error:.12g}")
    assert error < 1.0e-12, error


@pytest.mark.parametrize("dim", [2, 3])
def test_structured_cell_size_is_the_analytic_value_and_tracks_deformation(dim):
    """A regular box has an exact answer, and an affine deform scales it.

    ``elementRes=4`` on the unit box gives cells of side 0.25, so
    ``volume**(1/dim)`` is 0.25 whatever the dimension. Applying a linear map
    multiplies every cell volume by ``|det|``, hence every length by
    ``|det|**(1/dim)`` -- which the field must follow after ``deform``.
    """
    mesh = uw.meshing.StructuredQuadBox(
        minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim, qdegree=3,
        elementRes=(4,) * dim)
    mesh.cell_size()
    field = mesh._cell_size_variable

    actual = np.asarray(field.array[:, 0, 0])
    error = float(np.abs(actual - 0.25).max(initial=0.0))
    assert max(uw.mpi.comm.allgather(error)) < 1.0e-12, error

    coordinates = np.array(mesh.X.coords)
    coordinates[:, 0] = AFFINE[0, 0] * coordinates[:, 0] + AFFINE[0, 1] * coordinates[:, 1]
    mesh.deform(coordinates)

    expected = 0.25 * AFFINE_DET ** (1.0 / dim)
    actual = np.asarray(field.array[:, 0, 0])
    error = float(np.abs(actual - expected).max(initial=0.0))
    uw.pprint(f"CELL_SIZE_GEOMETRY tensor dim={dim} deformed max_error={error:.12g}")
    assert max(uw.mpi.comm.allgather(error)) < 1.0e-12, (error, expected)


def test_the_global_minimum_agrees_with_the_field():
    """``get_min_radius()`` reduces the same field ``cell_size()`` exposes.

    On a regular box every cell is the same size, so the global minimum is that
    size -- which also pins the reduction to the analytic value rather than to
    whatever the field happens to hold.
    """
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), qdegree=2)
    mesh.cell_size()
    assert mesh.get_min_radius() == pytest.approx(0.25, rel=1.0e-12)
    assert float(np.asarray(mesh._cell_size_variable.array).max()) == pytest.approx(
        0.25, rel=1.0e-12)
