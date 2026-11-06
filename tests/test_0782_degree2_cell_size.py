"""A cell's size does not depend on the degree of its coordinate field (#908).

``Mesh._get_cell_radii`` is ``volume**(1/dim)``. On a coordinate field of degree 2
the FVM routine read the edge midpoints as the corners, so each cell's volume came
back :math:`2^{-\\mathrm{dim}}` of the truth and every length half of it, measured
on simplices and on quadrilaterals before the fix, and it printed
"Edge has 6 coordinates != 4" as it went.

The parallel half of this defect is in
``tests/parallel/test_0782_degree2_mesh_empty_rank.py``.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _radii(mesh):
    return np.asarray(mesh._cell_radii).reshape(-1)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_a_regular_triangle_grid_has_its_closed_form_radius(degree):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, regular=True,
        qdegree=3, degree=degree)
    # 32 cells of area 1/32, and a radius is area**(1/2).
    assert _radii(mesh).size == 32
    assert np.allclose(_radii(mesh), (1.0 / 32.0) ** 0.5, rtol=1e-9, atol=0.0)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_a_quadrilateral_grid_has_its_closed_form_radius(degree):
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), degree=degree)
    assert _radii(mesh).size == 16
    assert np.allclose(_radii(mesh), 0.25, rtol=1e-9, atol=0.0)


@pytest.mark.parametrize("make_mesh", [
    pytest.param(lambda degree: uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0), cellSize=0.5, regular=True,
        qdegree=3, degree=degree), id="tetrahedra"),
    pytest.param(lambda degree: uw.meshing.StructuredQuadBox(
        elementRes=(3, 3, 3), degree=degree), id="hexahedra"),
])
def test_the_cell_volumes_of_a_unit_cube_sum_to_one(make_mesh):
    assert np.isclose((_radii(make_mesh(2)) ** 3).sum(), 1.0, rtol=1e-12)


def test_the_cell_areas_of_a_manifold_are_the_areas_the_assembler_integrates():
    """A 2-D mesh in 3-D, where the Jacobian is not square. The baseline is the
    area of the surface from the integral of one, on the same straight facets."""
    mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.4, qdegree=3, degree=2)
    one = uw.discretisation.MeshVariable("one0782", mesh, 1, degree=1)
    one.array[:, 0, 0] = 1.0
    area = uw.maths.Integral(mesh, one.sym[0]).evaluate()
    assert np.isclose((_radii(mesh) ** 2).sum(), area, rtol=1e-12)


def test_a_curved_cell_has_the_integral_of_its_jacobian():
    r"""The map :math:`x' = x,\ y' = y(1 + c x)` lies in the degree-2 space, has
    :math:`\det J = 1 + c x`, and bends every edge not parallel to an axis. So a
    cell of area :math:`A` with centroid :math:`\bar x` maps to
    :math:`A (1 + c \bar x)`, which the three corners alone do not give."""
    c = 0.5
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, regular=True,
        qdegree=3, degree=2)
    xbar = np.asarray(mesh._centroids)[:, 0].copy()
    nodes = np.array(mesh.dm.getCoordinatesLocal().array).reshape(-1, 2)
    nodes[:, 1] *= 1.0 + c * nodes[:, 0]
    mesh._deform_mesh(nodes)

    assert np.allclose(_radii(mesh) ** 2, (1.0 / 32.0) * (1.0 + c * xbar), rtol=1e-11, atol=0.0)
