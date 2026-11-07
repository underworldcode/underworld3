"""A cell's size does not depend on the degree of its coordinate field (#908).

``Mesh._get_cell_radii`` is ``volume**(1/dim)``. On a coordinate field of higher
degree the FVM routine read the wrong closure nodes as the corners, and printed
"Edge has 6 coordinates != 4" as it went. Against the true length the old value
was short by a factor of, at degree 2, 2 on triangles, quadrilaterals and
tetrahedra and 3.63 on hexahedra; at degree 3, 2.59 on triangles, about 3e6 on
quadrilaterals, 3 on tetrahedra and 4.06 on hexahedra.

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


def _corners(mesh):
    """Each cell's corners, from the degree-1 vertex table of the same mesh."""
    nodes = mesh._get_coords_for_basis(1, True)
    return np.asarray(nodes)[np.asarray(mesh._cell_node_indices(1, True))]


def test_each_tetrahedron_has_the_volume_of_its_corners():
    """Per cell, since a sum over the cube would pass with volume moved between
    cells. A regular tetrahedral box has cells of several sizes."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0), cellSize=0.5, regular=True,
        qdegree=3, degree=2)
    corners = _corners(mesh)
    edges = corners[:, 1:] - corners[:, :1]
    volumes = np.abs(np.linalg.det(edges)) / 6.0
    assert volumes.max() > 1.5 * volumes.min()
    assert np.allclose(_radii(mesh) ** 3, volumes, rtol=1e-12, atol=0.0)


def test_each_hexahedron_of_a_unit_cube_has_its_closed_form_volume():
    mesh = uw.meshing.StructuredQuadBox(elementRes=(3, 3, 3), degree=2)
    assert _radii(mesh).size == 27
    # 1e-10: this box's coordinates scatter by 2e-11 at degree 1 as well.
    assert np.allclose(_radii(mesh) ** 3, 1.0 / 27.0, rtol=1e-10, atol=0.0)


def test_each_cell_of_a_manifold_has_the_area_of_its_flat_facet():
    """A 2-D mesh in 3-D, where the Jacobian is not square. These cells are straight
    sided, so half the cross product of two edges is the closed form."""
    mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.4, qdegree=3, degree=2)
    corners = _corners(mesh)
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    assert np.allclose(_radii(mesh) ** 2, areas, rtol=1e-12, atol=0.0)


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


def _warped_volume(mesh, a, warp):
    nodes = np.array(mesh.dm.getCoordinatesLocal().array).reshape(-1, 3)
    mesh._deform_mesh(warp(a, *nodes.T.copy()).T)
    return (_radii(mesh) ** 3).sum()


@pytest.mark.parametrize("qdegree", [1, 2, 5])
def test_a_curved_tetrahedral_size_does_not_depend_on_the_solver_quadrature(qdegree):
    r"""A cell's size is a property of its coordinates, so it reads no solver's
    ``qdegree``, whether low enough to be wrong or high enough to be costly. The map
    :math:`x' = x + a y^2,\ y' = y + a z^2,\ z' = z + a x^2` lies in the degree-2
    space with :math:`\det J = 1 + 8 a^3 x y z`, so the cube's volume is
    :math:`1 + a^3`, and a one-point rule does not reach it."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0), cellSize=0.5, regular=True,
        qdegree=qdegree, degree=2)
    volume = _warped_volume(mesh, 0.5, lambda a, x, y, z: np.array(
        [x + a * y**2, y + a * z**2, z + a * x**2]))
    assert np.isclose(volume, 1.0 + 0.5**3, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("qdegree", [1, 2, 5])
def test_a_curved_hexahedral_size_does_not_depend_on_the_solver_quadrature(qdegree):
    r"""The map :math:`x' = x + a y^2 z^2` and its cyclic partners lie in the
    biquadratic space, and :math:`\det J` has degree 4 in each variable: two points a side leave 5.8e-5, three are exact. The cube's volume is
    :math:`1 - 3 a^2 / 5 + a^3 / 4`, which at :math:`a = 1/2` is 0.88125."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(3, 3, 3), degree=2, qdegree=qdegree)
    volume = _warped_volume(mesh, 0.5, lambda a, x, y, z: np.array(
        [x + a * y**2 * z**2, y + a * z**2 * x**2, z + a * x**2 * y**2]))
    assert np.isclose(volume, 0.88125, rtol=1e-13, atol=0.0)


def test_taking_the_cells_in_blocks_does_not_change_the_answer(monkeypatch):
    """Geometry is built a bounded block at a time. Cells straddling a block
    boundary are the ones a slip would lose or count twice."""
    from underworld3.cython import petsc_discretisation

    mesh = uw.meshing.StructuredQuadBox(elementRes=(3, 3, 3), degree=2)
    whole = petsc_discretisation.petsc_fem_get_local_cell_sizes(mesh)
    # 27 quadrature points a cell, so this is 2 cells a block, then 1 a block.
    for points in (54, 1):
        monkeypatch.setattr(petsc_discretisation, "_POINTS_PER_GEOMETRY_BLOCK", points)
        assert np.array_equal(
            petsc_discretisation.petsc_fem_get_local_cell_sizes(mesh), whole)
