"""Basis coordinates must reproduce affine simplex geometry, including halos.

The coarse tetrahedral interface mesh has ranks owning no coordinate rows at
16 MPI processes. Global interpolation previously failed during matrix assembly
on that partition, before any Stokes or pressure-jump calculation.
"""

import numpy as np
import pytest
import underworld3 as uw

pytestmark = pytest.mark.level_1


@pytest.fixture(scope="module")
def interface_mesh():
    return uw.meshing.BoxInternalBoundary(
        minCoords=(0, 0, 0), maxCoords=(1, 1, 1), zintCoord=0.5,
        cellSize=0.35, degree=1, qdegree=3, simplex=True,
    )


@pytest.mark.tier_b
@pytest.mark.parametrize("degree,continuous", [(1, True), (2, True), (0, False)])
def test_basis_coordinates_are_affine_simplex_nodes(interface_mesh, degree, continuous):
    mesh = interface_mesh
    coords = np.asarray(mesh._get_coords_for_basis(degree, continuous))
    rows = mesh._cell_node_indices(degree, continuous)
    dm = mesh.dm
    section = dm.getCoordinateSection()
    vertices_start, vertices_end = dm.getDepthStratum(0)
    cells_start, cells_end = dm.getHeightStratum(0)
    physical = dm.getCoordinatesLocal().array
    for cell in range(cells_start, cells_end):
        vertices = np.asarray([
            physical[section.getOffset(point):section.getOffset(point) + 3]
            for point in dm.getTransitiveClosure(cell)[0]
            if vertices_start <= point < vertices_end
        ])
        if degree == 0:
            expected = vertices.mean(axis=0, keepdims=True)
        elif degree == 1:
            expected = vertices
        else:
            expected = np.vstack([
                vertices,
                [(vertices[i] + vertices[j]) / 2
                 for i in range(4) for j in range(i + 1, 4)],
            ])
        actual = coords[rows[cell - cells_start]]
        distances = np.linalg.norm(actual[:, None, :] - expected[None, :, :], axis=2)
        assert actual.shape == expected.shape
        assert np.max(np.min(distances, axis=1)) < 2e-13
        assert np.max(np.min(distances, axis=0)) < 2e-13


@pytest.mark.tier_b
def test_shared_continuous_coordinates_are_identical(interface_mesh):
    """A shared FE node has exactly one coordinate value across MPI ranks."""
    mesh = interface_mesh
    coords = np.asarray(mesh._get_coords_for_basis(1, True))
    dm = mesh._basis_coordinate_dm(1, True)
    section, global_section = dm.getLocalSection(), dm.getGlobalSection()
    local = []
    for point in range(*dm.getDepthStratum(0)):
        offset = global_section.getOffset(point)
        global_offset = -(offset + 1) if offset < 0 else offset
        local.append((global_offset, coords[section.getOffset(point) // 3].copy()))
    gathered = uw.mpi.comm.allgather(local)
    dm.destroy()
    shared = {}
    for entries in gathered:
        for index, value in entries:
            if index in shared:
                np.testing.assert_array_equal(value, shared[index])
            else:
                shared[index] = value


@pytest.mark.tier_b
@pytest.mark.parametrize("degree,continuous", [(1, True), (2, True), (0, False)])
def test_surface_basis_coordinates_reproduce_embedded_affine_facets(degree, continuous):
    """The coordinate identity has three components on a two-dimensional surface."""
    mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.4, degree=1, qdegree=3)
    coords = np.asarray(mesh._get_coords_for_basis(degree, continuous))
    rows = mesh._cell_node_indices(degree, continuous)
    dm = mesh.dm
    section = dm.getCoordinateSection()
    physical = dm.getCoordinatesLocal().array
    vstart, vend = dm.getDepthStratum(0)
    cstart, cend = dm.getHeightStratum(0)
    for cell in range(cstart, cend):
        vertices = np.asarray([
            physical[section.getOffset(point):section.getOffset(point) + 3]
            for point in dm.getTransitiveClosure(cell)[0]
            if vstart <= point < vend
        ])
        if degree == 0:
            expected = vertices.mean(axis=0, keepdims=True)
        elif degree == 1:
            expected = vertices
        else:
            expected = np.vstack([vertices, [(vertices[i] + vertices[j]) / 2
                                  for i in range(3) for j in range(i + 1, 3)]])
        actual = coords[rows[cell - cstart]]
        distances = np.linalg.norm(actual[:, None, :] - expected[None, :, :], axis=2)
        assert actual.shape == expected.shape
        assert np.max(np.min(distances, axis=1)) < 2e-13
        assert np.max(np.min(distances, axis=0)) < 2e-13


@pytest.mark.tier_b
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_affine_basis_nodes_stay_in_the_closed_physical_bounding_box(degree):
    """An affine FE node cannot lie outside its vertices' convex hull."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.15)
    coords = mesh._get_coords_for_basis(degree, True)
    bounds = np.asarray(mesh.dm.getBoundingBox())
    assert np.all(coords >= bounds[:, 0])
    assert np.all(coords <= bounds[:, 1])
