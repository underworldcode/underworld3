"""add_particles_with_coordinates adds each point once, whatever the rank count.

Every rank is given the same coordinate array, and keeps the points its part of
the mesh claims (``Mesh.points_in_domain``). A point on a partition seam lies in
the cells of two or more ranks, so each of them claimed it and added it: on a unit
box at np 2, ten points became eleven (the parallel review desk, reviewing
#1252). It happens for a seam point inside the mesh and for one where the seam
meets a wall.

The baseline is exact: the swarm holds the given points, each once, at the given
coordinates. The test points are the mesh's own seam vertices (found from the
point star forest, so the partition decides them, not a guess), plus points well
inside cells.
"""
import numpy as np
import pytest

import underworld3 as uw

from empty_rank_fixtures import (cells_along_a_side_leaving_one_rank_empty,
                                 cells_per_rank, right_triangle_of_cells)

pytestmark = [pytest.mark.level_1, pytest.mark.mpi(min_size=2),
              pytest.mark.timeout(240, method="thread")]


def _seam_vertices(mesh):
    """The coordinates of the vertices on a partition seam, the same array on
    every rank (sorted). Collective."""
    dm = mesh.dm
    v0, v1 = dm.getDepthStratum(0)
    shared = sorted(p for p in mesh._seam_points() if v0 <= p < v1)
    coords = dm.getCoordinatesLocal().array.reshape(-1, mesh.cdim)
    local = coords[np.asarray(shared, dtype=int) - v0] if shared else np.zeros((0, mesh.cdim))
    every = np.concatenate(uw.mpi.comm.allgather(local), axis=0)
    return np.unique(every, axis=0)


def _cell_centroids(mesh, count):
    """Points well inside cells (centroids), the same array on every rank."""
    dm = mesh.dm
    c0, c1 = dm.getHeightStratum(0)
    v0, v1 = dm.getDepthStratum(0)
    coords = dm.getCoordinatesLocal().array.reshape(-1, mesh.cdim)
    centroids = []
    for cell in range(c0, c1):
        closure = dm.getTransitiveClosure(cell)[0]
        verts = [p - v0 for p in closure if v0 <= p < v1]
        centroids.append(coords[verts].mean(axis=0))
    local = np.asarray(centroids).reshape(-1, mesh.cdim)[:count]
    return np.unique(np.concatenate(uw.mpi.comm.allgather(local), axis=0), axis=0)


def _added(points, mesh):
    """Add ``points`` (the same on every rank); return every rank's particles,
    gathered and sorted."""
    swarm = uw.swarm.Swarm(mesh)
    swarm.add_particles_with_coordinates(points)
    local = np.asarray(swarm.coords).reshape(-1, mesh.cdim)
    every = np.concatenate(uw.mpi.comm.allgather(local), axis=0)
    return every[np.lexsort(every.T[::-1])]


def _assert_each_point_once(points, mesh, what):
    added = _added(points, mesh)
    expected = points[np.lexsort(points.T[::-1])]
    assert added.shape[0] == points.shape[0], (
        f"{points.shape[0]} {what} points became {added.shape[0]} particles at "
        f"np {uw.mpi.size}")
    np.testing.assert_array_equal(added, expected)


def _on_a_wall(points):
    return np.any((points == 0.0) | (points == 1.0), axis=1)


@pytest.mark.tier_b
def test_a_seam_vertex_inside_the_box_is_added_once():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.1)
    seam = _seam_vertices(mesh)
    inside = seam[~_on_a_wall(seam)]
    assert inside.shape[0] > 0, "this partition has no seam vertex inside the box"
    _assert_each_point_once(inside, mesh, "interior seam")


@pytest.mark.tier_b
def test_a_seam_vertex_on_a_wall_is_added_once():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.1)
    seam = _seam_vertices(mesh)
    on_wall = seam[_on_a_wall(seam)]
    assert on_wall.shape[0] > 0, "this partition has no seam vertex on a wall"
    _assert_each_point_once(on_wall, mesh, "wall-and-seam")


@pytest.mark.tier_b
def test_seam_and_cell_points_together_are_each_added_once():
    """The seam vertices mixed with points well inside cells, as a user's array
    would be."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.1)
    points = np.concatenate([_seam_vertices(mesh), _cell_centroids(mesh, 20)], axis=0)
    _assert_each_point_once(np.unique(points, axis=0), mesh, "seam and cell")


@pytest.mark.tier_b
def test_a_rank_with_no_cells_adds_nothing_and_the_others_add_each_point_once():
    size = uw.mpi.size
    dm = right_triangle_of_cells(cells_along_a_side_leaving_one_rank_empty(size),
                                 max(size - 1, 1))
    mesh = uw.discretisation.Mesh(dm, degree=1, qdegree=2)
    if size > 2:
        assert 0 in cells_per_rank(mesh)
    points = np.unique(np.concatenate([_seam_vertices(mesh), _cell_centroids(mesh, 20)],
                                      axis=0), axis=0)
    _assert_each_point_once(points, mesh, "triangle")
