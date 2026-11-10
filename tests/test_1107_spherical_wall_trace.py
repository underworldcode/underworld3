"""Original-wall multiplier traces require exact spherical-cone ownership."""

from types import SimpleNamespace

import numpy as np
import pytest
from petsc4py import PETSc

from underworld3.utilities._spherical_wall_trace import _inner_wall_cell_facets

pytestmark = pytest.mark.level_1


def _mesh_and_wall(components=1):
    comm = PETSc.COMM_WORLD.tompi4py()
    facets = np.eye(3)[None] if comm.rank == 0 else np.empty((0, 3, 3))
    ids = np.arange(6*components, dtype=PETSc.IntType).reshape(1, -1)
    if comm.rank != 0:
        ids = np.empty((0, 6*components), dtype=PETSc.IntType)
    return SimpleNamespace(dm=SimpleNamespace(comm=PETSc.COMM_WORLD)), dict(
        facets=facets, lid=ids, radius=1.,
    )


@pytest.mark.tier_b
@pytest.mark.parametrize("components", [1, 3])
def test_cell_facet_candidates_reach_remote_cells_and_empty_wall_ranks(components):
    mesh, wall = _mesh_and_wall(components)
    comm = PETSc.COMM_WORLD.tompi4py()
    cells = np.array([[[.7, .1, .1], [.1, .7, .1], [.1, .1, .7], [.4, .4, .4]]])
    if comm.rank != comm.size-1:
        cells = np.empty((0, 4, 3))
    result = _inner_wall_cell_facets(mesh, wall, cells)
    assert len(result) == len(cells)
    if len(cells):
        np.testing.assert_array_equal(result[0][0], np.eye(3)[None])
        np.testing.assert_array_equal(result[0][1], np.arange(6*components)[None])


@pytest.mark.tier_b
def test_cell_facet_candidates_keep_spherical_bowing_and_global_key_order():
    mesh, wall = _mesh_and_wall()
    comm = PETSc.COMM_WORLD.tompi4py()
    a, height = .2, np.sqrt(.96)
    cap = np.array([[a, 0, height], [-a/2, np.sqrt(3)*a/2, height],
                    [-a/2, -np.sqrt(3)*a/2, height]])
    if comm.rank == 0:
        wall["facets"] = np.array([cap, np.eye(3)])
        wall["lid"] = np.array([np.arange(100, 106), np.arange(6)])
    cell = np.array([[0, 0, 1.01], [.01, 0, 1.02], [0, .01, 1.02], [-.01, -.01, 1.02]])
    result = _inner_wall_cell_facets(mesh, wall, cell[None])
    triangles, ids = result[0]
    np.testing.assert_array_equal(ids, np.array([np.arange(6), np.arange(100, 106)]))
    np.testing.assert_array_equal(triangles, np.array([np.eye(3), cap]))


@pytest.mark.tier_b
def test_cell_facet_empty_queries_and_unequal_batch_count():
    mesh, wall = _mesh_and_wall()
    comm = PETSc.COMM_WORLD.tompi4py()
    count = 2051 if comm.rank == comm.size-1 else 0
    cell = np.array([[.7, .1, .1], [.1, .7, .1], [.1, .1, .7], [.4, .4, .4]])
    result = _inner_wall_cell_facets(mesh, wall, np.repeat(cell[None], count, axis=0))
    assert len(result) == count
    for triangles, ids in result:
        np.testing.assert_array_equal(triangles, np.eye(3)[None])
        np.testing.assert_array_equal(ids, np.arange(6)[None])
    assert _inner_wall_cell_facets(mesh, wall, np.empty((0, 4, 3))) == []


@pytest.mark.tier_b
def test_invalid_cell_cone_geometry_fails_on_every_rank():
    mesh, wall = _mesh_and_wall()
    comm = PETSc.COMM_WORLD.tompi4py()
    cells = np.empty((0, 4, 3))
    if comm.rank == comm.size-1:
        cells = np.array([[[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1.]]])
    with pytest.raises(ValueError, match="hemisphere"):
        _inner_wall_cell_facets(mesh, wall, cells)
