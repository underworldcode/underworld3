"""Projecting a discontinuous field to a continuous one averages the cells at a node (#905).

``uw.function.project_to_degree(var, 1, continuous=True)`` on a discontinuous variable
used to return each vertex value multiplied by the number of cells meeting there. The
interpolation matrix gives a continuous node shared by ``k`` cells one copy of each
cell's interpolant, in different source columns, so its row summed to ``k``.
``project_to_vertices`` and ``field_projection.write_vertices_to_viewer`` go through the
same matrix. On development eddab707 the projections below gave ratios of 2 to 7.

- A DG1 field that is linear everywhere projects to its exact P1 values: each cell's
  interpolant agrees at the shared node, so the average is the value.
- A DG0 field of 1 projects to 1 everywhere.
- A DG0 field with a different value in every cell projects, at each node, to the
  unweighted mean of the cells meeting there; a linear or constant field cannot tell a
  full average from a partial one, this can.
- A vector DG1 field projects component by component.
- A continuous source projects exactly as before: its rows already sum to 1, and the
  matrix is not touched.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.function import project_to_degree, project_to_vertices

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.fixture(scope="module")
def mesh():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25, qdegree=3)


def _linear(x):
    return 1.0 + x[:, 0] + 2.0 * x[:, 1]


def test_a_linear_dg1_field_projects_to_its_p1_values(mesh):
    d1 = uw.discretisation.MeshVariable("d1_905", mesh, 1, degree=1, continuous=False)
    d1.array[...] = _linear(np.asarray(d1.coords)).reshape(-1, 1, 1)
    nodes = mesh._get_coords_for_basis(1, True)
    projected = project_to_degree(d1, 1, True)[:, 0]
    assert np.allclose(projected, _linear(nodes), rtol=0.0, atol=1.0e-12)
    # the shorthand is the same projection
    assert np.allclose(project_to_vertices(d1)[:, 0], projected, rtol=0.0, atol=0.0)


def test_a_constant_dg0_field_projects_to_the_constant(mesh):
    d0 = uw.discretisation.MeshVariable("d0_905", mesh, 1, degree=0, continuous=False)
    d0.array[...] = 1.0
    projected = project_to_degree(d0, 1, True)[:, 0]
    assert np.allclose(projected, 1.0, rtol=0.0, atol=1.0e-12)


def test_a_continuous_source_is_projected_exactly_as_before(mesh):
    p2 = uw.discretisation.MeshVariable("p2_905", mesh, 1, degree=2, continuous=True)
    p2.array[...] = _linear(np.asarray(p2.coords)).reshape(-1, 1, 1)
    nodes = mesh._get_coords_for_basis(1, True)
    projected = project_to_degree(p2, 1, True)[:, 0]
    assert np.allclose(projected, _linear(nodes), rtol=0.0, atol=1.0e-12)


def test_a_per_cell_dg0_field_projects_to_the_mean_of_the_cells_at_each_node(mesh):
    d0 = uw.discretisation.MeshVariable("dc_905", mesh, 1, degree=0, continuous=False)
    centroids = np.asarray(d0.coords)
    d0.array[...] = np.sin(7.0 * centroids[:, 0] + 3.0 * centroids[:, 1]).reshape(-1, 1, 1)
    cell_value = {tuple(np.round(c, 12)): v for c, v in zip(centroids, d0.array[:, 0, 0])}

    # the cells meeting at each vertex, from the DM, keyed by coordinates
    dm = mesh.dm
    vs, ve = dm.getDepthStratum(0)
    cs, ce = dm.getHeightStratum(0)
    X = dm.getCoordinatesLocal().array.reshape(-1, 2)
    star = {}
    for c in range(cs, ce):
        verts = [q for q in dm.getTransitiveClosure(c)[0] if vs <= q < ve]
        centre = tuple(np.round(X[[q - vs for q in verts]].mean(axis=0), 12))
        for q in verts:
            star.setdefault(tuple(np.round(X[q - vs], 12)), []).append(cell_value[centre])

    nodes = mesh._get_coords_for_basis(1, True)
    projected = project_to_degree(d0, 1, True)[:, 0]
    expected = np.array([np.mean(star[tuple(np.round(n, 12))]) for n in nodes])
    assert np.allclose(projected, expected, rtol=0.0, atol=1.0e-12)


def test_a_vector_dg1_field_projects_component_by_component(mesh):
    v1 = uw.discretisation.MeshVariable("v1_905", mesh, 2, degree=1, continuous=False)
    x = np.asarray(v1.coords)
    v1.array[:, 0, 0] = _linear(x)
    v1.array[:, 0, 1] = 3.0 - x[:, 0]
    nodes = mesh._get_coords_for_basis(1, True)
    projected = project_to_degree(v1, 1, True)
    assert np.allclose(projected[:, 0], _linear(nodes), rtol=0.0, atol=1.0e-12)
    assert np.allclose(projected[:, 1], 3.0 - nodes[:, 0], rtol=0.0, atol=1.0e-12)
