"""A remapped field stays accurate when the wall moves, and is read only where it exists.

A wall node that slid along a convex boundary sits a sagitta outside the old polygon, and
a wall that moved outward takes nodes further out. The P2 basis overshoots past the wall,
so targets outside the old mesh read the nearest point of the old wall. Everything else
reads the plain FE field. The monotone clamp that stood in for this bounded every value by
its nearest DOFs on one rank: on the annulus slide it took the interior error from 8e-4 to
1e-2 in serial and 5e-2 in parallel, and put 8e-2 between rank counts.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _slide_walls_and_measure():
    """Slide the wall vertices a fifth of a cell. Returns the largest error of the
    remapped field on the wall DOFs and inside."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    field = (r - 0.5) * 2 + 0.5 * sympy.cos(4 * sympy.atan2(y, x)) * r
    T = uw.discretisation.MeshVariable("T0768", mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(field, T.coords).reshape(-1)

    coords = np.array(mesh.X.coords)
    radius = np.linalg.norm(coords, axis=1)
    wall = (np.abs(radius - 1.0) < 1.0e-9) | (np.abs(radius - 0.5) < 1.0e-9)
    angle = 0.2 * (1 / 8) / 0.5
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    coords[wall] = coords[wall] @ rotation
    mesh.deform(coords)

    dof = np.asarray(T.coords).reshape(-1, 2)
    error = np.abs(np.asarray(T.array).reshape(-1) - uw.function.evaluate(field, dof).reshape(-1))
    dof_radius = np.linalg.norm(dof, axis=1)
    on_wall = (dof_radius > 0.99) | (dof_radius < 0.51)
    largest = lambda values: uw.mpi.comm.allreduce(
        float(values.max()) if values.size else 0.0, op=uw.MPI.MAX)
    return largest(error[on_wall]), largest(error[~on_wall])


def test_the_remap_is_accurate_on_and_off_the_wall():
    wall_error, interior_error = _slide_walls_and_measure()
    assert wall_error < 1.0e-2, wall_error       # 2.7e-2 past the wall, 4.4e-3 read inside
    assert interior_error < 2.0e-3, interior_error    # 9.8e-3 under the clamp, 8.3e-4 without


@pytest.mark.parametrize("right_wall, expected", [(0.9, 0.9), (1.1, 1.0)])
def test_a_wall_that_moves_reads_the_field_only_where_it_exists(right_wall, expected):
    """T = x on a unit box. Moving inward, the wall is still inside the old mesh and
    reads the exact value. Moving outward, it is outside and reads the old wall."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 4, qdegree=3)
    T = uw.discretisation.MeshVariable(f"T0768b{right_wall}", mesh, 1, degree=2)
    T.array[:, 0, 0] = np.asarray(T.coords)[:, 0]

    coords = np.array(mesh.X.coords)
    coords[np.abs(coords[:, 0] - 1.0) < 1.0e-9, 0] = right_wall
    mesh.deform(coords)

    dof = np.asarray(T.coords)
    value = np.asarray(T.array).reshape(-1)
    on_wall = np.abs(dof[:, 0] - right_wall) < 1.0e-9
    assert on_wall.any()
    assert np.abs(value[on_wall] - expected).max() < 1.0e-9
    inside_old_mesh = dof[:, 0] < min(right_wall, 1.0) - 1.0e-9
    assert np.abs(value[inside_old_mesh] - dof[inside_old_mesh, 0]).max() < 1.0e-9
