"""Exact affine tetrahedral jump data and orientation before spherical extension.

These tests concern the fitted planar interface only. They do not validate
true-sphere integration, spherical pressure lifting or the Stokes solve.
"""

import numpy as np
import pytest
import sympy as sp
import underworld3 as uw
from mpi4py import MPI

from test_1099_general_pressure_split import _check_lift, _stokes


pytestmark = pytest.mark.level_1


@pytest.mark.tier_b
@pytest.mark.parametrize("representation", ["analytical", "numerical", "callable"])
@pytest.mark.parametrize("orientation", [1, -1])
def test_tetrahedral_affine_jump_data_and_orientation(representation, orientation):
    """P1 facet traces reproduce a prescribed affine jump in either orientation."""
    mesh = uw.meshing.BoxInternalBoundary(
        minCoords=(0, 0, 0), maxCoords=(1, 1, 1), zintCoord=0.5,
        cellSize=0.35, degree=1, qdegree=3, simplex=True,
    )
    stokes = _stokes(mesh)
    x, y, _ = mesh.X
    if representation == "analytical":
        jump = 1 + 2*x - 3*y
    elif representation == "numerical":
        jump = uw.discretisation.MeshVariable("AffineJumpData", mesh, 1, degree=1)
        coordinates = np.asarray(jump.coords)
        jump.array[:, 0, 0] = 1 + 2*coordinates[:, 0] - 3*coordinates[:, 1]
    else:
        def jump(coordinates):
            return 1 + 2*coordinates[:, 0] - 3*coordinates[:, 1]

    split = stokes.pressure_split(
        jumps={"Internal": jump}, normals={"Internal": (0, 0, orientation)},
    )
    assert split.checks["interface_facets"] > 0
    _check_lift(mesh, split, {
        "Internal": (
            lambda coordinates: orientation * (
                1 + 2*coordinates[:, 0] - 3*coordinates[:, 1]
            ),
            2,
        ),
    })


@pytest.mark.tier_b
def test_tetrahedral_joint_jump_normal_reversal_preserves_lift():
    """Reversing both jump and orientation represents the same discontinuity."""
    mesh = uw.meshing.BoxInternalBoundary(
        minCoords=(0, 0, 0), maxCoords=(1, 1, 1), zintCoord=0.5,
        cellSize=0.35, degree=1, qdegree=3, simplex=True,
    )
    stokes = _stokes(mesh)
    x, y, _ = mesh.X
    forward = stokes.pressure_split(
        jumps={"Internal": 1 + 2*x - 3*y}, normals={"Internal": (0, 0, 1)},
    )
    saved_lift = np.array(forward.pressure_jump_field.array)
    reverse = stokes.pressure_split(
        jumps={"Internal": -1 - 2*x + 3*y}, normals={"Internal": (0, 0, -1)},
    )
    np.testing.assert_allclose(
        reverse.pressure_jump_field.array, saved_lift, rtol=0, atol=1e-12,
    )
    np.testing.assert_array_equal(forward.pressure_jump_field.array, saved_lift)
    np.testing.assert_allclose(
        reverse.regular_force.array, forward.regular_force.array,
        rtol=0, atol=1e-12,
    )


def _check_closed_interface_traces(mesh, split):
    """Check all two-cell facet traces, oriented by independently labelled volumes.

    The interface is polyhedral: affine jumps are interpolated at its vertices.
    Nothing here substitutes a true-sphere surface integral for those facets.
    """
    dm = mesh.dm
    first_cell, _ = dm.getHeightStratum(0)
    rows = mesh._cell_node_indices(1, False).reshape(-1, 4)
    coordinates = np.asarray(split.pressure_jump_field.coords)[rows]
    design = np.concatenate((np.ones((*coordinates.shape[:2], 1)), coordinates), axis=2)
    coefficients = np.linalg.solve(
        design, split.pressure_jump_field.array[rows, 0, 0][..., None],
    )[..., 0]
    # A cell outside a fitted chordal interface can still have |centroid| < R.
    # Material-volume labels identify sides without imposing true-sphere geometry.
    outer = mesh.cells_labelled("Outer", mesh.regions.Outer.value)
    inner = mesh.cells_labelled("Inner", mesh.regions.Inner.value)
    assert np.all(outer != inner)
    first_vertex, last_vertex = dm.getDepthStratum(0)
    first_facet, last_facet = dm.getHeightStratum(1)
    coordinate_section = dm.getCoordinateSection()
    vertex_coordinates = dm.getCoordinatesLocal().array
    maximum_error, checked_interface_facets = 0.0, 0
    for facet in range(first_facet, last_facet):
        adjacent = dm.getSupport(facet)
        if len(adjacent) != 2:
            continue  # Exterior walls and rank seams have no local paired trace.
        vertices = [
            vertex for vertex in dm.getTransitiveClosure(facet)[0]
            if first_vertex <= vertex < last_vertex
        ]
        points = np.array([
            vertex_coordinates[
                coordinate_section.getOffset(vertex):coordinate_section.getOffset(vertex) + 3
            ] for vertex in vertices
        ])
        left, right = np.asarray(adjacent, dtype=int) - first_cell
        expected = np.zeros(len(points))
        if dm.getLabel("Internal").getValue(facet) == mesh.boundaries.Internal.value:
            assert outer[left] != outer[right]
            expected = (int(outer[right]) - int(outer[left])) * (
                1 + 2*points[:, 0] - 3*points[:, 1]
            )
            checked_interface_facets += 1
        facet_design = np.column_stack((np.ones(len(points)), points))
        error = facet_design @ (coefficients[right] - coefficients[left]) - expected
        maximum_error = max(maximum_error, float(np.max(np.abs(error))))
    assert uw.mpi.comm.allreduce(checked_interface_facets, op=MPI.SUM) > 0
    assert uw.mpi.comm.allreduce(maximum_error, op=MPI.MAX) < 1e-10
    native_integral = uw.maths.Integral(mesh, split.pressure_jump).evaluate()
    assert native_integral == pytest.approx(split.pressure_jump_integral, abs=1e-10)


@pytest.mark.tier_b
def test_closed_spherical_polyhedral_affine_jump_and_reversal():
    """A closed fitted interface retains its prescribed jump and no other jumps."""
    mesh = uw.meshing.SphericalShellInternalBoundary(
        radiusInner=1.22, radiusInternal=2, radiusOuter=2.22,
        cellSize=1/8, degree=1, qdegree=3,
    )
    stokes = _stokes(mesh)
    x, y, z = mesh.X
    radius = (x*x + y*y + z*z)**sp.Rational(1, 2)
    outward = (x/radius, y/radius, z/radius)
    forward = stokes.pressure_split(
        jumps={"Internal": 1 + 2*x - 3*y}, normals={"Internal": outward},
    )
    _check_closed_interface_traces(mesh, forward)
    saved_lift = np.array(forward.pressure_jump_field.array)
    reverse = stokes.pressure_split(
        jumps={"Internal": -1 - 2*x + 3*y},
        normals={"Internal": tuple(-component for component in outward)},
    )
    np.testing.assert_allclose(
        reverse.pressure_jump_field.array, saved_lift, rtol=0, atol=1e-12,
    )
    np.testing.assert_array_equal(forward.pressure_jump_field.array, saved_lift)
    np.testing.assert_allclose(
        reverse.regular_force.array, forward.regular_force.array,
        rtol=0, atol=1e-12,
    )
