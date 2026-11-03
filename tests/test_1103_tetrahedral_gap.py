"""Independent polynomial, moment and rigid-motion gates for tetrahedral blocks."""

from itertools import combinations
from math import factorial

import numpy as np
import pytest

from underworld3.utilities._tetrahedral_gap import (
    _tetrahedral_p2_basis, _tetrahedron_quadrature, _tetrahedral_stokes_blocks,
)

pytestmark = pytest.mark.level_1


def _vertices():
    return np.array([
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, -2, 3], [3, -2, 3], [2, 1, 3], [1, -1, 7]],
    ], dtype=float)


def _nodes(vertices):
    edges = [(vertices[:, i] + vertices[:, j])/2 for i, j in combinations(range(4), 2)]
    return np.concatenate((vertices, np.stack(edges, axis=1)), axis=1)


@pytest.mark.tier_b
def test_partition_unity_and_quadratic_reproduction():
    vertices = _vertices()
    nodes = _nodes(vertices)
    values, gradients, _ = _tetrahedral_p2_basis(vertices, nodes)
    np.testing.assert_allclose(values, np.broadcast_to(np.eye(10), values.shape), atol=1e-14)
    points, _ = _tetrahedron_quadrature(vertices, 4)
    values, gradients, bary = _tetrahedral_p2_basis(vertices, points)
    np.testing.assert_allclose(values.sum(axis=-1), 1, atol=1e-14)
    np.testing.assert_allclose(gradients.sum(axis=2), 0, atol=1e-14)
    np.testing.assert_allclose(np.einsum("eqi,eid->eqd", bary, vertices), points, atol=1e-14)
    x, y, z = nodes.transpose(2, 0, 1)
    nodal = x*x + x*y + 2*y*y - z*z + 3*z + 4
    x, y, z = points.transpose(2, 0, 1)
    exact = x*x + x*y + 2*y*y - z*z + 3*z + 4
    exact_gradient = np.stack((2*x+y, x+4*y, 3-2*z), axis=-1)
    np.testing.assert_allclose(np.einsum("eqi,ei->eq", values, nodal), exact, atol=1e-13)
    np.testing.assert_allclose(np.einsum("eqid,ei->eqd", gradients, nodal), exact_gradient, atol=1e-13)


@pytest.mark.tier_b
def test_volume_and_polynomial_moments():
    points, weights = _tetrahedron_quadrature(_vertices()[:1], 5)
    for powers in ((0, 0, 0), (1, 0, 0), (2, 1, 0), (1, 1, 2), (0, 0, 4)):
        measured = np.sum(weights[0] * np.prod(points[0]**np.array(powers), axis=1))
        exact = np.prod([factorial(power) for power in powers]) / factorial(sum(powers) + 3)
        np.testing.assert_allclose(measured, exact, rtol=2e-14)
    vertices = _vertices()[1:]
    points, weights = _tetrahedron_quadrature(vertices, 4)
    volume = 4.0  # det(diag(2,3,4))/6
    np.testing.assert_allclose(weights.sum(), volume, rtol=1e-14)
    np.testing.assert_allclose(np.einsum("eq,eqd->ed", weights, points),
                               volume*vertices.mean(axis=1), rtol=1e-14)


@pytest.mark.tier_b
def test_stiffness_energy_symmetry_and_six_rigid_modes():
    vertices = _vertices()
    points, weights = _tetrahedron_quadrature(vertices, 4)
    blocks = _tetrahedral_stokes_blocks(vertices, points, weights)
    np.testing.assert_allclose(blocks["A"], blocks["A"].swapaxes(1, 2), atol=2e-14)
    nodes = _nodes(vertices)
    for axis in np.eye(3):
        rotation = np.cross(axis, nodes).reshape(-1, 30)
        translation = np.broadcast_to(axis, nodes.shape).reshape(-1, 30)
        for velocity in (rotation, translation):
            np.testing.assert_allclose(np.einsum("eij,ej->ei", blocks["A"], velocity), 0, atol=1e-13)
            np.testing.assert_allclose(np.einsum("eij,ej->ei", blocks["B"], velocity), 0, atol=1e-13)
    matrix = np.array([[2, -1, 3], [4, 1, 0], [2, 5, -2]], dtype=float)
    velocity = np.einsum("ij,enj->eni", matrix, nodes).reshape(-1, 30)
    strain = (matrix + matrix.T)/2
    energy = np.einsum("ei,eij,ej->e", velocity, blocks["A"], velocity)
    np.testing.assert_allclose(energy, 2*np.sum(strain*strain)*weights.sum(axis=1), rtol=2e-14)
    np.testing.assert_allclose(np.einsum("eij,ej->ei", blocks["B"], velocity),
                               -np.trace(matrix)*blocks["pressure_mean"], atol=2e-13)


@pytest.mark.tier_b
def test_constant_pressure_divergence_matches_boundary_flux():
    vertices = _vertices()
    points, weights = _tetrahedron_quadrature(vertices, 4)
    blocks = _tetrahedral_stokes_blocks(vertices, points, weights)
    flux = np.zeros((len(vertices), 10, 3))
    edge_pairs = list(combinations(range(4), 2))
    for omitted in range(4):
        face = [i for i in range(4) if i != omitted]
        triangle = vertices[:, face]
        area_normal = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])/2
        inward = vertices[:, omitted] - triangle.mean(axis=1)
        area_normal *= np.where(np.einsum("ed,ed->e", area_normal, inward) > 0, -1, 1)[:, None]
        # P2 vertex functions integrate to zero on a face; its edge functions
        # each integrate to one third of the face area.
        for edge in combinations(face, 2):
            flux[:, 4 + edge_pairs.index(edge)] += area_normal/3
    np.testing.assert_allclose(blocks["B"].sum(axis=1), -flux.reshape(-1, 30), atol=1e-13)
    np.testing.assert_allclose(
        blocks["pressure_mean"],
        np.broadcast_to(weights.sum(axis=1)[:, None]/4, blocks["pressure_mean"].shape),
        atol=1e-14,
    )


@pytest.mark.tier_b
def test_rotation_integrals_reproduce_rigid_field_gram_matrix():
    vertices = _vertices()
    points, weights = _tetrahedron_quadrature(vertices, 4)
    blocks = _tetrahedral_stokes_blocks(vertices, points, weights)
    nodes = _nodes(vertices)
    nodal_rotations = np.stack([np.cross(axis, nodes) for axis in np.eye(3)], axis=-1).reshape(-1, 30, 3)
    quadrature_rotations = np.stack([np.cross(axis, points) for axis in np.eye(3)], axis=-1)
    gram = np.einsum("eim,ein->emn", nodal_rotations, blocks["rotation"])
    exact = np.einsum("eq,eqdm,eqdn->emn", weights, quadrature_rotations, quadrature_rotations)
    np.testing.assert_allclose(gram, exact, rtol=2e-14, atol=1e-13)
    np.testing.assert_allclose(blocks["unit_force"].sum(axis=1), weights.sum(axis=1), atol=1e-14)


@pytest.mark.tier_b
def test_reversed_orientation_and_empty_batches():
    vertices = _vertices()[:, [0, 2, 1, 3]]
    points, weights = _tetrahedron_quadrature(vertices, 4)
    np.testing.assert_allclose(weights.sum(axis=1), [1/6, 4], rtol=1e-14)
    empty = np.empty((0, 4, 3))
    points, weights = _tetrahedron_quadrature(empty, 3)
    blocks = _tetrahedral_stokes_blocks(empty, points, weights)
    assert blocks["A"].shape == (0, 30, 30)
    assert blocks["rotation"].shape == (0, 30, 3)
