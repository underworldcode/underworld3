"""Local curtain geometry and consistency identities, without stability claims."""

from itertools import combinations

import numpy as np
import pytest

from underworld3.utilities._spherical_curtain import (
    _spherical_curtain_quadrature, _curtain_stokes_blocks,
)
from underworld3.utilities._spherical_gap import _spherical_facet_quadrature
from underworld3.utilities._tetrahedral_gap import _tetrahedral_p2_basis

pytestmark = pytest.mark.level_1


def _nodes(vertices):
    return np.vstack((vertices, [(vertices[i]+vertices[j])/2 for i, j in combinations(range(4), 2)]))


def _geometry():
    a, b = np.array([1., 0, 0]), np.array([0., 1, 0])
    interior = np.array([.2, .2, .1])
    return np.array([a, b, [0, 0, 1], interior]), np.array([a, b, [0, 0, -1], interior])


@pytest.mark.tier_b
@pytest.mark.parametrize("radius", [1., 2.22])
def test_curtain_sector_minus_triangle_area_and_first_moment(radius):
    angle = 0.7
    a, b = radius*np.array([1., 0, 0]), radius*np.array([np.cos(angle), np.sin(angle), 0])
    rule = _spherical_curtain_quadrature(np.array([a, b]), np.array([0., 0, 1]), radius, 24, 4)
    area = (radius**2*angle-np.linalg.norm(np.cross(a, b)))/2
    sector_moment = radius**3/3*np.array([np.sin(angle), 1-np.cos(angle), 0])
    triangle_moment = np.linalg.norm(np.cross(a, b))*(a+b)/6
    np.testing.assert_allclose(rule["weights"].sum(), area, rtol=2e-14)
    np.testing.assert_allclose(np.einsum("q,qd->d", rule["weights"], rule["points"]),
                               sector_moment-triangle_moment, rtol=2e-14, atol=1e-15)
    np.testing.assert_allclose(rule["normal"], [0, 0, -1], atol=1e-15)
    reverse = _spherical_curtain_quadrature(np.array([b, a]), np.array([0., 0, 1]), radius, 24, 4)
    np.testing.assert_allclose(reverse["normal"], rule["normal"], atol=1e-15)
    np.testing.assert_allclose(reverse["weights"].sum(), area, rtol=2e-14)


@pytest.mark.tier_b
@pytest.mark.parametrize("sign", [-1, 1])
def test_pressure_consistency_closes_curtain_jump_of_conforming_affine_field(sign):
    vertices_a, vertices_b = _geometry()
    rule = _spherical_curtain_quadrature(vertices_a[:2], vertices_a[:3].mean(axis=0), 1., 24, 4)
    blocks = _curtain_stokes_blocks(vertices_a, vertices_b, rule["points"], rule["weights"], rule["normal"], sign)
    # This affine field vanishes on the common mesh face (a,b,interior),
    # yet its continuation generally does not vanish on the radial curtain.
    mesh_normal = np.cross(vertices_a[1]-vertices_a[0], vertices_a[3]-vertices_a[0])
    nodes_a = _nodes(vertices_a)
    scalar = (nodes_a-vertices_a[0])@mesh_normal
    velocity = np.concatenate(((scalar[:, None]*rule["normal"]).ravel(), np.zeros(30)))
    flux = np.ones(8)@blocks["B"]@velocity
    area = np.pi/4-.5
    moment = np.array([1/6, 1/6, 0])  # quarter-disk moment minus triangle moment
    exact_flux = sign*mesh_normal@(moment-area*vertices_a[0])
    assert abs(exact_flux) > 1e-4
    np.testing.assert_allclose(flux, exact_flux, rtol=2e-13)
    np.testing.assert_allclose(blocks["A"], blocks["A"].T, atol=1e-14)


@pytest.mark.tier_b
def test_consistency_preserves_globally_rigid_rotations():
    vertices_a, vertices_b = _geometry()
    rule = _spherical_curtain_quadrature(vertices_a[:2], vertices_a[:3].mean(axis=0), 1., 24, 4)
    blocks = _curtain_stokes_blocks(vertices_a, vertices_b, rule["points"], rule["weights"], rule["normal"], 1)
    nodes = np.vstack((_nodes(vertices_a), _nodes(vertices_b)))
    for axis in np.eye(3):
        rigid = np.cross(axis, nodes).ravel()
        np.testing.assert_allclose(blocks["jump"]@rigid, 0, atol=1e-14)
        np.testing.assert_allclose(blocks["average_traction"]@rigid, 0, atol=1e-14)
        np.testing.assert_allclose(blocks["A"]@rigid, 0, atol=1e-14)
        np.testing.assert_allclose(blocks["B"]@rigid, 0, atol=1e-14)


@pytest.mark.tier_b
def test_one_gap_divergence_theorem_includes_all_three_curtains():
    vertices_a, _ = _geometry()
    triangle = vertices_a[:3]
    rule = _spherical_facet_quadrature(triangle[None], 1., 24, 4)
    gap_points = rule["gap_points"].reshape(1, -1, 3)
    _, gradients, _ = _tetrahedral_p2_basis(vertices_a[None], gap_points)
    volume_divergence = np.einsum("q,qid->id", rule["gap_weights"].ravel(), gradients[0])
    shape, _, _ = _tetrahedral_p2_basis(vertices_a[None], rule["surface_points"])
    spherical_flux = np.einsum("q,qi,qd->id", rule["surface_weights"][0], shape[0], rule["surface_normals"][0])
    shape, _, _ = _tetrahedral_p2_basis(vertices_a[None], rule["planar_points"])
    planar_normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
    planar_normal /= np.linalg.norm(planar_normal)
    if planar_normal@triangle.mean(axis=0) < 0:
        planar_normal *= -1
    planar_flux = np.einsum("q,qi,d->id", rule["planar_weights"][0], shape[0], planar_normal)
    curtain_flux = np.zeros((10, 3))
    for pair in combinations(range(3), 2):
        curtain = _spherical_curtain_quadrature(triangle[list(pair)], triangle.mean(axis=0), 1., 24, 4)
        shape, _, _ = _tetrahedral_p2_basis(vertices_a[None], curtain["points"][None])
        curtain_flux += np.einsum("q,qi,d->id", curtain["weights"], shape[0], curtain["normal"])
    np.testing.assert_allclose(volume_divergence, spherical_flux-planar_flux+curtain_flux, atol=2e-12)
