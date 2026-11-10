"""Physical tetrahedron/sphere intersections, independent of facet ownership."""

import json
from pathlib import Path

import numpy as np
import pytest

from underworld3.utilities._spherical_cut_cell import (
    _inner_cell_relation, _spherical_cut_cell_quadrature, _spherical_cut_cell_surface,
)
from underworld3.utilities._spherical_gap import _spherical_facet_quadrature
from underworld3.utilities._tetrahedral_gap import (
    _tetrahedral_p2_basis, _tetrahedron_quadrature, _tetrahedral_stokes_blocks,
)

pytestmark = pytest.mark.level_1


@pytest.mark.tier_b
def test_wholly_void_tetrahedron_has_exactly_zero_physical_support():
    # The ball is convex: every point of this tetrahedron lies inside it.
    vertices = np.array([
        [-.03829638026620944, .0615853167629137, 1.2178426154547708],
        [.01951163053887158, .05779929358888627, 1.218473856073383],
        [-.0446747668350843, -.03366640294686156, 1.2187168409933657],
        [0., 0., 1.22],
    ])
    rule = _spherical_cut_cell_quadrature(vertices, 1.22, 12, 4)
    assert rule["volume_weights"].size == 0
    assert rule["surface_weights"].size == 0


@pytest.mark.tier_b
@pytest.mark.parametrize("radius", [.4, 1.3])
def test_octant_tetrahedron_volume_surface_and_quadratic_moment(radius):
    # For R>2/sqrt(3), the sphere protrudes through x+y+z=2 in one cap;
    # R<sqrt(2) keeps this cap away from all three coordinate planes.
    vertices = np.vstack((np.zeros(3), 2*np.eye(3)))
    rule = _spherical_cut_cell_quadrature(vertices, radius, 28, 4)
    height = max(0., radius-2/np.sqrt(3))
    cap_volume = np.pi*height**2*(radius-height/3)
    volume = 8/6-np.pi*radius**3/6+cap_volume
    area = np.pi*radius**2/2-2*np.pi*radius*height
    np.testing.assert_allclose(rule["volume_weights"].sum(), volume, rtol=3e-12)
    np.testing.assert_allclose(rule["surface_weights"].sum(), area, rtol=3e-12)
    if height == 0:
        moment = np.sum(rule["volume_weights"]*rule["volume_points"][:, 0]**2)
        np.testing.assert_allclose(moment, 2**5/60-np.pi*radius**5/30, rtol=3e-12)
    blocks = _tetrahedral_stokes_blocks(
        vertices[None], rule["volume_points"][None], rule["volume_weights"][None])
    # Positive physical strain energy; six affine rigid motions have zero strain.
    eigenvalues = np.linalg.eigvalsh(blocks["A"][0])
    assert eigenvalues.min() >= -2e-12*eigenvalues.max()
    assert np.count_nonzero(eigenvalues > 2e-12*eigenvalues.max()) == 24


@pytest.mark.tier_b
def test_volume_divergence_equals_flat_and_spherical_boundary_flux():
    vertices = np.vstack((np.zeros(3), 2*np.eye(3)))
    rule = _spherical_cut_cell_quadrature(vertices, 1.3, 28, 4)
    # F=x: div(F)=3. The physical inner wall normal is -x/R.
    flat = np.sum(rule["face_weights"]*np.einsum(
        "qi,qi->q", rule["face_points"], rule["face_normals"]))
    sphere = -1.3*rule["surface_weights"].sum()
    np.testing.assert_allclose(flat+sphere, 3*rule["volume_weights"].sum(), rtol=3e-12)


@pytest.mark.tier_b
def test_cell_classification_checks_faces_and_edges_not_only_vertex_radii():
    void = .2*np.vstack((np.zeros(3), np.eye(3)))
    physical = np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0], [2, 0, 1.]])
    # All four corners are outside the ball, but the edge from (-2,0,0)
    # to (2,0,0) crosses it. Vertex-only classification would miss this cut.
    crossing = np.array([[-2, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 3.]])
    np.testing.assert_array_equal(_inner_cell_relation(np.stack((void, physical, crossing)), 1.),
                                  [-1, 1, 0])
    assert _inner_cell_relation(np.empty((0, 4, 3)), 1.).shape == (0,)


@pytest.mark.tier_b
@pytest.mark.parametrize("components", [1, 3])
def test_wall_rule_splits_piecewise_multiplier_before_quadrature(components):
    radius = .4
    a, b, c = np.eye(3)
    midpoint = (a+b)/np.sqrt(2)
    facets = radius*np.array([[a, midpoint, c], [midpoint, b, c]])
    lids = np.arange(12*components).reshape(2, 6*components)
    vertices = np.vstack((np.zeros(3), 2*np.eye(3)))
    points, weights, ids, values = _spherical_cut_cell_surface(
        vertices, radius, 28, facets, lids)
    reference = _spherical_facet_quadrature(facets, radius, 48, 4)
    for facet, rows, y, w in zip(facets, lids, reference["planar_points"],
                                reference["surface_weights"], strict=True):
        bary = y@np.linalg.inv(facet.T).T
        basis = np.column_stack((bary*(2*bary-1), 4*bary[:, 0]*bary[:, 1],
                                 4*bary[:, 0]*bary[:, 2], 4*bary[:, 1]*bary[:, 2]))
        if components == 3:
            basis *= radius/np.linalg.norm(y, axis=1)[:, None]
        selected = np.all(ids == rows, axis=1)
        computed = np.sum(weights[selected, None]*values[selected], axis=0)
        expected = np.sum(w[:, None]*basis, axis=0)
        # Independent Duffy-surface and clipped polar-cone parameterizations.
        np.testing.assert_allclose(computed, expected, rtol=3e-11, atol=3e-13*radius**2)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), radius, rtol=2e-14)


@pytest.mark.tier_b
def test_original_h16_negative_mode_has_nonnegative_physical_strain_energy():
    # The original operator's independently verified energy was -13.36072438.
    # This test asserts the physical strain-energy identity, not a preference
    # between solver policies or an iteration-dependent field comparison.
    data = json.loads((Path(__file__).parent/"data/spherical_void_patch.json").read_text())
    vertices = np.asarray(data["vertices"])
    coefficients = np.asarray(data["velocity_coefficients"]).reshape(-1, 10, 3)
    radius = data["radius_inner"]
    relation = _inner_cell_relation(vertices, radius)
    assert np.count_nonzero(relation == -1) == 1
    for cell, (v, nodal) in enumerate(zip(vertices, coefficients, strict=True)):
        if relation[cell] == 1:
            points, weights = _tetrahedron_quadrature(v[None], 4)
        else:
            rule = _spherical_cut_cell_quadrature(v, radius, 12, 7)
            points, weights = rule["volume_points"][None], rule["volume_weights"][None]
        if weights.size == 0:
            assert relation[cell] == -1
            continue
        _, gradient, _ = _tetrahedral_p2_basis(v[None], points)
        derivative = np.einsum("eqid,ic->eqcd", gradient, nodal)
        strain = (derivative+derivative.swapaxes(-1, -2))/2
        density = 2*weights*np.sum(strain*strain, axis=(-1, -2))
        assert density.sum() >= -128*np.finfo(float).eps*np.abs(density).sum()
