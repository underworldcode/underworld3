"""Independent spherical area, volume and moment gates for radial gap rules."""

from itertools import product

import numpy as np
import pytest

from underworld3.utilities._spherical_gap import _spherical_facet_quadrature

pytestmark = pytest.mark.level_1


def _octahedron(radius: float) -> np.ndarray:
    return radius * np.asarray([np.diag(signs) for signs in product((-1, 1), repeat=3)])


@pytest.mark.tier_b
@pytest.mark.parametrize("radius", [1.0, 2.22])
def test_closed_sphere_area_volume_and_moments(radius):
    rule = _spherical_facet_quadrature(_octahedron(radius), radius, 24, 4)
    area = 4 * np.pi * radius**2
    volume = 4 * np.pi * radius**3 / 3
    octahedron_volume = 4 * radius**3 / 3
    np.testing.assert_allclose(rule["surface_weights"].sum(), area, rtol=2e-12)
    np.testing.assert_allclose(rule["gap_weights"].sum(), volume - octahedron_volume, rtol=2e-12)
    # Sphere radial second moment minus the octahedron's 2 R**5 / 5.
    radial_second_moment = np.sum(
        rule["gap_weights"] * np.sum(rule["gap_points"]**2, axis=-1)
    )
    np.testing.assert_allclose(radial_second_moment,
                               (4 * np.pi - 2) * radius**5 / 5, rtol=2e-12)
    x = rule["surface_points"]
    np.testing.assert_allclose(np.linalg.norm(x, axis=-1), radius, rtol=1e-14)
    np.testing.assert_allclose(np.einsum("fq,fqi->i", rule["surface_weights"], x), 0, atol=area * radius * 1e-14)
    moment = np.einsum("fq,fqi,fqj->ij", rule["surface_weights"], x, x)
    np.testing.assert_allclose(moment, np.eye(3) * area * radius**2 / 3, atol=area * radius**2 * 2e-12)
    # Divergence theorem for F=x/3 over the gap: the spherical flux minus
    # planar flux equals the volume between them.
    planar_flux = np.sqrt(3) * radius / 3 * rule["planar_weights"].sum() / 3
    np.testing.assert_allclose(area * radius / 3 - planar_flux,
                               rule["gap_weights"].sum(), rtol=2e-12)


@pytest.mark.tier_b
def test_orientation_empty_batches_and_refinement():
    facets = _octahedron(1.0)
    forward = _spherical_facet_quadrature(facets, 1.0, 24, 3)
    reverse = _spherical_facet_quadrature(facets[:, ::-1], 1.0, 24, 3)
    np.testing.assert_allclose(forward["gap_weights"].sum(), reverse["gap_weights"].sum(), rtol=1e-14)
    coarse = _spherical_facet_quadrature(facets, 1.0, 4, 3)
    assert abs(coarse["surface_weights"].sum() - 4 * np.pi) > 1e-5
    empty = _spherical_facet_quadrature(np.empty((0, 3, 3)), 1.0, 4, 3)
    assert empty["surface_points"].shape == (0, 16, 3)
    assert empty["gap_points"].shape == (0, 16, 3, 3)
    assert empty["gap_weights"].sum() == 0


@pytest.mark.tier_b
def test_radial_order_seven_integrates_degree_thirteen_load_at_fixed_angles():
    # x**11 represents degree-nine force times a P2 test function. The radial
    # volume Jacobian raises the polynomial degree to thirteen. Fix the angular
    # rule in both the numerical and independently integrated radial moments.
    radius = 2.22
    facets = radius*np.eye(3)[None]
    fine = _spherical_facet_quadrature(facets, radius, 8, 7)
    distance = np.linalg.norm(fine["planar_points"], axis=-1)
    nx = fine["surface_normals"][..., 0]
    exact_radial_moment = np.sum(
        fine["surface_weights"]/radius**2 * nx**11
        * (radius**14-distance**14)/14
    )
    fine_moment = np.sum(fine["gap_weights"]*fine["gap_points"][..., 0]**11)
    np.testing.assert_allclose(fine_moment, exact_radial_moment, rtol=5e-14)
    coarse = _spherical_facet_quadrature(facets, radius, 8, 4)
    coarse_moment = np.sum(coarse["gap_weights"]*coarse["gap_points"][..., 0]**11)
    assert abs(coarse_moment-exact_radial_moment) > 1e-8*abs(exact_radial_moment)


@pytest.mark.tier_b
@pytest.mark.parametrize("facets,radius,surface_order,radial_order", [
    (np.zeros((1, 3, 2)), 1, 4, 3),
    (_octahedron(1), 2, 4, 3),
    (_octahedron(1), -1, 4, 3),
    (_octahedron(1), 1, 0, 3),
    (_octahedron(1), 1, 4, 1.5),
    (np.array([[[1, 0, 0], [0, 1, 0], [-1, 0, 0]]]), 1, 4, 3),
])
def test_invalid_geometry_and_orders(facets, radius, surface_order, radial_order):
    with pytest.raises(ValueError):
        _spherical_facet_quadrature(facets, radius, surface_order, radial_order)
