"""Independent pressure-extraction signs, radial lift and circular load moments."""

import numpy as np
import pytest
import underworld3 as uw
from underworld3.utilities._cylindrical_pressure import (
    _extended_regular_force as extended_regular_force,
    _load_moments as load_moments,
    _p2_shapes as p2_shapes,
    _profile_values as profile_values,
)

pytestmark = pytest.mark.level_1


@pytest.mark.tier_b
def test_native_pressure_lift_reproduces_positive_prescribed_jump():
    """The standalone Kramer sign has [p]=+cos(n theta), hence amplitude=-1."""
    mesh = uw.meshing.AnnulusInternalBoundary(
        radiusOuter=2.22, radiusInternal=2.0, radiusInner=1.22,
        cellSize_Inner=0.25, cellSize_Internal=0.25, cellSize_Outer=0.25,
        qdegree=2, degree=1,
    )
    velocity = uw.discretisation.MeshVariable("LiftVelocity", mesh, 2, degree=2)
    pressure = uw.discretisation.MeshVariable("LiftPressure", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
    lift = stokes.cylindrical_pressure_lifting(
        interface_radius=2.0, outer_radius=2.22, wavenumber=2,
        amplitude=-1.0, quadrature_points=8,
    )
    points = np.array([[2.0 + 1e-7, 0], [1.9, 0], [2.22, 0]])
    values = uw.function.evaluate(lift.pressure_jump, points).reshape(-1)
    np.testing.assert_allclose(values, [1 - 1e-7 / 0.22, 0, 0], rtol=1e-12, atol=1e-12)
    assert lift.checks["internal_chord_count"] > 0
    assert lift.checks["maximum_load_moment_error"] < 1e-11
    assert lift.checks["native_force_integral_error"] < 1e-10

@pytest.mark.tier_b
def test_profile_has_prescribed_jump_and_vanishes_at_outer_radius():
    value, derivative = profile_values(np.array([2.0, 2.3, 2.4]), 2.0, 0.3)
    np.testing.assert_allclose(value, [1, 0, -1 / 3], atol=1e-14)
    np.testing.assert_allclose(derivative, -1 / 0.3, atol=1e-14)


@pytest.mark.tier_b
def test_regular_force_is_positive_gradient_of_extracted_profile():
    points = np.array([[2.04, 0.03], [1.98, 0.5], [-2.05, 0.1]])
    actual = extended_regular_force(points, 2.0, 0.3, 2, 1.7)
    numerical = np.empty_like(actual)
    for component in range(2):
        displacement = np.eye(2)[component] * 1e-6
        values = []
        for shifted in (points + displacement, points - displacement):
            radius = np.linalg.norm(shifted, axis=1)
            profile, _ = profile_values(radius, 2.0, 0.3)
            values.append(1.7 * profile * np.cos(2 * np.arctan2(shifted[:, 1], shifted[:, 0])))
        numerical[:, component] = (values[0] - values[1]) / 2e-6
    np.testing.assert_allclose(actual, numerical, rtol=1e-8, atol=1e-8)


@pytest.mark.tier_b
def test_triangle_load_against_independent_polar_domain_integration():
    radius, alpha = 2.0, 0.05
    vertices = np.array(
        [
            [radius * np.cos(alpha), -radius * np.sin(alpha)],
            [radius * np.cos(alpha), radius * np.sin(alpha)],
            [2.15, 0],
        ]
    )
    moments, cap_area, chords, _ = load_moments(vertices[None], radius, 0.4, 2, 1.0, 12)
    edges = np.column_stack((vertices[1] - vertices[0], vertices[2] - vertices[0]))
    inverse = np.linalg.inv(edges)
    nodes, weights = np.polynomial.legendre.leggauss(24)
    nodes, weights = (nodes + 1) / 2, weights / 2
    independent = np.zeros((6, 2))
    pressure_divergence = 0.0
    for sign in (-1, 1):
        endpoint = vertices[0 if sign < 0 else 1]
        edge = vertices[2] - endpoint
        determinant = endpoint[0] * vertices[2, 1] - endpoint[1] * vertices[2, 0]
        for t, wt in zip(nodes, weights):
            theta = sign * alpha * t
            direction = np.array([np.cos(theta), np.sin(theta)])
            upper = determinant / (direction[0] * edge[1] - direction[1] * edge[0])
            for s, ws in zip(nodes, weights):
                r = radius + s * (upper - radius)
                point = r * direction
                xi = inverse @ (point - vertices[0])
                bary = np.r_[1 - xi.sum(), xi]
                force = extended_regular_force(point, radius, 0.4, 2, 1.0)
                independent += (
                    r * (upper - radius) * alpha * wt * ws * p2_shapes(bary)[:, None] * force
                )
                profile, _ = profile_values(r, radius, 0.4)
                pressure_divergence += (
                    r * (upper - radius) * alpha * wt * ws * profile * np.cos(2 * theta)
                )
    np.testing.assert_allclose(moments[0], independent, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(cap_area, radius**2 * (alpha - np.sin(2 * alpha) / 2), rtol=1e-10)
    np.testing.assert_array_equal(chords, [1])
    # Independently verify the weak sign and boundary term using v=(x,0).
    boundary_flux = 0.0
    for first, second in ((1, 2), (2, 0)):
        edge = vertices[second] - vertices[first]
        normal_ds = np.array([-edge[1], edge[0]])  # clockwise triangle
        for t, wt in zip(nodes, weights):
            point = vertices[first] + t * edge
            profile, _ = profile_values(np.linalg.norm(point), radius, 0.4)
            boundary_flux += (
                wt * profile * np.cos(2 * np.arctan2(point[1], point[0])) * point[0] * normal_ds[0]
            )
    for t, wt in zip(nodes, weights):
        theta = alpha * (2 * t - 1)
        boundary_flux -= 2 * alpha * wt * radius**2 * np.cos(2 * theta) * np.cos(theta) ** 2
    x_dofs = np.r_[vertices[:, 0], (vertices[[0, 1, 2], 0] + vertices[[1, 2, 0], 0]) / 2]
    np.testing.assert_allclose(
        moments[0, :, 0] @ x_dofs, boundary_flux - pressure_divergence, rtol=1e-9, atol=1e-12
    )
