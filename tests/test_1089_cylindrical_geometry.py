"""Hard mathematical baselines for affine annulus geometry corrections.

No Stokes model is run. Independent polar integration verifies circular cuts;
symbolic polynomial stresses verify derivatives; exact line masses verify
reaction and compatibility moments. Run serial or on multiple MPI ranks.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

from underworld3.utilities._cylindrical_pressure import (
    _extended_regular_force as extended_regular_force,
    _load_moments as load_moments,
    _p2_shapes as p2_shapes,
    _profile_values as profile_values,
)
from underworld3.utilities._cylindrical_wall import (
    _numerical_trace as numerical_trace,
    _solve_with_wall_updates,
    _wall_solve_succeeded as wall_solve_succeeded,
    _zero_torque_density as zero_torque_density,
)
from underworld3.utilities._cylindrical_reactions import (
    _circular_reaction_field as circular_reaction_field,
)

pytestmark = pytest.mark.level_1


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


@pytest.mark.tier_b
def test_converged_zero_increment_warm_start_is_not_a_ksp_failure():
    assert wall_solve_succeeded(dict(converged=True, ksp_reason=0, nonlinear_iterations=0))
    assert wall_solve_succeeded(dict(converged=True, ksp_reason=2, nonlinear_iterations=1))
    assert not wall_solve_succeeded(dict(converged=False, ksp_reason=0, nonlinear_iterations=0))
    assert not wall_solve_succeeded(dict(converged=True, ksp_reason=0, nonlinear_iterations=1))
    assert not wall_solve_succeeded(dict(converged=True, ksp_reason=-3, nonlinear_iterations=1))


@pytest.mark.tier_b
def test_p2_p1_wall_derivatives_against_symbolic_physical_stress():
    x, y = sp.symbols("x y", real=True)
    radius = sp.sqrt(x * x + y * y)
    radial = sp.Matrix([x, y]) / radius
    tangent = sp.Matrix([-y, x]) / radius
    velocity = sp.Matrix([x * x + 2 * x * y - y * y + 3 * x, -x * x - 2 * x * y - y * y - 3 * y])
    regular_pressure = 1 + 2 * x - sp.Rational(7, 10) * y
    viscosity = 1.3
    for offset in (1.1, 2.02):
        vertices = np.array([[[offset, 0.02], [offset + 0.12, 0.02], [offset + 0.04, 0.16]]])
        nodes = np.concatenate(
            (vertices, (vertices[:, [0, 1, 2]] + vertices[:, [1, 2, 0]]) / 2), axis=1
        )
        v = np.stack(
            [
                sp.lambdify((x, y), value, "numpy")(nodes[..., 0], nodes[..., 1])
                for value in velocity
            ],
            axis=-1,
        )
        p = sp.lambdify((x, y), regular_pressure, "numpy")(vertices[..., 0], vertices[..., 1])
        bary = np.array([[0.2, 0.3, 0.5], [0.4, 0.2, 0.4]])
        points = np.einsum("qi,eid->eqd", bary, vertices)
        result = numerical_trace(vertices, v, p, points, viscosity, 2.0, 2.22, 2, 0.8)
        jump = (
            -0.8 * (2.22 - radius) / 0.22 * (x * x - y * y) / (x * x + y * y) if offset > 2 else 0
        )
        gradient = velocity.jacobian([x, y])
        stress = viscosity * (gradient + gradient.T) - (regular_pressure + jump) * sp.eye(2)
        rr = (radial.T * stress * radial)[0]
        rt = (radial.T * stress * tangent)[0]
        expressions = dict(
            radial_velocity_derivative=(radial.T * gradient * radial)[0],
            rr=rr,
            tt=(tangent.T * stress * tangent)[0],
            rt=rt,
            rt_derivative=radial[0] * sp.diff(rt, x) + radial[1] * sp.diff(rt, y),
            rr_derivative=radial[0] * sp.diff(rr, x) + radial[1] * sp.diff(rr, y),
        )
        for key, expression in expressions.items():
            expected = sp.lambdify((x, y), expression, "numpy")(points[..., 0], points[..., 1])
            np.testing.assert_allclose(result[key], expected, rtol=1e-11, atol=1e-11, err_msg=key)


@pytest.mark.tier_b
def test_rigid_rotation_has_no_stress_or_radial_velocity_transfer():
    vertices = np.array([[[1.1, 0.1], [1.2, 0.1], [1.1, 0.2]]])
    nodes = np.concatenate(
        (vertices, (vertices[:, [0, 1, 2]] + vertices[:, [1, 2, 0]]) / 2), axis=1
    )
    velocity = 2.3 * np.stack((-nodes[..., 1], nodes[..., 0]), axis=-1)
    points = np.einsum("qi,eid->eqd", np.array([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1]]), vertices)
    result = numerical_trace(vertices, velocity, np.zeros((1, 3)), points, 1.0, 2.0, 2.22, 2, 0.0)
    for value in result.values():
        np.testing.assert_allclose(value, 0, atol=1e-11)


@pytest.mark.tier_b
def test_smooth_pressure_needs_no_sheet_radius_or_jump():
    vertices = np.array([[[1.1, 0.1], [1.2, 0.1], [1.1, 0.2]]])
    points = vertices.mean(axis=1)[:, None, :]
    trace = numerical_trace(
        vertices,
        np.zeros((1, 6, 2)),
        np.ones((1, 3)) * 2.3,
        points,
        1.0,
        2.22,
        2.22,
        2,
        0.0,
    )
    np.testing.assert_allclose(trace["rr"], -2.3)
    np.testing.assert_allclose(trace["tt"], -2.3)
    np.testing.assert_allclose(trace["rr_derivative"], 0, atol=1e-13)


@pytest.mark.tier_b
def test_torque_projection_preserves_normal_nodal_loads_on_irregular_trace():
    angles = np.linspace(0, 2 * np.pi, 41)[:-1]
    angles += 0.025 * np.sin(3 * angles)
    vertices = 1.7 * np.column_stack((np.cos(angles), np.sin(angles)))
    edges = np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1)
    edge_nodes = np.stack((edges[:, 0], edges[:, 1], edges.mean(axis=1)), axis=1)
    nodes, index = np.unique(edge_nodes.reshape(-1, 2), axis=0, return_inverse=True)
    index = index.reshape(-1, 3)
    rotation = np.column_stack((-nodes[:, 1], nodes[:, 0]))
    density = 0.4 * rotation + nodes * np.sin(2 * np.arctan2(nodes[:, 1], nodes[:, 0]))[:, None]
    balanced, checks = zero_torque_density(edges, nodes, density)
    loads = []
    for field in (density, balanced):
        value = np.zeros_like(nodes)
        for edge, rows in zip(edges, index):
            mass = (
                np.linalg.norm(edge[1] - edge[0])
                / 30
                * np.array([[4, -1, 2], [-1, 4, 2], [2, 2, 16]])
            )
            np.add.at(value, rows, mass @ field[rows])
        loads.append(value)
    np.testing.assert_allclose(
        np.sum(loads[0] * rotation), checks["raw_boundary_torque"], atol=1e-13
    )
    np.testing.assert_allclose(np.sum(loads[1] * rotation), 0, atol=1e-13)
    normal = nodes / np.linalg.norm(nodes, axis=1)[:, None]
    np.testing.assert_allclose(np.sum((loads[1] - loads[0]) * normal, axis=1), 0, atol=1e-13)


@pytest.mark.tier_b
def test_spatial_reaction_transfer_preserves_cbf_residual_and_removes_pressure_gauge():
    radius, count = 1.7, 48
    theta = np.arange(count) * 2 * np.pi / count
    theta += 0.015 * np.sin(3 * theta)
    vertices = radius * np.column_stack((np.cos(theta), np.sin(theta)))
    edges = np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1)
    edge_nodes = np.stack((edges[:, 0], edges[:, 1], edges.mean(axis=1)), axis=1)
    nodes, index = np.unique(edge_nodes.reshape(-1, 2), axis=0, return_inverse=True)
    index = index.reshape(-1, 3)
    z, w = np.polynomial.legendre.leggauss(16)
    t, w = (z + 1) / 2, w / 2
    shape = np.array([(1 - t) * (1 - 2 * t), t * (2 * t - 1), 4 * t * (1 - t)]).T
    points = np.einsum("qi,eid->eqd", shape, edge_nodes)
    r = np.linalg.norm(points, axis=-1)
    harmonic = (points[..., 0] ** 2 - points[..., 1] ** 2) / r**2
    normal = edges.mean(axis=1) / np.linalg.norm(edges.mean(axis=1), axis=1)[:, None]
    length = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    polygon_weights = length[:, None] * w
    circle_weights = (
        polygon_weights
        * radius
        / r
        * np.abs(np.sum(normal[:, None, :] * points / r[..., None], axis=-1))
    )
    masses = []
    for weights in (polygon_weights, circle_weights):
        mass = np.zeros((len(nodes), len(nodes)))
        for e, rows in enumerate(index):
            mass[np.ix_(rows, rows)] += np.einsum("q,qi,qj->ij", weights[e], shape, shape)
        masses.append(mass)
    for sign in (-1, 1):
        for gauge in (0, 2.3):
            pressure = r * harmonic + gauge
            normal_nodes = sign * edge_nodes / np.linalg.norm(edge_nodes, axis=-1)[..., None]
            traction = -pressure[..., None] * sign * normal[:, None, :]
            loads = np.zeros(len(nodes))
            for e, rows in enumerate(index):
                load = -np.einsum(
                    "q,qi,qd,id->i", polygon_weights[e], shape, traction[e], normal_nodes[e]
                )
                np.add.at(loads, rows, load)
            raw = np.linalg.solve(masses[0], loads)
            trace = dict(rr=-pressure, tt=-pressure, rt=np.zeros_like(r), rr_derivative=-harmonic)
            _, corrected, checks = circular_reaction_field(edges, trace, raw, radius, sign, 16)
            expected = np.zeros(len(nodes))
            for e, rows in enumerate(index):
                np.add.at(
                    expected,
                    rows,
                    np.einsum("q,qi,q->i", circle_weights[e], shape, radius * harmonic[e]),
                )
            expected -= expected.sum() / masses[1].sum() * (masses[1] @ np.ones(len(nodes)))
            np.testing.assert_allclose(masses[1] @ corrected, expected, atol=1e-13)
            # A numerical CBF residual must survive the geometric correction.
            residual = np.cos(5 * np.arctan2(nodes[:, 1], nodes[:, 0]))
            residual -= (masses[0] @ residual).sum() / masses[0].sum()
            _, changed, _ = circular_reaction_field(edges, trace, raw + residual, radius, sign, 16)
            np.testing.assert_allclose(
                masses[1] @ (changed - corrected), masses[0] @ residual, atol=1e-13
            )
            assert checks["maximum_mass_moment_error"] < 1e-13


@pytest.mark.tier_b
def test_wall_update_sequence_accepts_zero_increment_and_retains_failure():
    class Wall:
        count = 0

        def update(self):
            self.count += 1
            return {"update": self.count}

    class Stokes:
        count = 0
        u = SimpleNamespace(array=np.zeros((0, 1, 2)))
        _rotated_pressure_gauge_residual = 0.0

        def solve(self):
            self.count += 1
            self._rotated_freeslip_info = dict(
                converged=True, ksp_reason=2 if self.count == 1 else 0,
                nonlinear_iterations=1 if self.count == 1 else 0,
                ksp_its=[3] if self.count == 1 else [], rnorm=1e-13, rnorm0=1.0)

    wall, stokes = Wall(), Stokes()
    history = _solve_with_wall_updates(stokes, {"Upper": wall}, 4)
    assert stokes.count == 5 and wall.count == 4
    assert len(history) == 5 and history[-1]["boundary_updates"] == {}
    assert history[-1]["ksp_reason"] == 0
    assert history[-1]["velocity_change_maximum"] == 0
    failed_info = dict(converged=False, ksp_reason=-3, nonlinear_iterations=1,
                       ksp_its=[9], rnorm=0.02, rnorm0=1.0)
    failed = SimpleNamespace(solve=lambda: None, u=stokes.u,
                             _rotated_freeslip_info=failed_info,
                             _rotated_pressure_gauge_residual=0.0)
    with pytest.raises(RuntimeError, match="iteration 0 failed"):
        _solve_with_wall_updates(failed, {}, 4)
    assert len(failed.cylindrical_wall_history) == 1
    assert failed.cylindrical_wall_history[0]["ksp_reason"] == -3
