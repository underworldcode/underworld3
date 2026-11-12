"""Free-slip circular-wall transfer: exact polynomial, load and solver contracts."""

import numpy as np
import pytest
import sympy as sp
import underworld3 as uw

from underworld3.utilities._cylindrical_wall import (
    _FreeSlipWall,
    _numerical_trace as numerical_trace,
    _tangent_load_density,
    _zero_torque_density as zero_torque_density,
)

pytestmark = pytest.mark.level_1


def _stokes(mesh):
    velocity = uw.discretisation.MeshVariable("WallVelocity", mesh, 2, degree=2)
    pressure = uw.discretisation.MeshVariable("WallPressure", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = sp.Integer(1)
    stokes.saddle_preconditioner = sp.Integer(1)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-9
    stokes.petsc_options["snes_type"] = "ksponly"
    stokes.petsc_options["ksp_atol"] = 1e-12
    return stokes


@pytest.mark.tier_b
def test_general_dg_lift_has_exact_cell_side_boundary_values():
    """DG P1 lifting is evaluated by the boundary-adjacent cell, without averaging."""
    mesh = uw.meshing.Annulus(radiusInner=1.22, radiusOuter=2.22,
                              cellSize=1/16, qdegree=3, degree=1)
    stokes = _stokes(mesh)
    lift = uw.discretisation.MeshVariable("WallLiftDG", mesh, 1, degree=1, continuous=False)
    rows = mesh._cell_node_indices(1, False).reshape(-1, 3)
    # Distinct constants per cell expose accidental continuous interpolation.
    lift.array[rows, 0, 0] = np.arange(len(rows))[:, None] + 2
    wall = _FreeSlipWall(stokes, "Upper", 2.22, 1, 1, lift, 6)
    _, actual = wall.pressure_values()
    local = np.repeat((wall.cells+2)[:, None], 6, axis=1)
    expected = np.concatenate(uw.mpi.comm.allgather(local))
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-12)
    # The expression returned by pressure_split identifies that same DG field.
    expression_wall = _FreeSlipWall(stokes, "Lower", 1.22, -1, 1, lift.sym[0], 6)
    assert expression_wall.pressure_jump.sym[0] == lift.sym[0]


@pytest.mark.tier_b
def test_tangent_mass_projection_preserves_its_exact_nodal_moments():
    theta = np.linspace(0, 2*np.pi, 25)[:-1]
    vertices = 1.7*np.column_stack((np.cos(theta), np.sin(theta)))
    edges = np.stack((vertices, np.roll(vertices, -1, axis=0)), axis=1)
    gauss, weight = np.polynomial.legendre.leggauss(6)
    t = (gauss+1)/2
    points = (1-t)[None, :, None]*edges[:, None, 0]+t[None, :, None]*edges[:, None, 1]
    traction = np.stack((points[..., 0]**2, -points[..., 1]), axis=-1)
    nodes, density, checks = _tangent_load_density(edges, traction, 1, 6)
    assert checks["maximum_normal_nodal_load"] < 1e-13
    assert checks["maximum_mass_moment_error"] < 1e-13
    shape = np.array([(1-t)*(1-2*t), t*(2*t-1), 4*t*(1-t)]).T
    lookup = {tuple(node): row for row, node in enumerate(nodes)}
    mass_load = np.zeros_like(nodes)
    for edge in edges:
        index = [lookup[tuple(node)] for node in (edge[0], edge[1], edge.mean(axis=0))]
        mass = np.linalg.norm(edge[1]-edge[0])/30*np.array([[4,-1,2],[-1,4,2],[2,2,16]])
        np.add.at(mass_load, index, mass@density[index])
    raw = np.einsum("eq,qi,eqd->eid", np.linalg.norm(edges[:,1]-edges[:,0], axis=1)[:,None]*weight/2, shape, traction)
    exact = np.zeros_like(nodes)
    for edge, values in zip(edges, raw):
        index = [lookup[tuple(node)] for node in (edge[0], edge[1], edge.mean(axis=0))]
        np.add.at(exact, index, values)
    normal = nodes/np.linalg.norm(nodes, axis=1)[:,None]
    exact -= np.sum(exact*normal, axis=1)[:,None]*normal
    np.testing.assert_allclose(mass_load, exact, atol=1e-13)


@pytest.mark.tier_b
@pytest.mark.parametrize("slip", ["free", "zero"])
def test_wall_iteration_preserves_exact_quiescent_solution(slip):
    """Zero force has u=p=0 on both physical circles at every lagged update."""
    mesh = uw.meshing.Annulus(radiusInner=1.22, radiusOuter=2.22,
                              cellSize=1/16, qdegree=3, degree=1)
    stokes = _stokes(mesh)
    stokes.bodyforce = sp.Matrix([0, 0])
    if slip == "free":
        stokes.petsc_use_nullspace = True
        add_wall = stokes.add_cylindrical_wall_transfer
        options = dict(viscosity=1, pressure_jump=sp.S.Zero)
        solve = stokes.solve_cylindrical_wall_updates
    else:
        add_wall = stokes.add_cylindrical_zero_slip_transfer
        options = {}
        solve = stokes.solve_cylindrical_zero_slip_updates
    walls = {name: add_wall(boundary=name, radius=radius, sign=sign, quadrature=6, **options)
             for name, radius, sign in (("Lower", 1.22, -1), ("Upper", 2.22, 1))}
    history = solve(walls=walls, updates=2)
    assert len(history) == 3
    for step in history:
        assert step["converged"]
        assert step["velocity_change_maximum"] < 1e-12
        for residuals in step["boundary_velocity"].values():
            key = "true_circle_normal_velocity_rms" if slip == "free" else "true_circle_velocity_rms"
            assert residuals[key] < 1e-12
    assert np.max(np.abs(stokes.u.array), initial=0) < 1e-12


@pytest.mark.tier_b
def test_general_pressure_split_and_zero_slip_preserve_hydrostatic_balance():
    """A constant prescribed interface jump supports zero flow exactly."""
    mesh = uw.meshing.AnnulusInternalBoundary(
        radiusInner=1.22, radiusInternal=2, radiusOuter=2.22,
        cellSize_Inner=1/16, cellSize_Internal=1/16, cellSize_Outer=1/16,
        qdegree=3, degree=1)
    stokes = _stokes(mesh)
    split = stokes.pressure_split(jumps={"Internal": sp.Integer(1)},
        normals={"Internal": mesh.CoordinateSystem.unit_e_0})
    stokes.bodyforce = split.regular_force.sym
    walls = {name: stokes.add_cylindrical_zero_slip_transfer(
        boundary=name, radius=radius, sign=sign, quadrature=6)
        for name, radius, sign in (("Lower", 1.22, -1), ("Upper", 2.22, 1))}
    stokes.solve_cylindrical_zero_slip_updates(walls=walls, updates=2)
    velocity_l2 = np.sqrt(abs(uw.maths.Integral(mesh, stokes.u.sym.dot(stokes.u.sym)).evaluate()))
    assert velocity_l2 < 1e-8


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
        jump = (
            -0.8 * (2.22 - radius) / 0.22 * (x * x - y * y) / (x * x + y * y) if offset > 2 else 0
        )
        jump_values = sp.lambdify((x, y), jump, "numpy")(points[..., 0], points[..., 1])
        result = numerical_trace(vertices, v, p, points, viscosity, jump_values)
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
    result = numerical_trace(vertices, velocity, np.zeros((1, 3)), points, 1.0, np.zeros(points.shape[:2]))
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
        np.zeros(points.shape[:2]),
    )
    np.testing.assert_allclose(trace["rr"], -2.3)
    np.testing.assert_allclose(trace["tt"], -2.3)
    np.testing.assert_allclose(trace["rt_derivative"], 0, atol=1e-13)


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
