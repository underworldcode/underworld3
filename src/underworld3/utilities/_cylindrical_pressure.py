"""Private circular source extraction for affine two-dimensional Stokes meshes.

The caller selects this mathematical correction explicitly. No reference Stokes
solution enters the source. Owned-cell checks are reduced once; force values
also populate local ghost cells. Public entry: Stokes.cylindrical_pressure_lifting.

Pressure decomposition is motivated by Chang and Kwak (2009), "Extraction method
for Stokes Flow with jumps in the pressure", arXiv:0911.4769,
https://arxiv.org/abs/0911.4769, Section 2, equations (4)-(6). The prescribed
radial lift, exact-circle cap integration, DG P2 moment representation and
outer-chord natural datum are this UW3 implementation. It does not implement
the paper's Section 3 harmonic extension or Section 4 CR discretisation.

Only nondimensional mesh coordinates, fields and input scalars are supported.
Active reference-unit scaling is outside this helper's supported contract.
"""

from typing import NamedTuple

import numpy as np
import sympy as sp
import underworld3 as uw
from mpi4py import MPI
from petsc4py import PETSc


class _PressureLift(NamedTuple):
    pressure_jump: object
    regular_force: object
    natural_datum: object
    pressure_jump_integral: float
    checks: dict


def _p2_shapes(barycentric: np.ndarray) -> np.ndarray:
    """P2 values in vertex order 0,1,2 followed by edges 01,12,20."""
    barycentric = np.asarray(barycentric)
    return np.stack(
        [barycentric[..., i] * (2 * barycentric[..., i] - 1) for i in range(3)]
        + [4 * barycentric[..., i] * barycentric[..., j] for i, j in ((0, 1), (1, 2), (2, 0))],
        axis=-1,
    )

def _profile_values(radius: np.ndarray, interface_radius: float, width: float):
    """Linear radial lifting and derivative, extended below R for cap integration."""
    s = (np.asarray(radius) - interface_radius) / width
    return 1 - s, np.full_like(s, -1 / width)

def _extended_regular_force(points, interface_radius, width, wavenumber, amplitude):
    """Smooth outer branch of A*grad(b*cos(n*theta)), including its cap extension."""
    radius = np.linalg.norm(points, axis=-1)
    theta = np.arctan2(points[..., 1], points[..., 0])
    radial = points / radius[..., None]
    tangent = np.stack((-radial[..., 1], radial[..., 0]), axis=-1)
    value, derivative = _profile_values(radius, interface_radius, width)
    return amplitude * (
        (derivative * np.cos(wavenumber * theta))[..., None] * radial
        - (wavenumber * value * np.sin(wavenumber * theta) / radius)[..., None] * tangent
    )

def _load_moments(vertices, interface_radius, width, wavenumber, amplitude, quadrature_points):
    """P2 force moments on outer fitted triangles, excluding each circular cap.

    Interior quadrature integrates the smooth outer branch on each triangle.
    Polar cap quadrature removes the region r<R exactly in geometry. No
    analytical Stokes velocity, pressure or stress enters this construction.
    This polar cap subtraction and DG P2 moment representation are UW3
    constructions, not the interface approximation in Chang and Kwak (2009).
    """
    edges = np.stack((vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=-1)
    inverse = np.linalg.inv(edges)
    area = np.abs(np.linalg.det(edges)) / 2
    moments = np.zeros((len(vertices), 6, 2))
    lift_integral = np.zeros(len(vertices))
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    nodes, weights = (nodes + 1) / 2, weights / 2
    for x, wx in zip(nodes, weights):
        for y, wy in zip(nodes, weights):
            bary = np.array([1 - x, x * (1 - y), x * y])
            point = np.einsum("i,nij->nj", bary, vertices)
            force = _extended_regular_force(point, interface_radius, width, wavenumber, amplitude)
            moments += (
                (area * 2 * x * wx * wy)[:, None, None]
                * _p2_shapes(bary)[None, :, None]
                * force[:, None, :]
            )
            radius = np.linalg.norm(point, axis=1)
            lift_integral -= (
                area
                * 2
                * x
                * wx
                * wy
                * amplitude
                * (1 - (radius - interface_radius) / width)
                * np.cos(wavenumber * np.arctan2(point[:, 1], point[:, 0]))
            )

    cap_area = np.zeros(len(vertices))
    chord_count = np.zeros(len(vertices), dtype=int)
    radii = np.linalg.norm(vertices, axis=-1)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        selected = np.flatnonzero(
            np.isclose(radii[:, first], interface_radius, rtol=0, atol=1e-10)
            & np.isclose(radii[:, second], interface_radius, rtol=0, atol=1e-10)
        )
        if not len(selected):
            continue
        left, right = vertices[selected, first], vertices[selected, second]
        normal = left + right
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
        distance = np.sum(left * normal, axis=1)
        half_angle = np.arctan2(np.linalg.norm(right - left, axis=1) / 2, distance)
        measured_area = np.zeros(len(selected))
        for t, wt in zip(nodes, weights):
            angle = (2 * t - 1) * half_angle
            direction = np.cos(angle)[:, None] * normal + np.sin(angle)[:, None] * tangent
            lower = distance / np.cos(angle)
            thickness = interface_radius - lower
            for s, ws in zip(nodes, weights):
                radius = lower + s * thickness
                point = radius[:, None] * direction
                xi = np.einsum("nij,nj->ni", inverse[selected], point - vertices[selected, 0])
                bary = np.column_stack((1 - xi.sum(axis=1), xi))
                if bary.min() < -1e-9:
                    raise ValueError("Circular cap leaves its adjacent affine triangle.")
                weight = radius * thickness * ws * 2 * half_angle * wt
                force = _extended_regular_force(
                    point, interface_radius, width, wavenumber, amplitude
                )
                moments[selected] -= (
                    weight[:, None, None] * _p2_shapes(bary)[..., None] * force[:, None, :]
                )
                lift_integral[selected] += (
                    weight
                    * amplitude
                    * (1 - (radius - interface_radius) / width)
                    * np.cos(wavenumber * np.arctan2(point[:, 1], point[:, 0]))
                )
                measured_area += weight
        expected_area = interface_radius**2 * half_angle - distance**2 * np.tan(half_angle)
        if not np.allclose(measured_area, expected_area, rtol=1e-8, atol=1e-13):
            raise ArithmeticError("Polar quadrature did not reproduce circular-cap areas.")
        cap_area[selected] += measured_area
        chord_count[selected] += 1
    return moments, cap_area, chord_count, lift_integral

def _create_circular_pressure_lift(
    mesh, interface_radius, outer_radius, wavenumber, amplitude, quadrature_points
):
    """Return exact-circle p_jump, a moment-equivalent DG P2 force, and global checks.

    A linear radial profile avoids introducing a narrow transition layer.
    Local cell connectivity includes ghosts; each DG cell is populated locally,
    then owned-cell diagnostics are summed once across MPI ranks.

    References
    ----------
    K. S. Chang and D. Y. Kwak (2009), "Extraction method for Stokes
    Flow with jumps in the pressure", arXiv:0911.4769,
    https://arxiv.org/abs/0911.4769, Section 2, equations (4)-(6).
    The regular-plus-singular pressure split motivates this construction;
    its radial profile, cut loads and boundary datum are UW3 adaptations.
    """
    if mesh.dim != 2 or mesh.cdim != 2:
        raise ValueError("Circular pressure extraction requires a two-dimensional affine mesh.")
    if not 0 < interface_radius < outer_radius:
        raise ValueError("Require 0 < interface_radius < outer_radius.")
    if quadrature_points < 2:
        raise ValueError("Circular-cut quadrature requires at least two points.")
    width = outer_radius - interface_radius
    force_field = uw.discretisation.MeshVariable(
        "CircleRegularForce", mesh, 2, degree=2, continuous=False
    )
    vertex_rows = mesh._cell_node_indices(1, True).reshape(-1, 3)
    vertices = np.asarray(mesh._get_coords_for_basis(1, True))[vertex_rows]
    outer = np.linalg.norm(vertices.mean(axis=1), axis=1) > interface_radius
    moments, cap_area, chord_count, lift_integral = _load_moments(
        vertices[outer], interface_radius, width, wavenumber, amplitude, quadrature_points
    )
    edges = np.stack((vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=-1)
    area = np.abs(np.linalg.det(edges)) / 2

    # Match native P2 weak loads, rather than interpolate pointwise source values.
    fe = PETSc.FE().createDefault(2, 1, True, mesh.qdegree, "circle_force_mass_")
    points, weights = fe.getQuadrature().getData()
    xy = (np.asarray(points).reshape(-1, 2) + 1) / 2
    bary = np.column_stack((1 - xy.sum(axis=1), xy))
    shapes = _p2_shapes(bary)
    mass = np.einsum("qi,qj,q->ij", shapes, shapes, np.asarray(weights).reshape(-1) / 2)
    fe.destroy()
    if np.linalg.matrix_rank(mass) != 6:
        raise ValueError("Quadrature must integrate the full P2 mass matrix (qdegree >= 2).")
    nodal = np.zeros((len(vertices), 6, 2))
    nodal[outer] = np.einsum("ij,njk->nik", np.linalg.inv(mass), moments / area[outer, None, None])
    reproduced = area[outer, None, None] * np.einsum("ij,njk->nik", mass, nodal[outer])
    moment_error = float(np.max(np.abs(reproduced - moments))) if len(moments) else 0.0
    if moment_error > 1e-11:
        raise ArithmeticError(f"DG P2 load-moment representation failed: {moment_error}")

    rows = mesh._cell_node_indices(2, False).reshape(-1, 6)
    coordinates = np.asarray(force_field.coords)[rows]
    xi = np.einsum("nij,nkj->nki", np.linalg.inv(edges), coordinates - vertices[:, None, 0])
    bary = np.concatenate((1 - xi.sum(axis=-1, keepdims=True), xi), axis=-1)
    force_field.array[rows, 0, :] = np.einsum("nki,nij->nkj", _p2_shapes(bary), nodal)
    owned = mesh._get_owned_cells_mask()[outer]
    expected_force = uw.mpi.comm.allreduce(moments[owned].sum(axis=(0, 1)), op=MPI.SUM)
    actual_force = np.array(
        [uw.maths.Integral(mesh, force_field.sym[i]).evaluate() for i in range(2)]
    )
    force_error = float(np.max(np.abs(actual_force - expected_force)))
    if force_error > 1e-10:
        raise ArithmeticError(f"Native DG force integral differs from the cut load: {force_error}")
    checks = dict(
        width=width,
        quadrature_points=quadrature_points,
        circular_segment_area=float(uw.mpi.comm.allreduce(cap_area[owned].sum(), op=MPI.SUM)),
        internal_chord_count=int(uw.mpi.comm.allreduce(chord_count[owned].sum(), op=MPI.SUM)),
        maximum_load_moment_error=float(uw.mpi.comm.allreduce(moment_error, op=MPI.MAX)),
        pressure_lift_integral=float(uw.mpi.comm.allreduce(lift_integral[owned].sum(), op=MPI.SUM)),
        native_force_integral_error=force_error,
    )
    radius, theta = mesh.CoordinateSystem.xR
    interface_radius, width, amplitude = [sp.nsimplify(value, rational=True) for value in (interface_radius, width, amplitude)]
    s = (radius - interface_radius) / width
    lift = sp.Piecewise(
        (-amplitude * (1 - s) * sp.cos(wavenumber * theta), radius > interface_radius),
        (0, True),
    )
    return _PressureLift(lift, force_field, -lift * mesh.Gamma, checks["pressure_lift_integral"], checks)


def _pressure_jump_values(points, interface_radius, outer_radius, wavenumber, amplitude):
    """Prescribed pressure extraction and its smooth one-sided Cartesian gradient."""
    radius = np.linalg.norm(points, axis=-1)
    above = radius > interface_radius
    width = outer_radius - interface_radius
    profile, _ = _profile_values(radius, interface_radius, width)
    pressure = -amplitude * above * profile * np.cos(
        wavenumber * np.arctan2(points[..., 1], points[..., 0]))
    gradient = -(above[..., None] * _extended_regular_force(
        points, interface_radius, width, wavenumber, amplitude))
    return pressure, gradient
