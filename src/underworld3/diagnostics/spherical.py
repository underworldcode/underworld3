"""Independent P2/P1 error integration for affine spherical-shell Stokes fields.

Report computational-polyhedron norms and physical inner-cell restriction
norms, with the backend's explicit outer continuation. Inner-wall quadrature
uses each cut cell's polynomial and is partitioned by the original multiplier
facet cones. Historical signed inner-cap norms are optional and retain their
distinct name. Pressure alignment is separate for every evaluated domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np
from mpi4py import MPI
import sympy as sp
import underworld3 as uw
from underworld3.utilities._spherical_gap import _spherical_facet_quadrature
from underworld3.utilities._field_layout import _layout
from underworld3.utilities._spherical_cut_cell import (
    _inner_cell_relation, _spherical_cut_cell_quadrature, _spherical_cut_cell_surface,
)
from underworld3.utilities._spherical_stokes_gap import _canonical_rows, _wall_faces
from underworld3.utilities._spherical_wall_trace import _inner_wall_cell_facets
from underworld3.utilities._tetrahedral_gap import _tetrahedral_p2_basis, _tetrahedron_quadrature


class ExactReference(Protocol):
    def evaluate(self, field: str, coords: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class QuadratureOrders:
    volume: int
    surface: int
    radial: int


def _physical_volume_batches(vertices: np.ndarray, owned: np.ndarray, radius: float,
                             orders: QuadratureOrders, batch_size: int):
    """Integrate each owned tetrahedron's actual restriction outside the ball."""
    relation = _inner_cell_relation(vertices[owned], radius)
    outside = owned[relation == 1]
    for start in range(0, len(outside), batch_size):
        cells = outside[start:start+batch_size]
        points, weights = _tetrahedron_quadrature(vertices[cells], orders.volume)
        yield cells, points, weights
    for cell in owned[relation == 0]:
        rule = _spherical_cut_cell_quadrature(vertices[cell], radius,
                                              max(12, orders.surface), orders.radial)
        yield np.array([cell]), rule['volume_points'][None], rule['volume_weights'][None]


def _physical_inner_surface_rules(mesh, vertices, owned, wall, radius, surface_order):
    """Collectively prepare inner cell/facet intersections; later batches are local."""
    cells = owned[_inner_cell_relation(vertices[owned], radius) == 0]
    candidates = _inner_wall_cell_facets(mesh, wall, vertices[cells])
    rules = []
    for cell, (facets, lids) in zip(cells, candidates, strict=True):
        points, weights, ids, values = _spherical_cut_cell_surface(
            vertices[cell], radius, max(12, surface_order), facets, lids)
        rules.append((cell, points, weights, ids, values))
    missing = mesh.dm.comm.tompi4py().allreduce(sum(len(rule[1]) == 0 for rule in rules), op=MPI.SUM)
    if missing:
        raise ArithmeticError(f"Physical inner-cut cells have no diagnostic wall quadrature: {missing}")
    return rules


def _jump_values(mesh: uw.discretisation.Mesh, expression: sp.Expr,
                 points: np.ndarray) -> np.ndarray:
    """Evaluate scalar coordinate lifting in Cartesian mesh.N coordinates."""
    if expression.is_number:
        return np.full(points.shape[:-1], float(expression))
    values = uw.function.evaluate(expression, np.ascontiguousarray(points.reshape(-1, 3)),
                                  coord_sys=mesh.N)
    return np.asarray(values, dtype=float).reshape(points.shape[:-1])


def _norm_result(error_squared: float, reference_squared: float, measure: float,
                 cancellation_scale: float, reference_zero_scale: float,
                 reference_is_prescribed_zero: bool = False) -> dict:
    """Retain absolute RMS when a relative error has a zero reference norm."""
    if not np.isfinite([error_squared, reference_squared, measure, cancellation_scale,
                        reference_zero_scale]).all() or measure <= 0:
        raise ArithmeticError("Error integration requires finite norms and positive measure")
    tolerance = 128 * np.finfo(float).eps * cancellation_scale
    if error_squared < -tolerance or reference_squared < -tolerance:
        raise ArithmeticError("Signed-gap quadrature produced a negative squared norm; increase quadrature orders")
    error_squared, reference_squared = max(0., error_squared), max(0., reference_squared)
    zero_reference = (reference_is_prescribed_zero
                      or reference_squared <= (64*np.finfo(float).eps)**2 * reference_zero_scale)
    return dict(relative_error_fraction=None if zero_reference else float(np.sqrt(error_squared/reference_squared)),
                absolute_rms_error=float(np.sqrt(error_squared/measure)),
                reference_rms=float(np.sqrt(reference_squared/measure)),
                relative_error_defined=not zero_reference,
                reference_is_prescribed_zero=reference_is_prescribed_zero)


def spherical_stokes_errors(mesh: uw.discretisation.Mesh,
                             velocity: uw.discretisation.MeshVariable,
                             reduced_pressure: uw.discretisation.MeshVariable,
                             coordinate_pressure_jump: sp.Expr,
                             exact: ExactReference, radius_inner: float, radius_outer: float,
                             orders: QuadratureOrders, batch_size: int,
                             inner_boundary: str, outer_boundary: str,
                             *, viscosity: float,
                             reference_zero_wall_velocity_components: tuple[str, ...] = (),
                             include_legacy_signed_gap: bool = False) -> dict:
    """Return global errors for owned-cell P2 velocity and continuous P1 q.

    Total pressure is q + coordinate_pressure_jump. The jump must contain only
    mesh.N coordinates (zero or the true spherical Heaviside lift). Volume
    offsets are means of numerical-minus-reference total pressure on each
    domain; wall pressure uses the physical inner-cut volume offset. Reference
    pressure norms retain the analytical reference's stated gauge. Stress
    errors use the full Frobenius norm of constant-viscosity Cauchy
    stress, with the corresponding domain pressure offset applied. Reference
    stress is evaluated in the analytical reference's unchanged pressure gauge.
    All relative values are fractions. Relative wall-velocity error is undefined (None) when
    its analytical component is zero to floating-point precision, scaled by the
    larger of the full analytical wall norm and physical-volume reference
    velocity RMS times sqrt(wall area). Three unfiltered rotation moments are saved.
    The caller may declare mathematically zero wall-velocity components
    ('normal', 'tangential'). Their relative error is undefined even when
    floating analytical evaluation leaves a small residual. Raw reference
    RMS and absolute errors are retained without modifying either field.
    The physical_inner_cut_volume key uses actual inner-cell restriction and
    explicit outer continuation. Request include_legacy_signed_gap=True to
    additionally retain the old continued-cap norm as physical_signed_gap_volume.
    The caller is responsible for syncing fields and selecting quadrature orders
    sufficient to resolve discontinuous forcing/lifting and analytical fields.
    """
    if not (0 < radius_inner < radius_outer) or not np.isfinite([radius_inner, radius_outer]).all():
        raise ValueError("Require finite ordered positive spherical radii")
    if not np.isfinite(viscosity) or viscosity <= 0:
        raise ValueError("Require finite positive constant viscosity")
    if not isinstance(include_legacy_signed_gap, bool):
        raise ValueError("The historical signed-gap selection must be a boolean")
    if (not isinstance(reference_zero_wall_velocity_components, tuple)
            or any(name not in ('normal', 'tangential') for name in reference_zero_wall_velocity_components)
            or len(set(reference_zero_wall_velocity_components)) != len(reference_zero_wall_velocity_components)):
        raise ValueError("Declare zero reference components as a tuple drawn from normal/tangential")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1
           for n in (orders.volume, orders.surface, orders.radial, batch_size)):
        raise ValueError("Quadrature orders and batch size must be positive integers")
    if (velocity.mesh is not mesh or reduced_pressure.mesh is not mesh
            or velocity.degree != 2 or velocity.num_components != 3
            or reduced_pressure.degree != 1 or reduced_pressure.num_components != 1
            or not velocity.continuous or not reduced_pressure.continuous):
        raise ValueError("Require continuous P2 vector velocity and P1 scalar pressure on this mesh")
    jump = uw.function.expressions._unwrap_for_compilation(
        sp.sympify(coordinate_pressure_jump), keep_constants=False, return_self=False)
    if any(hasattr(type(atom), "meshvar") for atom in jump.atoms(sp.Function)):
        raise ValueError("Pressure jump must be scalar and coordinate-only")
    comm = mesh.dm.comm.tompi4py()
    vertex_rows = mesh._cell_node_indices(1, True)
    vertices = np.asarray(mesh._get_coords_for_basis(1, True))[vertex_rows]
    urows = _canonical_rows(mesh, velocity, vertices)
    prows = _canonical_rows(mesh, reduced_pressure, vertices)
    uvalues = np.asarray(velocity.array)[urows, 0, :]
    qvalues = np.asarray(reduced_pressure.array)[prows, 0, 0]
    owned = np.flatnonzero(mesh._get_owned_cells_mask())
    walls = [(inner_boundary, radius_inner, -1.), (outer_boundary, radius_outer, 1.)]
    wall_data = [(name, radius, sign, _wall_faces(mesh, vertices, urows, owned, name, radius))
                 for name, radius, sign in walls]
    velocity_map, _ = _layout(velocity)
    inner_wall = wall_data[0][3]
    inner_wall.update(radius=radius_inner,
                      lid=(velocity_map.reshape(-1, 3)[:, 0][inner_wall['rows']]//3))
    inner_rules = _physical_inner_surface_rules(mesh, vertices, owned, inner_wall,
                                               radius_inner, orders.surface)
    relation = _inner_cell_relation(vertices[owned], radius_inner)
    active_domains = (0, 1, 2) if include_legacy_signed_gap else (0, 2)

    def quadrature_batches():
        for selected, domains in ((owned[relation == 1], active_domains),
                                   (owned[relation != 1], (0, 1) if include_legacy_signed_gap else (0,))):
            for start in range(0, len(selected), batch_size):
                cells = selected[start:start+batch_size]
                points, weights = _tetrahedron_quadrature(vertices[cells], orders.volume)
                yield cells, points, weights, domains
        for cells, points, weights in _physical_volume_batches(
                vertices, owned[relation == 0], radius_inner, orders, batch_size):
            yield cells, points, weights, (2,)
        for _, radius, sign, wall in wall_data:
            if sign < 0 and not include_legacy_signed_gap:
                continue
            for start in range(0, len(wall['cells']), batch_size):
                cells = wall['cells'][start:start+batch_size]
                rule = _spherical_facet_quadrature(wall['facets'][start:start+batch_size],
                                                  radius, orders.surface, orders.radial)
                domains = (1,) if sign < 0 else ((1, 2) if include_legacy_signed_gap else (2,))
                yield cells, rule['gap_points'].reshape(len(cells), -1, 3), sign*rule['gap_weights'].reshape(len(cells), -1), domains

    def numerical_fields(cells: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape, gradient, bary = _tetrahedral_p2_basis(vertices[cells], points)
        u = np.einsum('eqi,eid->eqd', shape, uvalues[cells])
        p = np.einsum('eqi,ei->eq', bary, qvalues[cells]) + _jump_values(mesh, jump, points)
        return u, p, gradient

    # A separate pressure pass precedes every squared-error calculation.
    alignment = np.zeros((3, 2))
    for cells, points, weights, domains in quadrature_batches():
        _, _, bary = _tetrahedral_p2_basis(vertices[cells], points)
        pressure = np.einsum('eqi,ei->eq', bary, qvalues[cells])+_jump_values(mesh, jump, points)
        reference = np.asarray(exact.evaluate('pressure', points.reshape(-1, 3))).reshape(weights.shape)
        values = [weights.sum(), np.sum(weights*(pressure-reference))]
        for domain in domains:
            alignment[domain] += values
    alignment = comm.allreduce(alignment, op=MPI.SUM)
    if np.any(alignment[list(active_domains), 0] <= 0):
        raise ArithmeticError("Integrated shell volume must be positive in every evaluated domain")
    offsets = np.zeros(3)
    offsets[list(active_domains)] = alignment[list(active_domains), 1]/alignment[list(active_domains), 0]
    # Columns: squared velocity error/reference, pressure error/reference,
    # absolute contribution scale, numerical/reference rotation moments,
    # squared full-stress error/reference and their absolute contribution scale.
    sums = np.zeros((3, 14))
    for cells, points, weights, domains in quadrature_batches():
        u, p, gradient = numerical_fields(cells, points)
        grad_u = np.einsum('eqid,eic->eqcd', gradient, uvalues[cells])
        unaligned_stress = viscosity*(grad_u+np.swapaxes(grad_u, -1, -2))-p[..., None, None]*np.eye(3)
        ue = np.asarray(exact.evaluate('velocity', points.reshape(-1, 3))).reshape(points.shape)
        pe = np.asarray(exact.evaluate('pressure', points.reshape(-1, 3))).reshape(weights.shape)
        se = np.asarray(exact.evaluate('stress', points.reshape(-1, 3))).reshape(*weights.shape, 3, 3)
        for domain in domains:
            integrands = np.stack((np.sum((u-ue)**2, axis=-1), np.sum(ue**2, axis=-1),
                                   (p-offsets[domain]-pe)**2, pe**2), axis=-1)
            sums[domain, :4] += np.sum(weights[..., None]*integrands, axis=(0, 1))
            sums[domain, 4] += np.sum(np.abs(weights[..., None]*integrands))
            sums[domain, 5:8] += np.sum(weights[..., None]*np.cross(points, u), axis=(0, 1))
            sums[domain, 8:11] += np.sum(weights[..., None]*np.cross(points, ue), axis=(0, 1))
            stress = unaligned_stress+offsets[domain]*np.eye(3)
            stress_integrands = np.stack((np.sum((stress-se)**2, axis=(-2, -1)),
                                         np.sum(se**2, axis=(-2, -1))), axis=-1)
            sums[domain, 11:13] += np.sum(weights[..., None]*stress_integrands, axis=(0, 1))
            sums[domain, 13] += np.sum(np.abs(weights[..., None]*stress_integrands))
    sums = comm.allreduce(sums, op=MPI.SUM)
    results = {}
    labels = ('polyhedral_volume', 'physical_signed_gap_volume', 'physical_inner_cut_volume')
    for domain in active_domains:
        label = labels[domain]
        values, measure = sums[domain], alignment[domain, 0]
        results[label] = dict(measure=float(measure), pressure_alignment_offset=float(offsets[domain]),
            velocity=_norm_result(values[0], values[1], measure, values[4], values[1]),
            pressure=_norm_result(values[2], values[3], measure, values[4], values[3]),
            stress=_norm_result(values[11], values[12], measure, values[13], values[12]),
            rotation_moments_numerical=values[5:8].tolist(),
            rotation_moments_reference=values[8:11].tolist())

    results['physical_sphere_walls'] = {}
    for name, radius, sign, wall in wall_data:
        def wall_batches():
            if sign < 0:
                for cell, points, weights, _, _ in inner_rules:
                    yield np.array([cell]), points[None], weights[None], -points[None]/radius
            else:
                for start in range(0, len(wall['cells']), batch_size):
                    cells = wall['cells'][start:start+batch_size]
                    rule = _spherical_facet_quadrature(wall['facets'][start:start+batch_size],
                                                      radius, orders.surface, orders.radial)
                    yield cells, rule['surface_points'], rule['surface_weights'], sign*rule['surface_normals']
        # area, component error/reference pairs, full reference velocity norm.
        totals = np.zeros(8)
        for cells, points, weights, normals in wall_batches():
            u, p, _ = numerical_fields(cells, points)
            ue = np.asarray(exact.evaluate('velocity', points.reshape(-1, 3))).reshape(points.shape)
            pe = np.asarray(exact.evaluate('pressure', points.reshape(-1, 3))).reshape(weights.shape)
            un, une = np.sum(u*normals, axis=-1), np.sum(ue*normals, axis=-1)
            ut, ute = u-un[..., None]*normals, ue-une[..., None]*normals
            terms = np.stack((np.ones_like(weights), (p-offsets[2]-pe)**2, pe**2,
                              (un-une)**2, une**2, np.sum((ut-ute)**2, axis=-1),
                              np.sum(ute**2, axis=-1), np.sum(ue**2, axis=-1)), axis=-1)
            totals += np.sum(weights[..., None]*terms, axis=(0, 1))
        totals = comm.allreduce(totals, op=MPI.SUM)
        if totals[0] <= 0:
            raise ArithmeticError(f"Physical wall {name} has nonpositive area")
        velocity_zero_scale = max(totals[7], sums[2, 1]/alignment[2, 0]*totals[0])
        results['physical_sphere_walls'][name] = dict(radius=radius, area=float(totals[0]),
            pressure_alignment_offset=float(offsets[2]),
            velocity_zero_reference_scale_squared=float(velocity_zero_scale),
            pressure=_norm_result(totals[1], totals[2], totals[0], totals.sum(), totals[2]),
            normal_velocity=_norm_result(totals[3], totals[4], totals[0], totals.sum(), velocity_zero_scale,
                'normal' in reference_zero_wall_velocity_components),
            tangential_velocity=_norm_result(totals[5], totals[6], totals[0], totals.sum(), velocity_zero_scale,
                'tangential' in reference_zero_wall_velocity_components))
    results['policy'] = dict(integration_version='physical_inner_cut_v1', relative_error_units='fraction',
        physical_domain='actual tetrahedral restriction outside inner ball + explicit outer facet continuation',
        wall_evaluation='inner actual cut-cell trace partitioned by original facet cones; outer adjacent continuation',
        signed_gap_norm='method-specific discrete norm using boundary-adjacent polynomial continuation across actual tetrahedron faces',
        legacy_signed_gap_evaluated=include_legacy_signed_gap,
        physical_fe_restriction='evaluated at inner cut cells; no physical support assigned to wholly void cells',
        wall_trace='inner cut-cell physical trace; outer adjacent tetrahedron continuation',
        wall_velocity_zero_reference='64*machine epsilon times max(full wall reference norm, volume reference RMS*sqrt(wall area))',
        prescribed_zero_wall_velocity_components=list(reference_zero_wall_velocity_components),
        prescribed_zero_policy='Relative denominator undefined; retain raw reference RMS and absolute errors, without modifying fields',
        wall_pressure_gauge='physical inner-cut volume alignment',
        pressure_reference_gauge='analytical reference unchanged',
        stress='constant-viscosity Cauchy stress; full Frobenius norm; domain pressure alignment',
        viscosity=viscosity,
        quadrature_orders=dict(volume=orders.volume, surface=orders.surface, radial=orders.radial),
        inner_cut_surface_order=max(12, orders.surface),
        batch_size=batch_size, ranks=comm.size)
    return results


def spherical_stokes_field_differences(mesh, velocity, reduced_pressure,
                                      reference_velocity, reference_pressure,
                                      radius_inner, radius_outer, orders, batch_size,
                                      inner_boundary, outer_boundary, *, viscosity):
    """Compare two same-mesh/lift FE states without analytical evaluation.

    The caller must establish identical model identity and coordinate pressure
    lift. That common lift cancels. Pressure differences are aligned by their
    physical-volume mean; the same offset enters the Cauchy-stress difference.
    Return absolute L2 values so the caller can normalize by ONE independently
    integrated analytical reference norm, rather than evaluating it per state.
    """
    if not (0 < radius_inner < radius_outer) or not np.isfinite(
            [radius_inner, radius_outer, viscosity]).all() or viscosity <= 0:
        raise ValueError("Require finite ordered radii and positive viscosity")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1
           for n in (orders.volume, orders.surface, orders.radial, batch_size)):
        raise ValueError("Quadrature orders and batch size must be positive integers")
    for field, degree, count in ((velocity, 2, 3), (reference_velocity, 2, 3),
                                 (reduced_pressure, 1, 1), (reference_pressure, 1, 1)):
        if (field.mesh is not mesh or field.degree != degree
                or field.num_components != count or not field.continuous):
            raise ValueError("Field differences require same-mesh continuous P2 velocities and P1 pressures")
    vertices = np.asarray(mesh._get_coords_for_basis(1, True))[mesh._cell_node_indices(1, True)]
    urows = _canonical_rows(mesh, velocity, vertices)
    prows = _canonical_rows(mesh, reduced_pressure, vertices)
    ref_urows = _canonical_rows(mesh, reference_velocity, vertices)
    ref_prows = _canonical_rows(mesh, reference_pressure, vertices)
    du = np.asarray(velocity.array)[urows, 0, :]-np.asarray(reference_velocity.array)[ref_urows, 0, :]
    dp = np.asarray(reduced_pressure.array)[prows, 0, 0]-np.asarray(reference_pressure.array)[ref_prows, 0, 0]
    owned = np.flatnonzero(mesh._get_owned_cells_mask())
    outer = _wall_faces(mesh, vertices, urows, owned, outer_boundary, radius_outer)

    def batches():
        yield from _physical_volume_batches(vertices, owned, radius_inner, orders, batch_size)
        for start in range(0, len(outer['cells']), batch_size):
            cells = outer['cells'][start:start+batch_size]
            rule = _spherical_facet_quadrature(outer['facets'][start:start+batch_size],
                                              radius_outer, orders.surface, orders.radial)
            yield cells, rule['gap_points'].reshape(len(cells), -1, 3), rule['gap_weights'].reshape(len(cells), -1)

    comm = mesh.dm.comm.tompi4py()
    alignment = np.zeros(2)
    for cells, points, weights in batches():
        _, _, bary = _tetrahedral_p2_basis(vertices[cells], points)
        pressure = np.einsum('eqi,ei->eq', bary, dp[cells])
        alignment += [weights.sum(), np.sum(weights*pressure)]
    alignment = comm.allreduce(alignment, op=MPI.SUM)
    if not np.isfinite(alignment).all() or alignment[0] <= 0:
        raise ArithmeticError("Physical field-difference integration has nonfinite/nonpositive measure")
    offset = alignment[1]/alignment[0]
    totals = np.zeros(6)
    for cells, points, weights in batches():
        shape, gradient, bary = _tetrahedral_p2_basis(vertices[cells], points)
        u = np.einsum('eqi,eid->eqd', shape, du[cells])
        pressure = np.einsum('eqi,ei->eq', bary, dp[cells])-offset
        grad_u = np.einsum('eqid,eic->eqcd', gradient, du[cells])
        stress = viscosity*(grad_u+np.swapaxes(grad_u, -1, -2))-pressure[..., None, None]*np.eye(3)
        integrands = np.stack((np.sum(u*u, axis=-1), pressure*pressure,
                              np.sum(stress*stress, axis=(-2, -1))), axis=-1)
        contributions = weights[..., None]*integrands
        totals[:3] += np.sum(contributions, axis=(0, 1))
        totals[3:] += np.sum(np.abs(contributions), axis=(0, 1))
    totals = comm.allreduce(totals, op=MPI.SUM)
    if not np.isfinite(totals).all() or np.any(totals[:3] < -128*np.finfo(float).eps*totals[3:]):
        raise ArithmeticError("Physical field differences have nonfinite/negative squared norms")
    values = np.sqrt(np.maximum(totals[:3], 0))
    return dict(measure=float(alignment[0]), pressure_mean_difference=float(offset),
        velocity_l2_difference=float(values[0]), pressure_l2_difference=float(values[1]),
        stress_l2_difference=float(values[2]),
        policy=dict(integration_version='physical_inner_cut_v1',
                    domain='actual inner-cell restriction + explicit outer continuation',
                    pressure='physical-volume mean aligned; identical coordinate lift must be established by caller',
                    analytical_evaluations=0, viscosity=viscosity,
                    quadrature_orders=dict(volume=orders.volume, surface=orders.surface, radial=orders.radial),
                    inner_cut_surface_order=max(12, orders.surface), ranks=comm.size))
