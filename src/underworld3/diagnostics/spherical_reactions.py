"""True-sphere reaction diagnostics using the coupled method's trace spaces.

Inner FE stress/pressure use the actual cut-cell trace, partitioned by original
multiplier facet cones. Outer traces retain their explicit facet continuation.
Harmonic projections measure one target and all off-target energy without
altering, filtering, or replacing any numerical field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import sympy as sp
import underworld3 as uw
from underworld3.maths.spherical_harmonics import real_harmonic_values
from underworld3.utilities._spherical_gap import _spherical_facet_quadrature
from underworld3.utilities._field_layout import _layout, _scatter_values
from underworld3.utilities._spherical_stokes_gap import _canonical_rows, _wall_faces
from underworld3.utilities._tetrahedral_gap import _tetrahedral_p2_basis
from underworld3.diagnostics.spherical import _physical_inner_surface_rules


@dataclass(frozen=True)
class HarmonicTarget:
    ell: int
    m: int
    component: str
    normalization: str


class StressReference(Protocol):
    def evaluate(self, field: str, coords: np.ndarray) -> np.ndarray: ...


def _target_values(points: np.ndarray, target: HarmonicTarget) -> np.ndarray:
    """Include the constant diagnostic mode excluded by UW's flow-mode API."""
    if target.ell == 0:
        if (target.m, target.component, target.normalization) != (0, 'cos', 'real_orthonormal'):
            raise ValueError("Degree-zero diagnostics require m=0, cos, real_orthonormal")
        return np.full(points.shape[:-1], 1/np.sqrt(4*np.pi))
    return real_harmonic_values(points.reshape(-1, 3), degree=target.ell,
        order=target.m, component=target.component, normalization=target.normalization).reshape(points.shape[:-1])


def spherical_stokes_reactions(mesh: uw.discretisation.Mesh,
    velocity: uw.discretisation.MeshVariable, reduced_pressure: uw.discretisation.MeshVariable,
    coordinate_pressure_jump: sp.Expr, normal_reaction: uw.discretisation.MeshVariable,
    tangential_reaction: uw.discretisation.MeshVariable, components: int,
    exact: StressReference, viscosity: float, radius_inner: float, radius_outer: float,
    physical_pressure_offset: float, surface_order: int, radial_order: int,
    batch_size: int, target: HarmonicTarget, inner_boundary: str, outer_boundary: str) -> dict:
    """Integrate paired-gauge reactions, traction residuals, and raw mode errors.

    Free slip evaluates the scalar P2 multiplier on the planar facet. Zero slip
    recovers Cartesian mapped coefficients from saved nodal physical normal and
    tangential reactions, interpolates them on the planar facet, then applies
    R/|y|. The same pressure offset is subtracted from pressure and normal lambda;
    tangential reactions are unchanged. Inner domain normals have negative sign.
    All errors are absolute surface RMS. Harmonic off-target RMS uses a stable
    second integration pass after the globally reduced coefficient is known.
    """
    if components not in (1, 3):
        raise ValueError("Constraint components must be one (free slip) or three (zero slip)")
    if not np.isfinite([viscosity, radius_inner, radius_outer, physical_pressure_offset]).all() or not (viscosity > 0 and 0 < radius_inner < radius_outer):
        raise ValueError("Require finite gauge, positive viscosity and ordered positive radii")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (surface_order, radial_order, batch_size)):
        raise ValueError("Quadrature orders and batch size must be positive integers")
    for field, degree, count in ((velocity, 2, 3), (reduced_pressure, 1, 1),
                                 (normal_reaction, 2, 1), (tangential_reaction, 2, 3)):
        if field.mesh is not mesh or field.degree != degree or field.num_components != count or not field.continuous:
            raise ValueError("Require continuous P2 velocity/reactions and P1 pressure on this mesh")
    jump = uw.function.expressions._unwrap_for_compilation(sp.sympify(coordinate_pressure_jump), keep_constants=False, return_self=False)
    if any(hasattr(type(atom), 'meshvar') for atom in jump.atoms(sp.Function)):
        raise ValueError("Pressure jump must be coordinate-only")
    comm = mesh.dm.comm.tompi4py()
    vertices = np.asarray(mesh._get_coords_for_basis(1, True))[mesh._cell_node_indices(1, True)]
    urows = _canonical_rows(mesh, velocity, vertices)
    prows = _canonical_rows(mesh, reduced_pressure, vertices)
    nrows = _canonical_rows(mesh, normal_reaction, vertices)
    trows = _canonical_rows(mesh, tangential_reaction, vertices)
    uvalues = np.asarray(velocity.array)[urows, 0, :]
    qvalues = np.asarray(reduced_pressure.array)[prows, 0, 0]
    # UW coordinate/field access can build collective PETSc layouts. Fetch on
    # every rank before uneven local wall batches, including empty wall ranks.
    normal_values = np.asarray(normal_reaction.array)[:, 0, 0]
    tangent_values = np.asarray(tangential_reaction.array)[:, 0, :]
    normal_coordinates = np.asarray(normal_reaction.coords)
    normal_map, normal_count = _layout(normal_reaction)
    owned = np.flatnonzero(mesh._get_owned_cells_mask())
    results = {}
    for name, radius, sign in ((inner_boundary, radius_inner, -1.), (outer_boundary, radius_outer, 1.)):
        wall = _wall_faces(mesh, vertices, urows, owned, name, radius)

        def pressure_and_stress(cells, points):
            _, gradient, bary = _tetrahedral_p2_basis(vertices[cells], points)
            pressure = np.einsum('eqi,ei->eq', bary, qvalues[cells])-physical_pressure_offset
            if jump.is_number:
                pressure += float(jump)
            else:
                pressure += np.asarray(uw.function.evaluate(jump,
                    np.ascontiguousarray(points.reshape(-1, 3)), coord_sys=mesh.N)).reshape(pressure.shape)
            grad_u = np.einsum('eqid,eic->eqcd', gradient, uvalues[cells])
            stress = viscosity*(grad_u+np.swapaxes(grad_u, -1, -2))-pressure[..., None, None]*np.eye(3)
            return pressure, stress

        inner_rules = []
        inner_coefficients = []
        if sign < 0:
            nr = np.take_along_axis(nrows[wall['cells']], wall['positions'], axis=1)
            wall.update(radius=radius, lid=(components*normal_map[nr, None]
                        + np.arange(components)).reshape(-1, 6*components).astype(PETSc.IntType))
            inner_rules = _physical_inner_surface_rules(mesh, vertices, owned, wall, radius, surface_order)
            needed = (np.unique(np.concatenate([rule[3].ravel() for rule in inner_rules]))
                      if inner_rules else np.empty(0, dtype=PETSc.IntType))
            coefficient = PETSc.Vec().createMPI(components*normal_count, comm=mesh.dm.comm)
            try:
                coefficient.set(0)
                if components == 1:
                    values = normal_values[nr]
                else:
                    tr = np.take_along_axis(trows[wall['cells']], wall['positions'], axis=1)
                    coordinates = normal_coordinates[nr]
                    distance = np.linalg.norm(coordinates, axis=-1)
                    nodal_normal = -coordinates/distance[..., None]
                    values = (normal_values[nr, None]*nodal_normal+tangent_values[tr])*distance[..., None]/radius
                coefficient.setValues(wall['lid'].ravel(), values.ravel(), addv=PETSc.InsertMode.INSERT_VALUES)
                coefficient.assemble()
                fetched = _scatter_values(coefficient, needed)
            finally:
                coefficient.destroy()
            inner_coefficients = [fetched[np.searchsorted(needed, rule[3])].reshape(-1, 6, components)
                                  for rule in inner_rules]

        def batches() -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            if sign < 0:
                for (cell, x, weights, _, trace), coefficients in zip(inner_rules, inner_coefficients, strict=True):
                    points, normals = x[None], -x[None]/radius
                    pressure, stress = pressure_and_stress(np.array([cell]), points)
                    if components == 1:
                        scalar = np.sum(trace*coefficients[..., 0], axis=1)-physical_pressure_offset
                        rho = scalar[None, :, None]*normals
                    else:
                        rho = np.einsum('qi,qic->qc', trace, coefficients)[None]
                        rho -= physical_pressure_offset*normals
                    yield points, normals, weights[None], pressure, rho, stress
                return
            for start in range(0, len(wall['cells']), batch_size):
                cells = wall['cells'][start:start+batch_size]
                positions = wall['positions'][start:start+batch_size]
                rule = _spherical_facet_quadrature(wall['facets'][start:start+batch_size], radius, surface_order, radial_order)
                points, normals = rule['surface_points'], sign*rule['surface_normals']
                pressure, stress = pressure_and_stress(cells, points)
                planar_shape, _, _ = _tetrahedral_p2_basis(vertices[cells], rule['planar_points'])
                trace = np.take_along_axis(planar_shape, positions[:, None, :], axis=2)
                nr = np.take_along_axis(nrows[cells], positions, axis=1)
                nodal_normal = normal_values[nr]
                if components == 1:
                    scalar = np.einsum('eqi,ei->eq', trace, nodal_normal)-physical_pressure_offset
                    rho = scalar[..., None]*normals
                else:
                    tr = np.take_along_axis(trows[cells], positions, axis=1)
                    coordinates = normal_coordinates[nr]
                    distance = np.linalg.norm(coordinates, axis=-1)
                    nodal_n = sign*coordinates/distance[..., None]
                    nodal_rho = nodal_normal[..., None]*nodal_n+tangent_values[tr]
                    coefficient = nodal_rho*distance[..., None]/radius
                    rho = np.einsum('eqi,eid->eqd', trace, coefficient)
                    rho *= (radius/np.linalg.norm(rule['planar_points'], axis=-1))[..., None]
                    rho -= physical_pressure_offset*normals
                yield points, normals, rule['surface_weights'], pressure, rho, stress

        totals = np.zeros(9)
        for points, n, weights, pressure, rho, stress in batches():
            sigma_exact = np.asarray(exact.evaluate('stress', points.reshape(-1, 3))).reshape(*points.shape[:-1], 3, 3)
            exact_rho = -np.einsum('eqij,eqj->eqi', sigma_exact, n)
            normal, exact_normal = np.sum(rho*n, axis=-1), np.sum(exact_rho*n, axis=-1)
            tangent_error = (rho-normal[..., None]*n)-(exact_rho-exact_normal[..., None]*n)
            traction = np.einsum('eqij,eqj->eqi', stress, n)
            residual = rho+traction
            tangent_traction = traction-np.sum(traction*n, axis=-1)[..., None]*n
            harmonic = _target_values(points, target)
            terms = np.stack((np.ones_like(weights), (normal-exact_normal)**2,
                np.sum(tangent_error**2, axis=-1), np.sum(residual*n, axis=-1)**2,
                np.sum(residual**2, axis=-1), np.sum(tangent_traction**2, axis=-1),
                pressure*harmonic, normal*harmonic, harmonic**2), axis=-1)
            totals += np.sum(weights[..., None]*terms, axis=(0, 1))
        totals = comm.allreduce(totals, op=MPI.SUM)
        if not np.isfinite(totals).all() or totals[0] <= 0 or totals[8] <= 0:
            raise ArithmeticError(f"Nonfinite or nonpositive surface/mode integrals for wall {name}")
        coefficients = totals[6:8]/totals[8]
        off_target = np.zeros(2)
        for points, n, weights, pressure, rho, _ in batches():
            harmonic = _target_values(points, target)
            normal = np.sum(rho*n, axis=-1)
            off_target += [np.sum(weights*(pressure-coefficients[0]*harmonic)**2),
                           np.sum(weights*(normal-coefficients[1]*harmonic)**2)]
        off_target = comm.allreduce(off_target, op=MPI.SUM)
        if not np.isfinite(off_target).all():
            raise ArithmeticError(f"Nonfinite off-target energy for wall {name}")
        results[name] = dict(radius=radius, area=float(totals[0]),
            normal_reaction_error_rms=float(np.sqrt(totals[1]/totals[0])),
            tangential_reaction_error_rms=float(np.sqrt(totals[2]/totals[0])),
            normal_traction_residual_rms=float(np.sqrt(totals[3]/totals[0])),
            full_traction_residual_rms=float(np.sqrt(totals[4]/totals[0])),
            fe_tangential_traction_rms=float(np.sqrt(totals[5]/totals[0])),
            free_natural_tangential_traction_residual_rms=float(np.sqrt(totals[5]/totals[0])) if components == 1 else None,
            pressure_target_coefficient=float(coefficients[0]), normal_reaction_target_coefficient=float(coefficients[1]),
            pressure_off_target_rms=float(np.sqrt(off_target[0]/totals[0])),
            normal_reaction_off_target_rms=float(np.sqrt(off_target[1]/totals[0])))
    return dict(walls=results, policy=dict(components=components, pressure_and_normal_reaction_offset=physical_pressure_offset,
        integration_version='physical_inner_cut_v1',
        fe_trace='inner actual cut-cell trace partitioned by original multiplier cones; outer adjacent continuation',
        multiplier_trace='scalar P2 on planar patch' if components == 1 else 'Cartesian planar P2 coefficients times R/|y|',
        harmonic_projection='raw single-target diagnostic only; stable second-pass off-target RMS; no field mutation/filtering',
        target=dict(ell=target.ell, m=target.m, component=target.component, normalization=target.normalization),
        surface_order=surface_order, inner_cut_surface_order=max(12, surface_order),
        radial_order=radial_order, batch_size=batch_size, ranks=comm.size))
