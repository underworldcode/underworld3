"""Physical circular traction from standard fully Dirichlet Stokes reactions.

The vector consistent boundary flux (CBF) is retained as a weak load, corrected
by circle-minus-polygon numerical stress moments, then represented in a rotating
normal/tangent P2 trial basis. No pressure mean is removed and no reference
Stokes solution is used. Only boundary traces are replicated; small SciPy sparse
line systems follow the existing circular reaction/reconstruction implementation.

Zhong, Gurnis & Hulbert (1993), doi:10.1016/0031-9201(93)90078-N, motivates CBF.
Brunatova, Dokken, Valen-Sendstad & Hron (2025), doi:10.1002/cnm.70086,
Section 2.3.2, equations (9)-(10), describes full Dirichlet variational traction
recovery and normal/tangential decomposition. Those works do not establish
the circular moment transfer, rotating trial basis or reconstruction used here.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import sympy as sp
import underworld3 as uw
from underworld3.systems.solvers import _penalty_value

from ._cylindrical_dirichlet import _ZeroSlipWall, _facet_normals
from ._cylindrical_reactions import _validate_spatial_target
from ._cylindrical_reconstruction import _recover_circular_trace
from ._cylindrical_wall import _numerical_trace, _trace_rule


def _traction_rule(edges, radius, sign, quadrature):
    """Radially map polygon P2 tests onto the physical circle, with exact ds."""
    polygon = _trace_rule(edges, sign, 0, quadrature)
    polygon_normal = _facet_normals(edges, sign)
    points = polygon["points"]
    radial = points/np.linalg.norm(points, axis=-1)[..., None]
    circle = radius*radial
    tangent = np.stack((-radial[..., 1], radial[..., 0]), axis=-1)
    t = (np.polynomial.legendre.leggauss(quadrature)[0]+1)/2
    shape = np.stack(((1-t)*(1-2*t), t*(2*t-1), 4*t*(1-t)), axis=-1)
    edge_nodes = np.concatenate((edges, edges.mean(axis=1)[:, None]), axis=1)
    nodes, index = np.unique(edge_nodes.reshape(-1, 2), axis=0, return_inverse=True)
    normal = sign*radial
    circle_weight = polygon["weight"]*radius/np.linalg.norm(points, axis=-1)*np.abs(
        np.einsum("ed,eqd->eq", polygon_normal, radial))
    return dict(points=points, circle_points=circle, nodes=nodes, index=index.reshape(-1, 3),
        shape=shape, polygon_weight=polygon["weight"], circle_weight=circle_weight,
        polygon_normal=polygon_normal, normal=normal,
        frame=np.stack((normal, tangent), axis=-1), radius=radius, sign=sign)


def _regular_cartesian_stress(vertices, velocity, pressure, points, viscosity):
    """Continue the affine-cell P2/P1 regular-pressure stress one-sidedly."""
    trace = _numerical_trace(vertices, velocity, pressure, points, viscosity,
                             sheet_radius=0, outer_radius=1, mode=0, amplitude=0)
    radial = points/np.linalg.norm(points, axis=-1)[..., None]
    tangent = np.stack((-radial[..., 1], radial[..., 0]), axis=-1)
    return (trace["rr"][..., None, None]*np.einsum("eqi,eqj->eqij", radial, radial)
        + trace["tt"][..., None, None]*np.einsum("eqi,eqj->eqij", tangent, tangent)
        + trace["rt"][..., None, None]*(np.einsum("eqi,eqj->eqij", radial, tangent)
                                     + np.einsum("eqi,eqj->eqij", tangent, radial)))


def _transfer_circle_traction(rule, raw_traction, polygon_stress, circle_stress):
    r"""Preserve every transferred Cartesian moment in a rotating P2 trial space.

    ``raw_traction`` is the physical-sign vector CBF for the regular pressure q.
    Subtract polygon regular-stress moments and add circle physical-stress
    moments: :math:`R_c=R_q+L_c(\sigma_{q+p_J})-L_h(\sigma_q)`.
    The two unknowns are :math:`\sigma_{nn}` and :math:`t\cdot e_\theta`;
    the latter equals ``sign*sigma_rtheta``. No mean or modal fit is applied.
    """
    index, shape, size = rule["index"], rule["shape"], len(rule["nodes"])
    local_mass = np.einsum("eq,qi,qj->eij", rule["polygon_weight"], shape, shape)
    rows = np.broadcast_to(index[:, :, None], local_mass.shape).ravel()
    columns = np.broadcast_to(index[:, None, :], local_mass.shape).ravel()
    mass = coo_matrix((local_mass.ravel(), (rows, columns)), shape=(size, size)).tocsr()
    polygon_load = np.einsum("eq,qi,eqcd,ed->eic", rule["polygon_weight"], shape,
                             polygon_stress, rule["polygon_normal"])
    circle_load = np.einsum("eq,qi,eqcd,eqd->eic", rule["circle_weight"], shape,
                            circle_stress, rule["normal"])
    raw_load = mass@raw_traction
    target = raw_load.copy()
    np.add.at(target, index.ravel(), (circle_load-polygon_load).reshape(-1, 2))
    # Rows are Cartesian test components; columns are normal/tangent trials.
    local = np.einsum("eq,qi,qj,eqck->eicjk", rule["circle_weight"], shape, shape, rule["frame"])
    rows = np.broadcast_to(2*index[:, :, None, None, None]+np.arange(2)[None, None, :, None, None], local.shape)
    columns = np.broadcast_to(2*index[:, None, None, :, None]+np.arange(2)[None, None, None, None, :], local.shape)
    circular_mass = coo_matrix((local.ravel(), (rows.ravel(), columns.ravel())), shape=(2*size, 2*size)).tocsr()
    values = spsolve(circular_mass, target.ravel()).reshape(-1, 2)
    if not np.all(np.isfinite(values)):
        raise ArithmeticError("Circular rotating-frame traction solve returned nonfinite values.")
    sampled = np.einsum("qi,eik->eqk", shape, values[index])
    traction = np.einsum("eqk,eqck->eqc", sampled, rule["frame"])
    resultant = np.einsum("eq,eqc->c", rule["circle_weight"], traction)
    checks = dict(maximum_cartesian_moment_residual=float(np.max(np.abs(
        circular_mass@values.ravel()-target.ravel()))),
        cartesian_resultant=resultant.tolist(), cartesian_resultant_target=target.sum(axis=0).tolist(),
        raw_polygon_cartesian_resultant=raw_load.sum(axis=0).tolist(),
        circle_transfer_resultant_change=(target-raw_load).sum(axis=0).tolist(),
        polygon_numerical_stress_resultant=polygon_load.sum(axis=(0, 1)).tolist(),
        circle_numerical_stress_resultant=circle_load.sum(axis=(0, 1)).tolist(),
        cartesian_resultant_residual=float(np.max(np.abs(resultant-target.sum(axis=0)))),
        circular_length=float(rule["circle_weight"].sum()), pressure_mean_removed=False,
        normal_mean=float(np.sum(rule["circle_weight"]*sampled[..., 0])/(2*np.pi*rule["radius"])),
        tangential_mean=float(np.sum(rule["circle_weight"]*sampled[..., 1])/(2*np.pi*rule["radius"])))
    return values, checks


def _trace_change_checks(rule, source, target):
    """Measure changed nodal loads separately from conserved physical moments."""
    delta = np.einsum("qi,eik->eqk", rule["shape"], (target-source)[rule["index"]])
    traction = np.einsum("eqk,eqck->eqc", delta, rule["frame"])
    loads = np.zeros((len(source), 2))
    local = np.einsum("eq,qi,eqc->eic", rule["circle_weight"], rule["shape"], traction)
    np.add.at(loads, rule["index"].ravel(), local.reshape(-1, 2))
    return dict(maximum_changed_cartesian_nodal_moment=float(np.max(np.abs(loads))),
        maximum_cartesian_resultant_change=float(np.max(np.abs(np.einsum(
            "eq,eqc->c", rule["circle_weight"], traction)))),
        scalar_integral_changes=np.einsum("eq,eqk->k", rule["circle_weight"], delta).tolist(),
        fine_scale_cbf_moments_preserved=False)


def _boundary_pressure_values(stokes, wall):
    """Gather owned boundary-cell P1 values, including ranks with no wall cells."""
    rows = stokes.mesh._cell_node_indices(1, True).reshape(-1, 3)[wall.cells]
    local = np.asarray(stokes.p.array)[rows, 0, 0]
    return np.concatenate(uw.mpi.comm.allgather(local))


def _recover_cylindrical_zero_slip_traction(stokes, *, wall, normal_field, shear_field,
        recovered_normal_field, recovered_shear_field, viscosity, pressure_jump,
        degree, halfwidth_edges, quadrature):
    """Write distinct raw-circle/reconstructed physical normal and shear fields."""
    if not isinstance(wall, _ZeroSlipWall) or wall.stokes is not stokes:
        raise ValueError("Supply this Stokes system's zero-slip wall handle.")
    if getattr(stokes, "_rotated_freeslip_bcs", None):
        raise ValueError("Full Dirichlet traction recovery requires standard Stokes.")
    if not np.isfinite(viscosity) or viscosity <= 0:
        raise ValueError("Traction recovery requires finite positive constant viscosity.")
    constitutive_viscosity = uw.function.unwrap(stokes.constitutive_model.K,
                                               keep_constants=False, return_self=False)
    if (_penalty_value(stokes.penalty) != 0 or stokes.devss_viscosity is not None
            or constitutive_viscosity.is_number is not True
            or float(constitutive_viscosity) != viscosity):
        raise ValueError("Require matching constant viscosity and zero augmentation/DEVSS.")
    if int(quadrature) != quadrature or quadrature < 3:
        raise ValueError("Traction quadrature requires an integer number of points >= 3.")
    targets = (normal_field, shear_field, recovered_normal_field, recovered_shear_field)
    if len({id(field) for field in targets}) != 4:
        raise ValueError("Circular and reconstructed normal/shear targets must be four distinct fields.")
    for field in targets:
        _validate_spatial_target(wall, field)
    rule = _traction_rule(wall.edges, wall.radius, wall.sign, quadrature)
    np.testing.assert_array_equal(rule["nodes"], wall.nodes)
    velocity, pressure = wall.cell_values(), _boundary_pressure_values(stokes, wall)
    polygon_stress = _regular_cartesian_stress(wall.vertices, velocity, pressure, rule["points"], viscosity)
    circle_stress = _regular_cartesian_stress(wall.vertices, velocity, pressure, rule["circle_points"], viscosity)
    jump = np.asarray(uw.function.evaluate(pressure_jump, rule["circle_points"].reshape(-1, 2))).reshape(
        rule["circle_weight"].shape)
    circle_stress -= jump[..., None, None]*np.eye(2)
    # Standard CBF freshly reassembles in the current pressure/essential-data gauge.
    coordinates, traction = stokes.boundary_flux(wall.name, mass="consistent", remove_mean=False)
    # Native assembly first wires constitutive placeholder gradients to this solver.
    stress_difference = stokes.stress_deviator - 2*sp.nsimplify(viscosity, rational=True)*stokes.strainrate
    if any(sp.simplify(uw.function.unwrap(value, keep_constants=False, return_self=False)) != 0
           for value in stress_difference):
        raise ValueError("Traction recovery requires the symmetric Newtonian stress 2*viscosity*strainrate.")
    coordinates = np.concatenate(uw.mpi.comm.allgather(np.asarray(coordinates).reshape(-1, 2)))
    traction = np.concatenate(uw.mpi.comm.allgather(np.asarray(traction).reshape(-1, 2)))
    distance, index = cKDTree(coordinates).query(wall.nodes)
    if np.max(distance) > 1e-10*wall.radius:
        raise ValueError("The standard CBF trace does not cover this circular wall.")
    values, checks = _transfer_circle_traction(rule, traction[index], polygon_stress, circle_stress)
    recovered, reconstruction = [], []
    for component in range(2):
        nodes, value, component_checks = _recover_circular_trace(wall.edges, values[:, component],
            wall.radius, degree, halfwidth_edges, quadrature)
        np.testing.assert_array_equal(nodes, wall.nodes)
        recovered.append(value)
        reconstruction.append(component_checks)
    recovered = np.stack(recovered, axis=-1)
    # The public shear field is sigma_rtheta, not outward-normal tangential traction.
    for field, value in zip(targets, (values[:, 0], wall.sign*values[:, 1],
                                     recovered[:, 0], wall.sign*recovered[:, 1])):
        selected = wall.local_boundary
        field.array[selected, 0, 0] = value[wall.local_index[selected]]
    checks.update(quadrature=quadrature, viscosity=viscosity,
        pressure_transfer="R_regular+L_circle(sigma_physical)-L_polygon(sigma_regular)",
        physical_circle_pressure_jump_maximum=float(np.max(np.abs(jump))),
        reconstruction=_trace_change_checks(rule, values, recovered),
        component_reconstruction_checks=reconstruction,
        pressure_proxy="minus normal stress is a zero-slip boundary estimate, not the FE pressure field")
    return checks
