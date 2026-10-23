"""Private spatial reaction transfer from polygonal to circular annulus walls.

Only boundary traces are replicated. Sparse solves are one-dimensional line
mass systems; no volume field is gathered. The spatial target is independent
of Fourier fitting and the original solver reaction cache remains unchanged.

The consistent boundary flux (CBF) principle recovers surface stress from the
discrete momentum balance [1]_. The circular-minus-polygonal stress moments and
Taylor shift below are UW3 geometry adaptations, not a scheme or stability
result established in that paper.

References
----------
.. [1] Zhong, S., Gurnis, M., and Hulbert, G. (1993). Accurate determination
   of surface normal stress in viscous flow from a consistent boundary flux
   method. Physics of the Earth and Planetary Interiors, 78(1-2), 1-8.
   https://doi.org/10.1016/0031-9201(93)90078-N

Only nondimensional mesh coordinates, fields and input scalars are supported.
Active reference-unit scaling is outside this helper's supported contract.
"""

import numpy as np
import underworld3 as uw
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from ._cylindrical_wall import _trace_rule, _numerical_trace


def _circular_reaction_field(edges, trace, raw_reaction, radius, sign, quadrature):
    """Transfer CBF nodal moments to a circular P2 trace without Fourier fitting.

    R_circle = R_CBF + L_circle(sigma_h) - L_polygon(sigma_h). Remove each
    traction mean on its own geometry, then invert the circular consistent
    line mass. The circular stress uses the same first-order Taylor shift as
    the coefficient diagnostic. Keep the assembled CBF residual contribution.

    The residual-based stress recovery follows the CBF principle of Zhong,
    Gurnis and Hulbert (1993), cited in the module. The additive moment transfer
    is our extension to a physical circular trace; its accuracy is verified
    independently and no published stability proof is claimed for it.
    """
    rule = _trace_rule(edges, sign, 2, quadrature)
    t = (np.polynomial.legendre.leggauss(quadrature)[0] + 1) / 2
    shape = np.array([(1 - t) * (1 - 2 * t), t * (2 * t - 1), 4 * t * (1 - t)]).T
    edge_nodes = np.stack((edges[:, 0], edges[:, 1], edges.mean(axis=1)), axis=1)
    nodes, index = np.unique(edge_nodes.reshape(-1, 2), axis=0, return_inverse=True)
    index = index.reshape(-1, 3)
    normal_nodes = sign * edge_nodes / np.linalg.norm(edge_nodes, axis=-1)[..., None]
    r = np.linalg.norm(rule["points"], axis=-1)
    radial = rule["points"] / r[..., None]
    tangent = np.stack((-radial[..., 1], radial[..., 0]), axis=-1)
    nr = np.sum(rule["normal"][:, None, :] * radial, axis=-1)
    nt = np.sum(rule["normal"][:, None, :] * tangent, axis=-1)
    traction = (trace["rr"] * nr + trace["rt"] * nt)[..., None] * radial + (
        trace["rt"] * nr + trace["tt"] * nt
    )[..., None] * tangent
    polygon_load = -np.einsum("eq,qi,eqd,eid->ei", rule["weight"], shape, traction, normal_nodes)
    circle_weight = rule["weight"] * radius / r * np.abs(nr)
    shifted_stress = trace["rr"] + (radius - r) * trace["rr_derivative"]
    circle_load = -np.einsum("eq,qi,eq->ei", circle_weight, shape, shifted_stress)
    masses, loads = [], []
    for weight, load in ((rule["weight"], polygon_load), (circle_weight, circle_load)):
        local = np.einsum("eq,qi,qj->eij", weight, shape, shape)
        rows = np.broadcast_to(index[:, :, None], local.shape).ravel()
        columns = np.broadcast_to(index[:, None, :], local.shape).ravel()
        mass = coo_matrix((local.ravel(), (rows, columns)), shape=(len(nodes), len(nodes))).tocsr()
        total = np.zeros(len(nodes))
        np.add.at(total, index.ravel(), load.ravel())
        uniform = mass @ np.ones(len(nodes))
        total -= total.sum() / uniform.sum() * uniform
        masses.append(mass)
        loads.append(total)
    uniform = masses[0] @ np.ones(len(nodes))
    raw = np.asarray(raw_reaction).reshape(-1)
    raw_mean = float(uniform @ raw / uniform.sum())
    target = masses[0] @ (raw - raw_mean) + loads[1] - loads[0]
    corrected = spsolve(masses[1], target)
    checks = dict(
        raw_reaction_mean_removed=raw_mean,
        maximum_mass_moment_error=float(np.max(np.abs(masses[1] @ corrected - target))),
        circular_reaction_integral=float(np.sum(masses[1] @ corrected)),
        mapped_circle_length=float(circle_weight.sum()),
        correction_dof_l2=float(np.linalg.norm(corrected - raw)),
    )
    np.testing.assert_allclose(checks["mapped_circle_length"], 2 * np.pi * radius, rtol=1e-12)
    if max(checks["maximum_mass_moment_error"], abs(checks["circular_reaction_integral"])) > 1e-11:
        raise ArithmeticError(f"Circular reaction field moment/gauge failure: {checks}")
    return nodes, corrected, checks

def _validate_spatial_target(wall, field):
    """Boundary masks are valid only on the wall's own scalar P2 coordinates."""
    if (field.mesh is not wall.mesh or field.degree != 2
            or not field.continuous or field.num_components != 1):
        raise ValueError("Circular wall fields require a continuous scalar P2 target on the same mesh.")


def _recover_circular_topography(stokes, wall, field, buoyancy_scale):
    r"""Write circular-wall topography from CBF reactions and numerical moments.

    Recover polygonal normal traction from the discrete Stokes momentum
    residual using a consistent boundary mass, following the CBF principle
    of Zhong, Gurnis and Hulbert (1993; module reference). Apply our circular
    moment transfer, then use the signed buoyancy balance
    :math:`h=-\sigma_{nn}/(\Delta\rho g)`. No projected volume stress or
    analytical Stokes field replaces the assembled reaction.
    """
    if buoyancy_scale == 0:
        raise ValueError("Topography requires nonzero buoyancy_scale.")
    _validate_spatial_target(wall, field)
    coords, values = stokes.boundary_normal_traction(wall.name, mass="consistent")
    gathered_coords = np.concatenate(uw.mpi.comm.allgather(np.asarray(coords).reshape(-1, 2)))
    gathered_values = np.concatenate(uw.mpi.comm.allgather(np.asarray(values).reshape(-1)))
    distance, index = cKDTree(gathered_coords).query(wall.nodes)
    if distance.max() > 1e-10:
        raise ValueError(f"CBF nodes do not cover circular wall {wall.name}.")
    velocity, pressure = wall.cell_values()
    trace = _numerical_trace(wall.vertices, velocity, pressure, wall.rule["points"], *wall.physics)
    nodes, reaction, checks = _circular_reaction_field(
        wall.edges, trace, gathered_values[index], wall.radius, wall.sign, wall.quadrature
    )
    np.testing.assert_array_equal(nodes, wall.nodes)
    distance, index = cKDTree(nodes).query(np.asarray(field.coords))
    boundary = distance < 1e-10
    field.array[boundary, 0, 0] = -reaction[index[boundary]] / buoyancy_scale
    return checks

def _circular_field_coefficient(wall, field, mode):
    """Integrate a saved spatial P2 wall field on the physical circle (spectral only)."""
    _validate_spatial_target(wall, field)
    boundary = wall.local_boundary
    coords = np.concatenate(uw.mpi.comm.allgather(np.asarray(field.coords)[boundary]))
    values = np.concatenate(uw.mpi.comm.allgather(np.asarray(field.array)[boundary, 0, 0]))
    distance, index = cKDTree(coords).query(wall.nodes)
    np.testing.assert_allclose(distance, 0, atol=1e-10)
    t = (np.polynomial.legendre.leggauss(wall.quadrature)[0] + 1) / 2
    shape = np.array([(1 - t) * (1 - 2 * t), t * (2 * t - 1), 4 * t * (1 - t)]).T
    samples = np.einsum("qi,ei->eq", shape, values[index][wall.node_index])
    r = np.linalg.norm(wall.rule["points"], axis=-1)
    nr = np.sum(wall.rule["normal"][:, None, :] * wall.rule["points"] / r[..., None], axis=-1)
    weights = wall.rule["weight"] * wall.radius / r * np.abs(nr)
    harmonic = np.cos(mode * np.arctan2(wall.rule["points"][..., 1], wall.rule["points"][..., 0]))
    samples -= np.sum(weights * samples) / weights.sum()
    return float(np.sum(weights * samples * harmonic) / np.sum(weights * harmonic**2))
