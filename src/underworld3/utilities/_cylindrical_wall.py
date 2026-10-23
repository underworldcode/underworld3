"""Private lagged wall transfer on affine P2/continuous-P1 annuli.

Only owned boundary-cell traces are replicated across ranks. The sparse SciPy
solves below are bounded one-dimensional boundary mass systems, not volume
operators; the Stokes system remains distributed PETSc. Parallel collectives
are internal implementation details, never part of the notebook contract.
No analytical Stokes data or coefficient fitting enters a numerical wall update.

Taylor boundary transfer is motivated by N. M. Atallah, C. Canuto and G. Scovazzi
(2020), "The second-generation Shifted Boundary Method and its numerical
analysis", Computer Methods in Applied Mechanics and Engineering 372, 113341,
https://doi.org/10.1016/j.cma.2020.113341, Section 2.2, equations (8)-(13).
This lagged strong free-slip/traction projection is a UW3 adaptation, not their
Nitsche formulation. Their Section 4.2 assumes body-fitted Neumann boundaries;
its stability/convergence analysis does not cover this implementation.
"""

import numpy as np
import underworld3 as uw
from mpi4py import MPI
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from ._cylindrical_pressure import _pressure_jump_values


def _trace_rule(edges: np.ndarray, sign: int, mode: int, quadrature: int) -> dict:
    """Return P2-interpolated nodal radial tests and polygonal line quadrature.

    The weighted test is I_h(n_node*cos(mode*theta_node)); its normalization
    is the integral of the continuous cos² over the polygonal wall, as in UW3.
    """
    t, weight = np.polynomial.legendre.leggauss(quadrature)
    t, weight = (t + 1) / 2, weight / 2
    shape = np.array([(1 - t) * (1 - 2 * t), t * (2 * t - 1), 4 * t * (1 - t)]).T
    midpoint = edges.mean(axis=1)
    nodes = np.stack([edges[:, 0], edges[:, 1], midpoint], axis=1)
    node_normal = sign * nodes / np.linalg.norm(nodes, axis=2)[..., None]
    node_angle = np.arctan2(nodes[..., 1], nodes[..., 0])
    node_cosine = np.cos(mode * node_angle)
    node_tangent = sign * np.stack([-node_normal[..., 1], node_normal[..., 0]], axis=-1)
    points = (1 - t)[None, :, None] * edges[:, None, 0] + t[None, :, None] * edges[:, None, 1]
    return dict(
        points=points,
        normal=sign * midpoint / np.linalg.norm(midpoint, axis=1)[:, None],
        weight=np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)[:, None] * weight,
        harmonic=np.cos(mode * np.arctan2(points[..., 1], points[..., 0])),
        test=np.einsum("qi,eid->eqd", shape, node_normal * node_cosine[..., None]),
        test_one=np.einsum("qi,eid->eqd", shape, node_normal),
        tangent_test=np.einsum(
            "qi,eid->eqd", shape, node_tangent * np.sin(mode * node_angle)[..., None]
        ),
    )

def _tangent_load_density(edges: np.ndarray, traction: np.ndarray, sign: int, quadrature: int):
    """Project integrated loads tangentially at P2 nodes; preserve their moments.

    Project loads before inverting the mass matrix. Projecting pointwise density
    instead would generally inject normal nodal loads into the reaction.
    """
    rule = _trace_rule(edges, sign, 2, quadrature)
    t = (np.polynomial.legendre.leggauss(quadrature)[0] + 1) / 2
    shape = np.array([(1 - t) * (1 - 2 * t), t * (2 * t - 1), 4 * t * (1 - t)]).T
    points = np.stack((edges[:, 0], edges[:, 1], edges.mean(axis=1)), axis=1)
    nodes, index = np.unique(points.reshape(-1, 2), axis=0, return_inverse=True)
    index = index.reshape(-1, 3)
    local_mass = np.einsum("eq,qi,qj->eij", rule["weight"], shape, shape)
    local_load = np.einsum("eq,qi,eqd->eid", rule["weight"], shape, traction)
    row = np.broadcast_to(index[:, :, None], local_mass.shape).ravel()
    column = np.broadcast_to(index[:, None, :], local_mass.shape).ravel()
    mass = coo_matrix((local_mass.ravel(), (row, column)), shape=(len(nodes), len(nodes))).tocsr()
    load = np.zeros_like(nodes)
    np.add.at(load, index.ravel(), local_load.reshape(-1, 2))
    normal = sign * nodes / np.linalg.norm(nodes, axis=1)[:, None]
    tangent_load = load - np.sum(load * normal, axis=1)[:, None] * normal
    density = spsolve(mass, tangent_load)
    reconstructed = mass @ density
    checks = dict(
        nodes=len(nodes),
        edges=len(edges),
        maximum_mass_moment_error=float(np.max(np.abs(reconstructed - tangent_load))),
        maximum_normal_nodal_load=float(np.max(np.abs(np.sum(reconstructed * normal, axis=1)))),
        tangential_load_norm=float(np.linalg.norm(tangent_load)),
    )
    if checks["maximum_mass_moment_error"] > 1e-12 or checks["maximum_normal_nodal_load"] > 1e-12:
        raise ArithmeticError(f"Tangential boundary-load projection failed: {checks}")
    return nodes, density, checks

def _wall_solve_succeeded(info):
    """Accept a converged residual, including a valid zero-increment warm start."""
    return bool(
        info["converged"]
        and (
            info["ksp_reason"] > 0
            or (info["ksp_reason"] == 0 and info["nonlinear_iterations"] == 0)
        )
    )

def _solve_with_wall_updates(stokes, walls, updates):
    """Alternate converged Stokes solves and numerical wall updates.

    The final solve uses the last recorded boundary fields. A residual-accepted
    zero-increment warm start is valid. Partial history is stored on the solver
    before raising so the caller can record a failed step without accepting it.
    Requires the caller's ``petsc_use_pressure_nullspace=True`` configuration:
    accepted history reads the projected pressure-gauge residual. This helper
    does not install a pressure gauge or support another gauge configuration.
    The fixed-point schedule and choice of four updates are UW3 benchmark
    choices; they are not an iteration theorem from the shifted-boundary paper
    cited in this module's documentation.
    """
    if updates < 0 or int(updates) != updates:
        raise ValueError("Wall updates must be a non-negative integer.")
    velocity = stokes.u
    history = []
    stokes.cylindrical_wall_history = history
    for iteration in range(updates + 1):
        previous = np.asarray(velocity.array).copy()
        stokes.solve()
        info = stokes._rotated_freeslip_info
        step = dict(
            iteration=iteration,
            velocity_change_maximum=uw.mpi.comm.allreduce(
                float(np.max(np.abs(velocity.array - previous), initial=0.0)), op=MPI.MAX
            ),
            ksp_iterations=info["ksp_its"],
            ksp_reason=int(info["ksp_reason"]),
            nonlinear_iterations=int(info["nonlinear_iterations"]),
            converged=bool(info["converged"]),
            residual_norm=float(info["rnorm"]),
            pressure_gauge_residual=float(stokes._rotated_pressure_gauge_residual),
            boundary_updates={},
        )
        history.append(step)
        if not _wall_solve_succeeded(info):
            raise RuntimeError(
                f"Numerical wall iteration {iteration} failed: "
                f"KSP={info['ksp_reason']}, residual={info['rnorm']}, reference={info['rnorm0']}"
            )
        if iteration < updates:
            step["boundary_updates"] = {name: wall.update() for name, wall in walls.items()}
        uw.pprint(
            f"Numerical wall iteration {iteration}: "
            f"velocity change={step['velocity_change_maximum']:.3e}"
        )
    return history

def _zero_torque_density(edges, nodes, density):
    """Remove the M-weighted rigid-rotation load, preserving radial nodal loads.

    For affine P2 line traces M*w=diag(int(N_i))*w for w=(-y,x).
    Subtracting this linear density therefore changes only tangential loads.
    The radial sheet body force and true free-slip walls exert zero torque.
    """
    edge_nodes = np.stack((edges[:, 0], edges[:, 1], edges.mean(axis=1)), axis=1)
    distance, index = cKDTree(nodes).query(edge_nodes)
    if distance.max() > 1e-10:
        raise ValueError("Boundary density nodes do not match P2 edge nodes.")
    length = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    weights = np.zeros(len(nodes))
    np.add.at(weights, index.ravel(), (length[:, None] * np.array([1 / 6, 1 / 6, 2 / 3])).ravel())
    rotation = np.column_stack((-nodes[:, 1], nodes[:, 0]))
    torque = float(np.sum(weights[:, None] * rotation * density))
    coefficient = torque / np.sum(weights * np.sum(rotation**2, axis=1))
    balanced = density - coefficient * rotation
    return balanced, dict(
        raw_boundary_torque=torque,
        removed_rotation_density_coefficient=float(coefficient),
        corrected_boundary_torque=float(np.sum(weights[:, None] * rotation * balanced)),
    )

def _numerical_trace(
    vertices, velocity, pressure, points, viscosity, sheet_radius, outer_radius, mode, amplitude
):
    """Evaluate physical stress and radial derivatives from affine-cell FE data.

    Velocity nodes are vertices 012 then edge midpoints 01,12,20; pressure nodes
    are the three vertices. Spatial derivatives are Cartesian and one-sided in
    each boundary-adjacent triangle. Radial differentiation holds angle fixed.
    """
    inverse = np.linalg.inv(
        np.stack((vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=-1)
    )
    xi = np.einsum("eij,eqj->eqi", inverse, points - vertices[:, None, 0])
    bary = np.concatenate((1 - xi.sum(axis=-1, keepdims=True), xi), axis=-1)
    grad = np.stack((-inverse.sum(axis=1), inverse[:, 0], inverse[:, 1]), axis=1)
    shapes_grad = np.stack(
        [(4 * bary[..., i, None] - 1) * grad[:, None, i] for i in range(3)]
        + [
            4 * (bary[..., i, None] * grad[:, None, j] + bary[..., j, None] * grad[:, None, i])
            for i, j in ((0, 1), (1, 2), (2, 0))
        ],
        axis=2,
    )
    hessian = np.stack(
        [4 * np.einsum("ei,ej->eij", grad[:, i], grad[:, i]) for i in range(3)]
        + [
            4
            * (
                np.einsum("ei,ej->eij", grad[:, i], grad[:, j])
                + np.einsum("ei,ej->eij", grad[:, j], grad[:, i])
            )
            for i, j in ((0, 1), (1, 2), (2, 0))
        ],
        axis=1,
    )
    velocity_grad = np.einsum("eni,eqnj->eqij", velocity, shapes_grad)
    velocity_hessian = np.einsum("eni,enjk->eijk", velocity, hessian)
    radius = np.linalg.norm(points, axis=-1)
    radial = points / radius[..., None]
    tangent = np.stack((-radial[..., 1], radial[..., 0]), axis=-1)
    lifting = np.zeros_like(radius)
    lifting_grad = np.zeros_like(points)
    if amplitude != 0:
        lifting, lifting_grad = _pressure_jump_values(
            points, sheet_radius, outer_radius, mode, amplitude)
    physical_pressure = np.einsum("en,eqn->eq", pressure, bary) + lifting
    pressure_grad = np.einsum("en,enj->ej", pressure, grad)[:, None, :] + lifting_grad
    derivative = np.einsum("eqi,eqij,eqj->eq", radial, velocity_grad, radial)
    rr = 2 * viscosity * derivative - physical_pressure
    tt = (
        2 * viscosity * np.einsum("eqi,eqij,eqj->eq", tangent, velocity_grad, tangent)
        - physical_pressure
    )
    rt = viscosity * (
        np.einsum("eqi,eqij,eqj->eq", radial, velocity_grad, tangent)
        + np.einsum("eqi,eqij,eqj->eq", tangent, velocity_grad, radial)
    )
    rt_derivative = viscosity * (
        np.einsum("eqi,eijk,eqj,eqk->eq", radial, velocity_hessian, tangent, radial)
        + np.einsum("eqi,eijk,eqj,eqk->eq", tangent, velocity_hessian, radial, radial)
    )
    rr_derivative = 2 * viscosity * np.einsum(
        "eqi,eijk,eqj,eqk->eq", radial, velocity_hessian, radial, radial
    ) - np.sum(radial * pressure_grad, axis=-1)
    return dict(
        radial_velocity_derivative=derivative,
        rr=rr,
        tt=tt,
        rt=rt,
        rt_derivative=rt_derivative,
        rr_derivative=rr_derivative,
    )

class _NumericalWall:
    """Cache one wall's numerical boundary traces and lagged Taylor data.

    First-order transfer follows the Taylor identity in Atallah, Canuto and
    Scovazzi (2020), Section 2.2; see the full reference in this module.
    Strong normal constraints and compatible tangential loads are UW3's
    formulation, with no claim to inherit their Nitsche stability analysis.
    """

    def __init__(
        self,
        mesh,
        velocity,
        pressure,
        radius,
        sign,
        viscosity,
        sheet_radius,
        outer_radius,
        mode,
        amplitude,
        name,
        quadrature,
    ):
        if mesh.dim != 2 or mesh.cdim != 2 or velocity.degree != 2 or not velocity.continuous:
            raise ValueError("Wall transfer requires a two-dimensional continuous affine P2 velocity.")
        if pressure.degree != 1 or not pressure.continuous:
            raise ValueError("Wall transfer requires continuous affine P1 regular pressure.")
        if sign not in (-1, 1) or radius <= 0 or viscosity <= 0 or quadrature < 2:
            raise ValueError("Require positive radius/viscosity, sign +/-1 and quadrature >= 2.")
        self.mesh, self.velocity, self.pressure = mesh, velocity, pressure
        self.radius, self.sign, self.quadrature = radius, sign, quadrature
        self.name = name
        self.physics = (viscosity, sheet_radius, outer_radius, mode, amplitude)
        vertex_rows = mesh._cell_node_indices(1, True).reshape(-1, 3)
        vertices = np.asarray(mesh._get_coords_for_basis(1, True))[vertex_rows]
        owned = mesh._get_owned_cells_mask()
        cells, edges = [], []
        for first, second in ((0, 1), (1, 2), (2, 0)):
            chosen = np.flatnonzero(
                owned
                & np.all(
                    np.isclose(
                        np.linalg.norm(vertices[:, [first, second]], axis=2),
                        radius,
                        rtol=0,
                        atol=1e-10,
                    ),
                    axis=1,
                )
            )
            cells.extend(chosen)
            edges.extend(vertices[chosen][:, [first, second]])
        self.cells = np.array(cells, dtype=int)
        self.p_rows = vertex_rows[self.cells]
        self.v_rows = mesh._cell_node_indices(2, True).reshape(-1, 6)[self.cells]
        local_vertices = vertices[self.cells]
        standard = np.concatenate(
            (local_vertices, (local_vertices[:, [0, 1, 2]] + local_vertices[:, [1, 2, 0]]) / 2),
            axis=1,
        )
        distance = np.linalg.norm(
            np.asarray(velocity.coords)[self.v_rows][:, :, None, :] - standard[:, None, :, :],
            axis=-1,
        )
        self.v_order = distance.argmin(axis=1)
        if distance.size and np.max(distance.min(axis=1)) > 1e-10:
            raise ValueError("Wall transfer requires affine P2 velocity nodes.")
        np.testing.assert_allclose(
            np.asarray(pressure.coords)[self.p_rows], local_vertices, rtol=0, atol=1e-10
        )
        self.vertices = np.concatenate(uw.mpi.comm.allgather(local_vertices))
        self.edges = np.concatenate(uw.mpi.comm.allgather(np.asarray(edges).reshape(-1, 2, 2)))
        self.rule = _trace_rule(self.edges, sign, mode, quadrature)
        self.edge_nodes = np.stack(
            (self.edges[:, 0], self.edges[:, 1], self.edges.mean(axis=1)), axis=1
        )
        self.nodes, self.node_index = np.unique(
            self.edge_nodes.reshape(-1, 2), axis=0, return_inverse=True
        )
        self.node_index = self.node_index.reshape(-1, 3)
        self.normal_field = uw.discretisation.MeshVariable(
            "NormalShift" + name, mesh, 1, degree=2, continuous=True
        )
        self.traction_field = uw.discretisation.MeshVariable(
            "TangentShift" + name, mesh, 2, degree=2, continuous=True
        )
        np.testing.assert_array_equal(self.normal_field.coords, self.traction_field.coords)
        distance, index = cKDTree(self.nodes).query(np.asarray(self.normal_field.coords))
        self.local_boundary, self.local_index = distance < 1e-10, index
        self.normal_values = np.zeros(len(self.nodes))
        self.density = np.zeros_like(self.nodes)

    def cell_values(self):
        """Gather only owned wall-cell traces in the cached geometry order."""
        local_velocity = np.take_along_axis(
            np.asarray(self.velocity.array)[:, 0, :][self.v_rows], self.v_order[..., None], axis=1
        )
        velocity = np.concatenate(uw.mpi.comm.allgather(local_velocity))
        pressure = np.concatenate(
            uw.mpi.comm.allgather(np.asarray(self.pressure.array)[self.p_rows, 0, 0])
        )
        return velocity, pressure

    def update(self):
        """Rebuild Taylor data and project normal flux and tangential torque.

        The first-order boundary shift is motivated by Atallah, Canuto and
        Scovazzi (2020), Section 2.2, equations (8)-(13). Nodal load projection,
        zero-flux/zero-torque corrections and lagged solves are UW3 additions.
        """
        velocity, pressure = self.cell_values()
        points = self.rule["points"]
        trace = _numerical_trace(self.vertices, velocity, pressure, points, *self.physics)
        radius = np.linalg.norm(points, axis=-1)
        radial = points / radius[..., None]
        tangent = np.stack((-radial[..., 1], radial[..., 0]), axis=-1)
        shear = -(self.radius - radius) * trace["rt_derivative"]
        nr = np.sum(self.rule["normal"][:, None, :] * radial, axis=-1)
        nt = np.sum(self.rule["normal"][:, None, :] * tangent, axis=-1)
        traction = (trace["rr"] * nr + shear * nt)[..., None] * radial + (
            shear * nr + trace["tt"] * nt
        )[..., None] * tangent
        nodes, density, checks = _tangent_load_density(
            self.edges, traction, self.sign, self.quadrature
        )
        np.testing.assert_array_equal(nodes, self.nodes)
        density, torque_checks = _zero_torque_density(self.edges, nodes, density)
        checks.update(torque_checks)
        node_trace = _numerical_trace(
            self.vertices, velocity, pressure, self.edge_nodes, *self.physics
        )
        datum = (
            -self.sign
            * (self.radius - np.linalg.norm(self.edge_nodes, axis=-1))
            * node_trace["radial_velocity_derivative"]
        )
        length = np.linalg.norm(self.edges[:, 1] - self.edges[:, 0], axis=1)
        weights = np.zeros(len(nodes))
        normal = np.zeros(len(nodes))
        np.add.at(normal, self.node_index.ravel(), (length[:, None] * datum).ravel())
        np.add.at(
            weights, self.node_index.ravel(), np.broadcast_to(length[:, None], datum.shape).ravel()
        )
        normal /= weights
        # Integrate I_h(g_i*n_i).n_h exactly, rather than averaging nodal values.
        t = (np.polynomial.legendre.leggauss(self.quadrature)[0] + 1) / 2
        shape = np.array([(1 - t) * (1 - 2 * t), t * (2 * t - 1), 4 * t * (1 - t)]).T
        normals = self.sign * self.edge_nodes / np.linalg.norm(self.edge_nodes, axis=-1)[..., None]
        flux_weight = np.einsum(
            "eq,qi,eid,ed->ei", self.rule["weight"], shape, normals, self.rule["normal"]
        )
        raw_flux = np.sum(flux_weight * normal[self.node_index])
        removed_mean = raw_flux / np.sum(flux_weight)
        normal -= removed_mean
        checks.update(
            normal_datum_change=float(np.max(np.abs(normal - self.normal_values))),
            traction_density_change=float(np.max(np.abs(density - self.density))),
            normal_datum_maximum=float(np.max(np.abs(normal))),
            raw_normal_flux=float(raw_flux),
            removed_normal_datum_mean=float(removed_mean),
            corrected_normal_flux=float(np.sum(flux_weight * normal[self.node_index])),
        )
        self.normal_values, self.density = normal, density
        selected, index = self.local_boundary, self.local_index[self.local_boundary]
        self.normal_field.array[selected, 0, 0] = normal[index]
        self.traction_field.array[selected, 0, :] = density[index]
        return checks


def _add_cylindrical_wall_transfer(stokes, *, boundary, radius, sign, viscosity,
                                   interface_radius, outer_radius, wavenumber,
                                   amplitude, quadrature):
    wall = _NumericalWall(stokes.mesh, stokes.u, stokes.p, radius, sign,
                          viscosity, interface_radius, outer_radius, wavenumber,
                          amplitude, boundary, quadrature)
    radial = stokes.mesh.CoordinateSystem.unit_e_0
    stokes.add_rotated_freeslip_bc(wall.normal_field.sym[0], boundary,
                                  normal=sign * radial)
    stokes.add_natural_bc(-wall.traction_field.sym, boundary)
    return wall
