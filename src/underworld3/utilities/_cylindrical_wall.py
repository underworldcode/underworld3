r"""Lagged free-slip transfer from affine P2 chords to physical circles.

Only owned boundary-cell coefficients and one-dimensional boundary mass data
are replicated. The Stokes operator remains distributed PETSc; bounded 1-D
SciPy mass solves preserve P2 nodal load moments. No analytical Stokes state,
harmonic fitting, traction recovery or topography enters this method.

Atallah, Canuto & Scovazzi (2020), CMAME 372, 113341,
https://doi.org/10.1016/j.cma.2020.113341, Section 2.2, equations (8)--(13),
motivate Taylor boundary transfer. Our lagged strong normal constraints,
projected tangential loads and flux/torque compatibility are adaptations,
not their variational Nitsche scheme. No stability/convergence theorem or
fixed iteration count is inferred from that paper. Inspect the physical-circle
normal-velocity and shear-stress residuals in the returned solve history.
"""

import numpy as np
import sympy as sp
import underworld3 as uw
from mpi4py import MPI
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from ._cylindrical_dirichlet import _CircularWall, _p2_velocity, _facet_normals


def _tangent_load_density(edges: np.ndarray, traction: np.ndarray, sign: int, quadrature: int):
    """Project integrated loads tangentially at P2 nodes; preserve their moments.

    Project loads before inverting the mass matrix. Projecting pointwise density
    instead would generally inject normal nodal loads into the reaction.
    """
    _, weights = np.polynomial.legendre.leggauss(quadrature)
    rule = {"weight": np.linalg.norm(edges[:, 1]-edges[:, 0], axis=1)[:, None]*weights/2}
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
    vertices, velocity, pressure, points, viscosity, pressure_jump_values
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
    physical_pressure = np.einsum("en,eqn->eq", pressure, bary) + pressure_jump_values
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
    return dict(
        radial_velocity_derivative=derivative,
        rr=rr,
        tt=tt,
        rt=rt,
        rt_derivative=rt_derivative,
    )


def _p1_values(nodes, values, points):
    """Continue each scalar affine cell polynomial to the requested points."""
    design = np.concatenate((np.ones((*nodes.shape[:2], 1)), nodes), axis=-1)
    coefficients = np.linalg.solve(design, values[..., None])[..., 0]
    return coefficients[:, None, 0] + np.einsum("eqi,ei->eq", points, coefficients[:, 1:])


class _FreeSlipWall(_CircularWall):
    """A rotated normal datum and tangential natural datum on a fixed circle."""

    def __init__(self, stokes, boundary, radius, sign, viscosity, pressure_jump, quadrature):
        super().__init__(stokes, boundary, radius, sign, quadrature)
        if not np.isfinite(viscosity) or viscosity <= 0:
            raise ValueError("Circular free-slip transfer requires positive constant viscosity.")
        self.viscosity = float(viscosity)
        if isinstance(pressure_jump, uw.discretisation.MeshVariable):
            lift = pressure_jump
        else:
            lift = uw.function.expressions._unwrap_for_compilation(
                sp.sympify(pressure_jump), keep_constants=False, return_self=False)
            variables = [atom for atom in lift.atoms(sp.Function) if hasattr(type(atom), "meshvar")]
            if variables:
                variable = type(variables[0]).meshvar()
                if len(variables) != 1 or variable.num_components != 1 or lift != variable.sym[0]:
                    raise ValueError("Pass the prescribed scalar pressure-lift field or a coordinate expression.")
                lift = variable
        self.pressure_jump = lift
        if isinstance(lift, uw.discretisation.MeshVariable):
            if lift.mesh is not self.mesh or lift.degree != 1 or lift.num_components != 1:
                raise ValueError("Pressure-lift fields must be scalar P1 fields on the Stokes mesh.")
            self.lift_rows = self.mesh._cell_node_indices(1, lift.continuous).reshape(-1, 3)[self.cells]
        prefix = f"{stokes.instance_number}_{boundary}"
        self.normal_field = uw.discretisation.MeshVariable(
            "CircularNormalShift_" + prefix, self.mesh, 1, degree=2, continuous=True)
        self.traction_field = uw.discretisation.MeshVariable(
            "CircularTangentShift_" + prefix, self.mesh, 2, degree=2, continuous=True)
        self.normal_values = np.zeros(len(self.nodes))
        self.density = np.zeros_like(self.nodes)
        self.rule["normal"] = _facet_normals(self.edges, sign)
        _, weights = np.polynomial.legendre.leggauss(quadrature)
        self.rule["weight"] = np.linalg.norm(self.edges[:, 1]-self.edges[:, 0], axis=1)[:, None]*weights/2

    def pressure_values(self):
        """Gather regular pressure and one-sided prescribed lift at wall points."""
        pressure = np.concatenate(uw.mpi.comm.allgather(
            np.asarray(self.stokes.p.array)[self.p_rows, 0, 0]))
        points = self.rule["points"][self.local_edge_slice]
        if isinstance(self.pressure_jump, uw.discretisation.MeshVariable):
            field, rows = self.pressure_jump, self.lift_rows
            lift = _p1_values(np.asarray(field.coords)[rows], np.asarray(field.array)[rows, 0, 0], points)
        else:
            # Coordinate-only evaluation is rank-local; every rank also enters
            # the empty-point case. FE lifts use cell-side coefficients above.
            lift = np.asarray(uw.function.evaluate(self.pressure_jump,
                np.ascontiguousarray(points.reshape(-1, 2)), coord_sys=self.mesh.N)).reshape(points.shape[:2])
        return pressure, np.concatenate(uw.mpi.comm.allgather(lift))

    def update(self):
        """Transfer physical normal/shear conditions and enforce compatibility."""
        velocity = self.cell_values()
        pressure, lift = self.pressure_values()
        points = self.rule["points"]
        trace = _numerical_trace(self.vertices, velocity, pressure, points, self.viscosity, lift)
        radius = np.linalg.norm(points, axis=-1)
        radial = points/radius[..., None]
        tangent = np.stack((-radial[..., 1], radial[..., 0]), axis=-1)
        shear = -(self.radius-radius)*trace["rt_derivative"]
        nr = np.sum(self.rule["normal"][:, None]*radial, axis=-1)
        nt = np.sum(self.rule["normal"][:, None]*tangent, axis=-1)
        traction = (trace["rr"]*nr+shear*nt)[..., None]*radial + (
            shear*nr+trace["tt"]*nt)[..., None]*tangent
        nodes, density, checks = _tangent_load_density(self.edges, traction, self.sign, self.quadrature)
        density, torque = _zero_torque_density(self.edges, nodes, density)
        checks.update(torque)
        node_trace = _numerical_trace(self.vertices, velocity, pressure, self.edge_nodes,
                                      self.viscosity, np.zeros(self.edge_nodes.shape[:2]))
        datum = -self.sign*(self.radius-np.linalg.norm(self.edge_nodes, axis=-1))*node_trace["radial_velocity_derivative"]
        length = np.linalg.norm(self.edges[:, 1]-self.edges[:, 0], axis=1)
        weights, normal = np.zeros(len(nodes)), np.zeros(len(nodes))
        np.add.at(normal, self.node_index.ravel(), (length[:, None]*datum).ravel())
        np.add.at(weights, self.node_index.ravel(), np.broadcast_to(length[:, None], datum.shape).ravel())
        normal /= weights
        t = (np.polynomial.legendre.leggauss(self.quadrature)[0]+1)/2
        shape = np.array([(1-t)*(1-2*t), t*(2*t-1), 4*t*(1-t)]).T
        normals = self.sign*self.edge_nodes/np.linalg.norm(self.edge_nodes, axis=-1)[..., None]
        flux_weight = np.einsum("eq,qi,eid,ed->ei", self.rule["weight"], shape, normals, self.rule["normal"])
        raw_flux = np.sum(flux_weight*normal[self.node_index])
        removed = raw_flux/np.sum(flux_weight)
        normal -= removed
        checks.update(normal_datum_change=float(np.max(np.abs(normal-self.normal_values))),
            traction_density_change=float(np.max(np.abs(density-self.density))),
            normal_datum_maximum=float(np.max(np.abs(normal))), raw_normal_flux=float(raw_flux),
            removed_normal_datum_mean=float(removed),
            corrected_normal_flux=float(np.sum(flux_weight*normal[self.node_index])) )
        self.normal_values, self.density = normal, density
        selected, index = self.local_boundary, self.local_index[self.local_boundary]
        self.normal_field.array[selected, 0, 0] = normal[index]
        self.traction_field.array[selected, 0, :] = density[index]
        return checks

    def audit(self):
        """Report physical-circle normal velocity/shear, without analytical data."""
        velocity = self.cell_values()
        circle = self.rule["circle_points"]
        values = _p2_velocity(self.vertices, velocity, circle)
        normal = self.sign*circle/self.radius
        un = np.sum(values*normal, axis=-1)
        trace = _numerical_trace(self.vertices, velocity, np.zeros((len(self.edges), 3)),
                                 circle, self.viscosity, np.zeros(circle.shape[:2]))
        weight = self.rule["circle_weight"]
        return dict(true_circle_normal_velocity_rms=float(np.sqrt(np.sum(weight*un**2)/weight.sum())),
                    true_circle_shear_stress_rms=float(np.sqrt(np.sum(weight*trace["rt"]**2)/weight.sum())))


def _add_cylindrical_wall_transfer(stokes, *, boundary, radius, sign, viscosity,
                                   pressure_jump, quadrature):
    wall = _FreeSlipWall(stokes, boundary, radius, sign, viscosity, pressure_jump, quadrature)
    stokes.add_rotated_freeslip_bc(wall.normal_field.sym[0], boundary,
                                  normal=sign*stokes.mesh.CoordinateSystem.unit_e_0)
    stokes.add_natural_bc(-wall.traction_field.sym, boundary)
    return wall


def _solve_with_wall_updates(stokes, walls, updates):
    """Record accepted solves and physical residuals for a lagged wall iteration."""
    if int(updates) != updates or updates < 0:
        raise ValueError("Wall updates must be a non-negative integer.")
    if not stokes.petsc_use_pressure_nullspace:
        raise ValueError("Wall updates require petsc_use_pressure_nullspace=True.")
    if (len(walls) != 2 or {wall.sign for wall in walls.values()} != {-1, 1}
            or any(wall.stokes is not stokes for wall in walls.values())):
        raise ValueError("Supply this Stokes system's inner and outer free-slip handles together.")
    radii = {wall.sign: wall.radius for wall in walls.values()}
    if not radii[-1] < radii[1]:
        raise ValueError("The inner free-slip radius must be smaller than the outer radius.")
    history = []
    stokes.cylindrical_wall_history = history
    for iteration in range(int(updates)+1):
        previous = np.asarray(stokes.u.array).copy()
        stokes.solve()
        info = stokes._rotated_freeslip_info
        step = dict(iteration=iteration, converged=bool(info["converged"]),
            ksp_reason=int(info["ksp_reason"]), ksp_iterations=list(info["ksp_its"]),
            linear_solver={key: info[key] for key in (
                "velocity_ksp_type", "pressure_ksp_type", "ksp_max_it")},
            nonlinear_iterations=int(info["nonlinear_iterations"]), residual_norm=float(info["rnorm"]),
            velocity_change_maximum=uw.mpi.comm.allreduce(
                float(np.max(np.abs(stokes.u.array-previous), initial=0.0)), op=MPI.MAX),
            pressure_gauge_residual=float(stokes._rotated_pressure_gauge_residual),
            boundary_velocity={name: wall.audit() for name, wall in sorted(walls.items())},
            boundary_updates={})
        history.append(step)
        if not (np.isfinite(info["rnorm"]) and info["converged"] and (info["ksp_reason"] > 0 or
                (info["ksp_reason"] == 0 and info["nonlinear_iterations"] == 0))):
            raise RuntimeError(f"Circular wall solve {iteration} failed: {info['ksp_reason']}")
        if iteration < updates:
            step["boundary_updates"] = {name: wall.update() for name, wall in sorted(walls.items())}
    return history
