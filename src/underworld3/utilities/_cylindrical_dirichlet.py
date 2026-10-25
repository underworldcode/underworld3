"""Lagged zero-slip transfer from affine P2 chords to true circular walls.

Only owned boundary-cell velocity coefficients are replicated. The essential
datum is a separate P2 vector field, registered once and refreshed before each
standard Stokes solve. No pressure, analytical Stokes solution, harmonic fit or
traction recovery enters this correction. Coordinates and fields must be
nondimensional; the mesh must remain fixed after registration.

The shifted Dirichlet relation is motivated by Atallah, Canuto & Scovazzi
(2020), CMAME 372, 113341, https://doi.org/10.1016/j.cma.2020.113341,
Section 2.2, equations (8)-(13), and its Section 4 Stokes formulation;
see also Burman, Hansbo & Larson (2018),
"A cut finite element method with boundary value correction for the
incompressible Stokes' equations", https://arxiv.org/abs/1801.07463.
Here the complete affine-cell P2 polynomial is continued radially and its
boundary defect is used in a lagged strong essential condition. This differs
from those papers' variational Nitsche formulations; their stability and
convergence results do not establish convergence of this fixed-point scheme.
The joint zero-flux projection below is an additional compatibility condition.

A closer trial/test-space precedent is Ruas (2019), "Accuracy enhancement for
non-isoparametric finite-element simulations in curved domains; application to
fluid flow", Computers & Mathematics with Applications 77(6), 1756-1769,
https://doi.org/10.1016/j.camwa.2018.05.029, Sections 2/4, equations (29)-(30).
Its P2 velocity trials obey true-boundary point constraints; tests retain zero
polygon traces, with continuous P1 pressure. UW3's symmetric-stress form and
joint flux projection differ. A nonzero projection coefficient perturbs the
true-circle nodal constraint, so the physical residual must be reported. The
paper's regularity and stability results are not claimed for this adaptation.
"""

import numpy as np
import underworld3 as uw
from mpi4py import MPI
from scipy.spatial import cKDTree

from .boundary_flux import _boundary_stratum_is


def _edge_key(edge, tolerance):
    """Coordinate key independent of facet orientation and partition numbering."""
    return tuple(sorted(tuple(row) for row in np.rint(edge / tolerance).astype(np.int64)))


def _labelled_edge_keys(mesh, boundary, radius, tolerance):
    """Collect only labelled boundary facets, including ghosts on empty ranks."""
    dm = mesh.dm
    facets = _boundary_stratum_is(dm, mesh, boundary)
    indices = np.empty(0, dtype=int)
    if facets is not None:
        indices = np.asarray(facets.indices).copy()
        facets.destroy()
    f0, f1 = dm.getDepthStratum(1)
    v0, v1 = dm.getDepthStratum(0)
    section = dm.getCoordinateSection()
    coordinates = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
    local, unsupported = [], False
    for facet in indices:
        if f0 <= facet < f1:
            closure = dm.getTransitiveClosure(int(facet))[0]
            vertices = [int(point) for point in closure if v0 <= point < v1]
            if len(vertices) != 2:
                unsupported = True
            else:
                local.append([coordinates[section.getOffset(point) // 2] for point in vertices])
    if uw.mpi.comm.allreduce(unsupported, op=MPI.LOR):
        raise ValueError("Zero-slip transfer supports two-vertex affine boundary facets only.")
    edges = np.concatenate(uw.mpi.comm.allgather(np.asarray(local).reshape(-1, 2, 2)))
    if not len(edges) or not np.allclose(np.linalg.norm(edges, axis=-1), radius,
                                          rtol=0, atol=tolerance):
        raise ValueError("The labelled boundary must lie on the requested centered circle.")
    return {_edge_key(edge, tolerance) for edge in edges}


def _p2_velocity(vertices, velocity, points):
    """Evaluate Cartesian P2 cell polynomials, including one-sided continuation.

    Array shapes are ``(cells,3,2)``, ``(cells,6,2)`` and ``(cells,points,2)``.
    Velocity coefficients are ordered vertices 012, then midpoints 01,12,20.
    No radial-frame transformation or angular averaging is applied.
    """
    inverse = np.linalg.inv(np.stack(
        (vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=-1))
    xi = np.einsum("eij,eqj->eqi", inverse, points - vertices[:, None, 0])
    bary = np.concatenate((1 - xi.sum(axis=-1, keepdims=True), xi), axis=-1)
    shape = np.stack(
        [bary[..., i] * (2 * bary[..., i] - 1) for i in range(3)]
        + [4 * bary[..., i] * bary[..., j] for i, j in ((0, 1), (1, 2), (2, 0))],
        axis=-1)
    return np.einsum("eqn,end->eqd", shape, velocity)


def _zero_slip_datum(vertices, velocity, edge_nodes, radius):
    r"""Return ``u_h(x)-u_h(R*x/|x|)`` at endpoint, endpoint, midpoint nodes.

    The prescribed physical velocity is zero. For an affine P2 polynomial this
    is the complete quadratic Taylor shift, including its second derivative.
    The displacement is radially outward at both walls; outward-domain normal
    signs belong only to the flux calculation, never to this vector identity.
    """
    radii = np.linalg.norm(edge_nodes, axis=-1)
    true_points = radius * edge_nodes / radii[..., None]
    datum = (_p2_velocity(vertices, velocity, edge_nodes)
             - _p2_velocity(vertices, velocity, true_points))
    # Circle vertices have exactly zero shift; retain that essential value
    # despite floating-point error in the radial map.
    datum[:, :2] = 0.0
    return datum


def _facet_normals(edges, sign):
    """Use actual chord normals, oriented radially independently of edge order."""
    direction = edges[:, 1] - edges[:, 0]
    normal = np.column_stack((direction[:, 1], -direction[:, 0]))
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    orientation = np.sign(np.sum(normal * edges.mean(axis=1), axis=1))
    return sign * orientation[:, None] * normal


def _p2_facet_flux(edges, values, sign):
    """Integrate the P2 vector datum against the constant polygon outward normal.

    ``values`` has shape ``(edges,3,2)`` in endpoint,endpoint,midpoint order.
    Simpson weights are exact here. ``sign`` is -1 inner and +1 outer.
    """
    normal = _facet_normals(edges, sign)
    length = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    average = np.einsum("i,eid->ed", np.array([1, 1, 4]) / 6, values)
    return float(np.sum(length * np.sum(average * normal, axis=1)))


def _project_joint_flux(walls):
    r"""Remove total polygon flux with the smallest L2 normal-midpoint change.

    Each correction is ``-c*N_mid*n_facet``. Its flux is ``-c*2L/3`` and its
    squared L2 norm is ``c²*8L/15`` on an edge of length L. Consequently one
    coefficient shared by every edge of both walls minimizes the integrated
    L2 change in this bubble span. Vertex values and tangential midpoint data
    are unchanged. Individual wall fluxes are reported, not independently fit.

    Handles provide ``edges,nodes,node_index,values,sign``. Only ``values`` is
    changed, so the mathematical projection is testable without a solver.
    """
    raw, lengths = {}, {}
    for name, wall in sorted(walls.items()):
        raw[name] = _p2_facet_flux(wall.edges, wall.values[wall.node_index], wall.sign)
        lengths[name] = float(np.linalg.norm(wall.edges[:, 1] - wall.edges[:, 0], axis=1).sum())
    total_length = sum(lengths.values())
    if not np.isfinite(total_length) or total_length <= 0:
        raise ValueError("Joint flux projection requires nonempty circular walls.")
    raw_total = sum(raw.values())
    coefficient = raw_total / (2 * total_length / 3)
    corrected = {}
    for name, wall in sorted(walls.items()):
        mid = wall.node_index[:, 2]
        if len(np.unique(mid)) != len(mid):
            raise ValueError("Each circular boundary midpoint must belong to one edge.")
        normal = _facet_normals(wall.edges, wall.sign)
        wall.values[mid] -= coefficient * normal
        corrected[name] = _p2_facet_flux(wall.edges, wall.values[wall.node_index], wall.sign)
    corrected_total = sum(corrected.values())
    tolerance = 128 * np.finfo(float).eps * max(1.0, abs(raw_total))
    if abs(corrected_total) > tolerance:
        raise ArithmeticError("Joint P2 zero-slip flux projection failed.")
    return dict(
        raw_total_flux=float(raw_total), corrected_total_flux=float(corrected_total),
        removed_midpoint_normal_value=float(coefficient),
        l2_correction_norm=float(abs(coefficient) * np.sqrt(8 * total_length / 15)),
        boundaries={name: dict(raw_flux=raw[name], corrected_flux=corrected[name])
                    for name in sorted(walls)},
    )


class _ZeroSlipWall:
    """Fixed-mesh boundary geometry and one lagged vector essential datum."""

    def __init__(self, stokes, boundary, radius, sign, quadrature):
        mesh, velocity = stokes.mesh, stokes.u
        if (mesh.dim != 2 or mesh.cdim != 2 or not mesh.isSimplex
                or velocity.degree != 2 or not velocity.continuous):
            raise ValueError("Zero-slip transfer requires a two-dimensional continuous affine P2 velocity.")
        if stokes.p.degree != 1 or not stokes.p.continuous:
            raise ValueError("Zero-slip transfer requires continuous affine P1 regular pressure.")
        if uw.is_nondimensional_scaling_active() or uw.get_default_model().has_units_active():
            raise ValueError("Zero-slip transfer supports nondimensional coordinates and fields only.")
        if (not np.isfinite(radius) or radius <= 0 or sign not in (-1, 1)
                or int(quadrature) != quadrature or quadrature < 3):
            raise ValueError("Require finite positive radius, sign +/-1 and integer quadrature >= 3.")
        self.stokes, self.mesh, self.velocity = stokes, mesh, velocity
        self.name, self.radius, self.sign = boundary, float(radius), sign
        self.quadrature = int(quadrature)
        tolerance = 1e-10 * radius
        boundary_edges = _labelled_edge_keys(mesh, boundary, radius, tolerance)
        vertex_rows = mesh._cell_node_indices(1, True).reshape(-1, 3)
        vertices = np.asarray(mesh._get_coords_for_basis(1, True))[vertex_rows]
        owned = mesh._get_owned_cells_mask()
        cells, edges = [], []
        for first, second in ((0, 1), (1, 2), (2, 0)):
            chosen = np.flatnonzero(owned & np.all(np.isclose(
                np.linalg.norm(vertices[:, [first, second]], axis=2), radius,
                rtol=0, atol=tolerance), axis=1))
            for cell in chosen:
                edge = vertices[cell, [first, second]]
                if _edge_key(edge, tolerance) in boundary_edges:
                    cells.append(cell)
                    edges.append(edge)
        self.cells = np.asarray(cells, dtype=int)
        self.v_rows = mesh._cell_node_indices(2, True).reshape(-1, 6)[self.cells]
        local_vertices = vertices[self.cells]
        standard = np.concatenate((local_vertices,
            (local_vertices[:, [0, 1, 2]] + local_vertices[:, [1, 2, 0]]) / 2), axis=1)
        distance = np.linalg.norm(np.asarray(velocity.coords)[self.v_rows][:, :, None]
                                  - standard[:, None], axis=-1)
        self.v_order = distance.argmin(axis=1)
        error = float(np.max(distance.min(axis=1), initial=0.0))
        if uw.mpi.comm.allreduce(error, op=MPI.MAX) > tolerance:
            raise ValueError("Zero-slip transfer requires straight-sided affine P2 cells.")
        self.vertices = np.concatenate(uw.mpi.comm.allgather(local_vertices))
        self.edges = np.concatenate(uw.mpi.comm.allgather(np.asarray(edges).reshape(-1, 2, 2)))
        if not len(self.edges):
            raise ValueError("No owned facets lie on the requested circular wall.")
        if {_edge_key(edge, tolerance) for edge in self.edges} != boundary_edges:
            raise ValueError("Owned boundary-cell traces do not cover the labelled circular wall.")
        if np.any(np.linalg.norm(self.edges.mean(axis=1), axis=1) <= tolerance):
            raise ValueError("Circular boundary edges must span less than a semicircle.")
        self.edge_nodes = np.stack((self.edges[:, 0], self.edges[:, 1], self.edges.mean(axis=1)), axis=1)
        self.nodes, index = np.unique(self.edge_nodes.reshape(-1, 2), axis=0, return_inverse=True)
        self.node_index = index.reshape(-1, 3)
        # An exterior closed wall has one owned cell per edge and two edges at
        # each vertex. Reject internal circles and incomplete circular arcs.
        _, counts = np.unique(np.sort(self.node_index[:, :2], axis=1), axis=0, return_counts=True)
        _, incidence = np.unique(self.node_index[:, :2], return_counts=True)
        if np.any(counts != 1) or np.any(incidence != 2):
            raise ValueError("Transfer requires a complete exterior circular wall, not an interface or arc.")
        self.field = uw.discretisation.MeshVariable(
            "ZeroSlipShift" + boundary, mesh, 2, degree=2, continuous=True)
        distance, index = cKDTree(self.nodes).query(np.asarray(self.field.coords))
        self.local_boundary, self.local_index = distance < tolerance, index
        self.values = np.zeros_like(self.nodes)
        t, weight = np.polynomial.legendre.leggauss(self.quadrature)
        t, weight = (t + 1) / 2, weight / 2
        points = ((1 - t)[None, :, None] * self.edges[:, None, 0]
                  + t[None, :, None] * self.edges[:, None, 1])
        direction = self.edges[:, 1] - self.edges[:, 0]
        cross = np.abs(points[..., 0] * direction[:, None, 1]
                       - points[..., 1] * direction[:, None, 0])
        self.rule = dict(points=points,
            circle_points=radius * points / np.linalg.norm(points, axis=-1)[..., None],
            circle_weight=radius * cross / np.sum(points**2, axis=-1) * weight)

    def cell_values(self):
        """Collect only owned wall-cell P2 coefficients, including empty ranks."""
        local = np.take_along_axis(
            np.asarray(self.velocity.array)[:, 0, :][self.v_rows],
            self.v_order[..., None], axis=1)
        return np.concatenate(uw.mpi.comm.allgather(local))

    def update(self):
        """Replace raw lagged datum; joint flux projection must follow both walls."""
        datum = _zero_slip_datum(self.vertices, self.cell_values(), self.edge_nodes, self.radius)
        length = np.linalg.norm(self.edges[:, 1] - self.edges[:, 0], axis=1)
        value, weight = np.zeros_like(self.values), np.zeros(len(self.nodes))
        np.add.at(value, self.node_index.ravel(), (length[:, None, None] * datum).reshape(-1, 2))
        np.add.at(weight, self.node_index.ravel(), np.broadcast_to(length[:, None], datum.shape[:2]).ravel())
        value /= weight[:, None]
        change = float(np.max(np.linalg.norm(value - self.values, axis=1)))
        self.values = value
        return dict(datum_change_maximum=change,
                    datum_maximum=float(np.max(np.linalg.norm(value, axis=1))))

    def write(self):
        """Write the current compatible datum; the registered expression is fixed."""
        selected = self.local_boundary
        self.field.array[selected, 0, :] = self.values[self.local_index[selected]]

    def audit(self):
        """Measure the solved polynomial's zero-slip defect on the true circle.

        Full Gauss RMS uses the radial-map circular arclength Jacobian. Nodal
        maximum and the polygon essential residual are independent diagnostics;
        no pressure-gauge adjustment or trace filtering is applied.
        """
        velocity = self.cell_values()
        circle = self.radius * self.edge_nodes / np.linalg.norm(self.edge_nodes, axis=-1)[..., None]
        nodal = _p2_velocity(self.vertices, velocity, circle)
        gauss = _p2_velocity(self.vertices, velocity, self.rule["circle_points"])
        square = float(np.sum(self.rule["circle_weight"] * np.sum(gauss**2, axis=-1)))
        measure = float(self.rule["circle_weight"].sum())
        polygon = _p2_velocity(self.vertices, velocity, self.edge_nodes)
        return dict(
            true_circle_nodal_maximum=float(np.max(np.linalg.norm(nodal, axis=-1))),
            true_circle_gauss_maximum=float(np.max(np.linalg.norm(gauss, axis=-1))),
            true_circle_velocity_l2=float(np.sqrt(square)),
            true_circle_velocity_rms=float(np.sqrt(square / measure)),
            true_circle_measure=measure,
            polygon_essential_residual_maximum=float(np.max(np.linalg.norm(
                polygon - self.values[self.node_index], axis=-1))),
        )


def _add_cylindrical_zero_slip_transfer(stokes, *, boundary, radius, sign, quadrature):
    """Register one auxiliary P2 vector essential datum before the first solve."""
    if getattr(stokes, "_rotated_freeslip_bcs", None):
        raise ValueError("Zero-slip transfer uses standard Stokes, not rotated free slip.")
    wall = _ZeroSlipWall(stokes, boundary, radius, sign, quadrature)
    stokes.add_dirichlet_bc(wall.field.sym, boundary)
    return wall


def _solve_cylindrical_zero_slip_updates(stokes, *, walls, updates):
    """Solve with jointly compatible, lagged full-vector circular-wall data.

    Both inner and outer handles must be supplied. ``updates`` is an explicit
    experimental iteration count, not a convergence guarantee. Each history
    row retains the standard solver verdict and the physical-circle velocity
    defects so callers can assess fixed-point convergence. No rigid-rotation
    removal or torque projection is added here. The caller must enable the
    standard pressure nullspace; this helper does not install another gauge.
    """
    if int(updates) != updates or updates < 0:
        raise ValueError("Zero-slip updates must be a non-negative integer.")
    if not stokes.petsc_use_pressure_nullspace:
        raise ValueError("Zero-slip updates require petsc_use_pressure_nullspace=True.")
    if (len(walls) != 2 or {wall.sign for wall in walls.values()} != {-1, 1}
            or any(wall.stokes is not stokes for wall in walls.values())):
        raise ValueError("Supply this Stokes system's inner and outer zero-slip handles together.")
    radii = {wall.sign: wall.radius for wall in walls.values()}
    if not radii[-1] < radii[1]:
        raise ValueError("The inner zero-slip radius must be smaller than the outer radius.")
    history = []
    stokes.cylindrical_zero_slip_history = history
    for iteration in range(int(updates) + 1):
        flux = _project_joint_flux(walls)
        for _, wall in sorted(walls.items()):
            wall.write()
        previous = np.asarray(stokes.u.array).copy()
        stokes.solve()
        report = stokes.solve_report
        step = dict(iteration=iteration, flux_projection=flux,
            converged=bool(report.converged), reason=int(report.reason),
            reason_string=report.reason_str, nonlinear_iterations=int(report.nl_its),
            ksp_iterations=int(report.ksp_its), residual_norm=float(report.fnorm),
            velocity_change_maximum=uw.mpi.comm.allreduce(
                float(np.max(np.abs(stokes.u.array - previous), initial=0.0)), op=MPI.MAX),
            boundary_velocity={name: wall.audit() for name, wall in sorted(walls.items())},
            boundary_updates={})
        history.append(step)
        if not report.converged or report.bounded or report.deadline_expired or not np.isfinite(report.fnorm):
            raise RuntimeError(f"Zero-slip wall iteration {iteration} failed: {report}")
        if iteration < updates:
            step["boundary_updates"] = {name: wall.update() for name, wall in sorted(walls.items())}
        uw.pprint(f"Zero-slip wall iteration {iteration}: velocity change="
                  f"{step['velocity_change_maximum']:.3e}")
    return history
