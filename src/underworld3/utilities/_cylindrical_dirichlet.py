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

from .boundary_flux import _boundary_stratum_is

# How far, as a fraction of the declared radius, a labelled wall vertex may lie
# from the declared circle before the wall is refused as no longer on it. This is
# the only use of a tolerance here, and it is geometry, not identity: the wall's
# cells, vertices, chords and P2 nodes are identified topologically (the label's
# facets, their support cells, the DM's global point numbering and the P1 and P2
# coordinate sections), never by comparing coordinates.
_WALL_MATCH_RELATIVE_TOLERANCE = 1.0e-10


def _declared_circle_radius(mesh, boundary):
    """The radius of the origin-centred circle the mesh declares for ``boundary``.

    The circle is the mesh's declaration (``mesh.analytic_surface``), never a
    radius from the caller and never a measurement. Refused, with the
    accessor's reason, when the boundary has no declared circle: undeclared,
    released after the mesh was deformed, or declared as a plane. The
    registry is the same on every rank, so the verdict is too, and nothing
    here is collective.
    """
    surface = mesh.analytic_surface(boundary)
    if surface.kind != "radial":
        raise ValueError(
            f"boundary {boundary!r} is declared as a {surface.kind}, not a circle; "
            f"the circular-wall methods need a declared circle")
    if np.any(surface.centre != 0.0):
        raise ValueError(
            f"boundary {boundary!r} is declared as a circle centred at "
            f"{tuple(float(c) for c in surface.centre)}; the circular-wall methods support a circle "
            f"centred at the origin only")
    return float(surface.radius)


def _labelled_facets(dm, mesh, boundary):
    """This rank's labelled facets (edges) of ``boundary``, ghosts included, as
    DM point numbers. Not collective."""
    facets = _boundary_stratum_is(dm, mesh, boundary)
    indices = np.empty(0, dtype=np.int64)
    # PETSc may return a false IS wrapper with a null C handle on empty ranks.
    # Checking only Python None would dereference that null handle at .indices.
    if facets:
        indices = np.asarray(facets.indices, dtype=np.int64).copy()
        facets.destroy()
    f0, f1 = dm.getDepthStratum(1)
    return indices[(indices >= f0) & (indices < f1)]


def _global_point_numbers(dm):
    """The global number of every local DM point, owned or not. The point
    numbering stores ``-(g + 1)`` for a point another rank owns. Not collective
    beyond the numbering's own construction, which every rank enters."""
    numbering = dm.createPointNumbering()
    numbers = np.asarray(numbering.indices, dtype=np.int64).copy()
    numbering.destroy()
    return np.where(numbers >= 0, numbers, -(numbers + 1))


def _wall_off_circle(boundary, edges, radius):
    """The refusal for a labelled wall whose vertices are not on its declared
    circle to the tolerance. ``edges`` are gathered, so the message is the same
    on every rank."""
    deviation = float(np.max(np.abs(np.linalg.norm(edges, axis=-1) - radius))) / radius
    return ValueError(
        f"the labelled vertices of boundary {boundary!r} are off its declared circle "
        f"of radius {radius}: the largest distance is {deviation:.3e} of the radius, "
        f"beyond the tolerance of {_WALL_MATCH_RELATIVE_TOLERANCE:.1e} of the "
        f"radius. A wall that has left its declared circle must be released, not "
        f"used as that circle")


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


class _CircularWall:
    """Owned affine boundary-cell geometry shared by the two wall conditions."""

    def __init__(self, stokes, boundary, radius, sign, quadrature):
        mesh, velocity = stokes.mesh, stokes.u
        if (mesh.dim != 2 or mesh.cdim != 2 or not mesh.isSimplex
                or velocity.degree != 2 or not velocity.continuous):
            raise ValueError("Circular-wall transfer requires a two-dimensional continuous affine P2 velocity.")
        if stokes.p.degree != 1 or not stokes.p.continuous:
            raise ValueError("Circular-wall transfer requires continuous affine P1 regular pressure.")
        if uw.is_nondimensional_scaling_active() or uw.get_default_model().has_units_active():
            raise ValueError("Circular-wall transfer supports nondimensional coordinates and fields only.")
        if (not np.isfinite(radius) or radius <= 0 or sign not in (-1, 1)
                or int(quadrature) != quadrature or quadrature < 3):
            raise ValueError("Require finite positive radius, sign +/-1 and integer quadrature >= 3.")
        self.stokes, self.mesh, self.velocity = stokes, mesh, velocity
        self.name, self.sign = boundary, sign
        self.quadrature = int(quadrature)
        self._build_geometry(radius)

    def _build_geometry(self, radius):
        """Cache the wall's cells, chords, P2 node maps and quadrature rule.

        The wall is identified topologically. Its cells are the owned support
        cells of the labelled facets. A cell's vertices are put in its P1 row
        order and its P2 nodes in the order vertices 012, then edges 01, 12, 20,
        by matching DM points through the P1 and P2 coordinate sections. The
        gathered wall is keyed on the DM's global point numbers: a vertex node by
        its vertex's number, a midpoint node by its edge's. Coordinates enter only
        the off-circle refusal and the geometry (chords, normals, quadrature).

        Everything cached here follows from the mesh coordinates, so it is built
        again when they change (:meth:`refresh_geometry`). Collective.
        """
        mesh, velocity = self.mesh, self.velocity
        dm, cdim = mesh.dm, mesh.cdim
        boundary = self.name
        self.radius = float(radius)
        self._built_at_geometry_version = mesh._geometry_version
        c0, _ = dm.getHeightStratum(0)
        v0, v1 = dm.getDepthStratum(0)
        f0, f1 = dm.getDepthStratum(1)
        facets = _labelled_facets(dm, mesh, boundary)
        cones = [[int(point) for point in dm.getCone(int(facet))] for facet in facets]
        if uw.mpi.comm.allreduce(any(len(cone) != 2 for cone in cones), op=MPI.LOR):
            raise ValueError("Circular-wall transfer supports two-vertex affine boundary facets only.")
        numbers = _global_point_numbers(dm)
        p1_dm, p2_dm = mesh._basis_coordinate_dm(1, True), mesh._basis_coordinate_dm(2, True)
        p1, p2 = p1_dm.getLocalSection(), p2_dm.getLocalSection()
        vertex_rows = mesh._cell_node_indices(1, True).reshape(-1, 3)
        cell_rows = mesh._cell_node_indices(2, True).reshape(-1, 6)
        p1_coordinates = np.asarray(mesh._get_coords_for_basis(1, True))

        labelled = np.concatenate(uw.mpi.comm.allgather(
            p1_coordinates[[[p1.getOffset(v) // cdim for v in cone] for cone in cones]]
            .reshape(-1, 2, 2)))
        if not len(labelled):
            raise ValueError(f"Boundary {boundary!r} holds no facets.")
        deviation = np.max(np.abs(np.linalg.norm(labelled, axis=-1) - radius)) / radius
        if deviation > _WALL_MATCH_RELATIVE_TOLERANCE:
            raise _wall_off_circle(boundary, labelled, radius)

        owned = mesh._get_owned_cells_mask()
        cells, pairs, orders, wall_numbers = [], [], [], []
        for facet, cone in zip(facets, cones):
            for cell in dm.getSupport(int(facet)):
                k = int(cell) - c0
                if not owned[k]:
                    continue
                by_row = {p1.getOffset(int(point)) // cdim: int(point)
                          for point in dm.getTransitiveClosure(int(cell))[0] if v0 <= point < v1}
                corners = [by_row[row] for row in vertex_rows[k]]
                ends = {corners.index(vertex) for vertex in cone}
                first, second = next(pair for pair in ((0, 1), (1, 2), (2, 0)) if set(pair) == ends)
                edge_of = {frozenset(int(v) for v in dm.getCone(int(edge))): int(edge)
                           for edge in dm.getCone(int(cell))}
                standard = [p2.getOffset(vertex) // cdim for vertex in corners] + [
                    p2.getOffset(edge_of[frozenset((corners[i], corners[j]))]) // cdim
                    for i, j in ((0, 1), (1, 2), (2, 0))]
                rows = cell_rows[k].tolist()
                cells.append(k)
                pairs.append((first, second))
                orders.append([rows.index(row) for row in standard])
                wall_numbers.append((numbers[corners[first]], numbers[corners[second]],
                                     numbers[int(facet)]))
        self.cells = np.asarray(cells, dtype=int)
        self.p_rows = vertex_rows[self.cells]
        self.v_rows = cell_rows[self.cells]
        self.v_order = np.asarray(orders, dtype=int).reshape(-1, 6)
        local_vertices = p1_coordinates[self.p_rows]
        self.local_vertices = local_vertices
        counts = uw.mpi.comm.allgather(len(self.cells))
        first = sum(counts[:uw.mpi.rank])
        self.local_edge_slice = slice(first, first + len(self.cells))
        pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
        local_edges = local_vertices[np.arange(len(self.cells))[:, None], pairs]
        self.vertices = np.concatenate(uw.mpi.comm.allgather(local_vertices))
        self.edges = np.concatenate(uw.mpi.comm.allgather(local_edges.reshape(-1, 2, 2)))
        wall_numbers = np.concatenate(uw.mpi.comm.allgather(
            np.asarray(wall_numbers, dtype=np.int64).reshape(-1, 3)))
        labelled_numbers = np.concatenate(uw.mpi.comm.allgather(numbers[facets]))
        if set(wall_numbers[:, 2].tolist()) != set(labelled_numbers.tolist()):
            raise ValueError("Owned boundary-cell traces do not cover the labelled circular wall.")
        # Every vertex is on the circle (checked above), so a chord's midpoint is
        # at the origin only when the chord is a diameter.
        if np.any(np.linalg.norm(self.edges.mean(axis=1), axis=1) <= 0.0):
            raise ValueError("Circular boundary edges must span less than a semicircle.")
        self.edge_nodes = np.stack((self.edges[:, 0], self.edges[:, 1], self.edges.mean(axis=1)), axis=1)
        # Mesh coordinates are of degree 1 (#978), so a P2 midpoint node is the
        # chord's midpoint.
        self.node_numbers, index = np.unique(wall_numbers, return_inverse=True)
        self.node_index = index.reshape(-1, 3)
        self.nodes = np.empty((len(self.node_numbers), 2))
        self.nodes[self.node_index.ravel()] = self.edge_nodes.reshape(-1, 2)
        # An exterior closed wall has one owned cell per edge and two edges at
        # each vertex. Reject internal circles and incomplete circular arcs.
        _, counts = np.unique(np.sort(self.node_index[:, :2], axis=1), axis=0, return_counts=True)
        _, incidence = np.unique(self.node_index[:, :2], return_counts=True)
        if np.any(counts != 1) or np.any(incidence != 2):
            raise ValueError("Transfer requires a complete exterior circular wall, not an interface or arc.")
        # This rank's velocity rows on the wall: every local vertex and edge whose
        # global number is a wall node, whether or not a labelled facet holding it
        # is local here.
        self.local_boundary = np.zeros(len(velocity.coords), dtype=bool)
        self.local_index = np.zeros(len(velocity.coords), dtype=int)
        for start, stop in ((v0, v1), (f0, f1)):
            points = np.arange(start, stop)
            position = np.searchsorted(self.node_numbers, numbers[points])
            position = np.minimum(position, len(self.node_numbers) - 1)
            on_wall = self.node_numbers[position] == numbers[points]
            for point, node in zip(points[on_wall], position[on_wall]):
                if p2.getDof(int(point)):
                    row = p2.getOffset(int(point)) // cdim
                    self.local_boundary[row], self.local_index[row] = True, node
        p1_dm.destroy()
        p2_dm.destroy()
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

    def refresh_geometry(self):
        """Build the cached geometry again if the mesh coordinates moved or the
        declaration changed.

        The declaration is read on every call (``mesh.analytic_surface``), so a
        wall whose surface was released is refused with the reason whether or
        not any coordinate moved (:meth:`~underworld3.discretisation.Mesh.deform`,
        a direct coordinate write, a mover with the wall free, or a release
        alone). The read is not collective and the registry is the same on every
        rank, so every rank refuses or none does. The rebuild is keyed on the
        mesh's geometry generation (``_geometry_version``), never a coordinate
        comparison, and on the declared radius; the verdict is reduced so every
        rank rebuilds or none does. A move that kept the wall declared (a mover
        that slides the wall and restores it onto its circle) is followed: the
        geometry is rebuilt from the declared circle and the lagged datum
        restarted from zero. Collective. Returns whether the geometry was rebuilt.
        """
        radius = _declared_circle_radius(self.mesh, self.name)
        changed = (self.mesh._geometry_version != self._built_at_geometry_version
                   or radius != self.radius)
        if not uw.mpi.comm.allreduce(changed, op=MPI.LOR):
            return False
        self._build_geometry(radius)
        self._restart_lagged_datum()
        return True

    def _restart_lagged_datum(self):
        """Zero the lagged datum after a geometry rebuild."""
        raise NotImplementedError

    def cell_values(self):
        """Collect only owned wall-cell P2 coefficients, including empty ranks."""
        local = np.take_along_axis(
            np.asarray(self.velocity.array)[:, 0, :][self.v_rows],
            self.v_order[..., None], axis=1)
        return np.concatenate(uw.mpi.comm.allgather(local))

class _ZeroSlipWall(_CircularWall):
    """One lagged full-vector essential datum on a physical circle."""

    def __init__(self, stokes, boundary, radius, sign, quadrature):
        super().__init__(stokes, boundary, radius, sign, quadrature)
        self.field = uw.discretisation.MeshVariable(
            f"ZeroSlipShift_{stokes.instance_number}_{boundary}", self.mesh, 2,
            degree=2, continuous=True)
        self.values = np.zeros_like(self.nodes)

    def _restart_lagged_datum(self):
        self.values = np.zeros_like(self.nodes)

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


def _add_cylindrical_zero_slip_transfer(stokes, *, boundary, sign, quadrature):
    """Register one auxiliary P2 vector essential datum before the first solve."""
    if getattr(stokes, "_rotated_freeslip_bcs", None):
        raise ValueError("Zero-slip transfer uses standard Stokes, not rotated free slip.")
    radius = _declared_circle_radius(stokes.mesh, boundary)
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
    for _, wall in sorted(walls.items()):
        wall.refresh_geometry()
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
