"""Consistent Boundary Flux (CBF) recovery — general across solvers.

The residual of the assembled interior (volume) FEM problem, read at an essential-BC
boundary node, is the *consistent* nodal flux there (Gresho et al.). De-smearing that
nodal reaction with the boundary mass gives a pointwise surface flux:

  * scalar diffusion / advection-diffusion  →  surface heat flux  -k dT/dn  (Nusselt),
  * Stokes                                   →  boundary traction sigma.n   (sigma_nn).

This module holds the parts that do not depend on the equation: the boundary-node
gathering, the boundary-mass de-smear (lumped / consistent), and the field hand-off.
The equation-specific bit — extracting the nodal reaction — is the solver method
``_assemble_volume_reaction``, which returns each rank's RAW (per-rank) volume FEM
residual; the complete reaction at a boundary node shared across a partition cut is then
assembled here in ``_desmear`` by SUMMING each rank's partial contribution by coordinate.
In 3D, the complete boundary mass is assembled and solved once on rank zero; only the
recovered values needed by each rank are scattered back. This avoids replicating the
global P2 surface mesh and sparse solve on every rank.

``mass="auto"`` (default) uses a diagonal lumped mass where lumping is pointwise
sound (the 2D P1/P2 line traces and the 3D P1 triangle trace), and the consistent
mass otherwise. A 3D P2 triangle has exactly zero row sum at every vertex; a 2D trace
of degree ≥ 3 places its edge-interior nodes asymmetrically (Gauss-Jacobi), where
row-sum lumping is only O(h) pointwise — both take the consistent solve. A degree ≥ 3
trace also carries several interpolation nodes per edge point, and each keeps its own
coordinate — keying both by the edge's single coordinate silently collapsed them
(issue #459).
``remove_mean=False`` (default) keeps the physical mean flux (the Nusselt number);
set ``remove_mean=True`` for a gauge-free field (e.g. dynamic topography).
"""
import numpy as np
from mpi4py import MPI

# One rule for a boundary facet's outward normal and measure, shared with the two
# sibling accumulators (rotated_bc._boundary_velocity_nodes,
# Mesh._assemble_boundary_normal).
from underworld3.utilities.facet_normals import facet_measure_and_normal


# M_e = (area / 12) * _P1_TRIANGLE_MASS.
_P1_TRIANGLE_MASS = np.array(
    (
        (2.0, 1.0, 1.0),
        (1.0, 2.0, 1.0),
        (1.0, 1.0, 2.0),
    )
)

# M_e = (area / 180) * _P2_TRIANGLE_MASS. Node order: vertices
# (0, 1, 2), then edge nodes (01, 12, 20).
_P2_TRIANGLE_MASS = np.array(
    (
        (6.0, -1.0, -1.0, 0.0, -4.0, 0.0),
        (-1.0, 6.0, -1.0, 0.0, 0.0, -4.0),
        (-1.0, -1.0, 6.0, -4.0, 0.0, 0.0),
        (0.0, 0.0, -4.0, 32.0, 16.0, 16.0),
        (-4.0, 0.0, 0.0, 16.0, 32.0, 16.0),
        (0.0, -4.0, 0.0, 16.0, 16.0, 32.0),
    )
)


def _key(c, dim):
    return tuple(round(float(t), 9) for t in np.asarray(c).ravel()[:dim])


def _boundary_stratum_is(dm, mesh, boundary):
    """Facet stratum IS for ``boundary`` via the CONSOLIDATED ``UW_Boundaries`` label
    (boundaries are distinguished by value on this one label — the per-boundary labels
    named like the boundary do not survive mesh adaptation; only ``UW_Boundaries`` is
    rebuilt). Returns None if this rank owns no part of the boundary. Raises a clear
    error for an unknown boundary name."""
    match = [b.value for b in mesh.boundaries if b.name == boundary]
    if not match:
        raise ValueError(
            f"Unknown boundary {boundary!r}; known: {[b.name for b in mesh.boundaries]}")
    label = dm.getLabel("UW_Boundaries")
    if label is None:
        return None
    return label.getStratumIS(match[0])


def _point_coord(dm, dim, cvec, csec, v0, v1, q):
    """Coordinate of a DMPlex point (vertex → its coord; higher point → mean of its
    closure vertices)."""
    if v0 <= q < v1:
        return cvec[csec.getOffset(q) // dim]
    clo = dm.getTransitiveClosure(q)[0]
    verts = [int(c) for c in clo if v0 <= c < v1]
    return np.mean([cvec[csec.getOffset(v) // dim] for v in verts], axis=0)


def _trace_interior_coords(solver, degree):
    """Coordinates of the EDGE-INTERIOR interpolation nodes of a continuous Lagrange
    field of ``degree``, keyed by DMPlex edge point: ``{edge: (degree-1, cdim) array}``
    in section slot order. A degree-3 trace carries TWO nodes per edge; the coordinate
    section stores only one coordinate per point, so these must be built by
    interpolating the mesh coordinate field into a matching-degree space (issue #459).
    The space is created exactly as UW3 creates every field FE (``createDefault``,
    ``node_endpoints=False`` — see ``Mesh._get_coords_for_basis``), so the per-point
    node ordering matches the solver field's section by construction.
    COLLECTIVE on the DM's communicator — every rank must call this, boundary or not."""
    from petsc4py import PETSc

    mesh = solver.mesh
    cdim = mesh.cdim
    dmold = solver.dm.getCoordinateDM()
    dmold.createDS()
    dmnew = dmold.clone()
    prefix = "cbf_trace_coord_"
    options = PETSc.Options()
    options[prefix + "petscspace_degree"] = degree
    options[prefix + "petscdualspace_lagrange_continuity"] = True
    options[prefix + "petscdualspace_lagrange_node_endpoints"] = False
    fe = PETSc.FE().createDefault(
        mesh.dim, cdim, mesh.isSimplex, mesh.qdegree, prefix, PETSc.COMM_SELF)
    dmnew.setField(0, fe)
    dmnew.createDS()
    mat_interp, vec_scale = dmold.createInterpolation(dmnew)
    coords_new_g = dmnew.getGlobalVec()
    coords_new_l = dmnew.getLocalVec()
    mat_interp.mult(solver.dm.getCoordinates(), coords_new_g)
    dmnew.globalToLocal(coords_new_g, coords_new_l)
    arr = np.asarray(coords_new_l.array).reshape(-1, cdim).copy()
    sec = dmnew.getLocalSection()
    e0, e1 = dmnew.getDepthStratum(1)
    out = {}
    for e in range(e0, e1):
        ndof = sec.getDof(e)
        if ndof > 0:
            row = sec.getOffset(e) // cdim
            out[e] = arr[row: row + ndof // cdim]
    dmnew.restoreGlobalVec(coords_new_g)
    dmnew.restoreLocalVec(coords_new_l)
    mat_interp.destroy()
    if vec_scale is not None:
        vec_scale.destroy()
    fe.destroy()
    dmnew.destroy()
    return out


def _line_mass_1d(ts):
    """Consistent 1-D line-element mass per unit length for a Lagrange basis with
    nodes at parameters ``ts`` in [0, 1]: ``M_ij = ∫ L_i L_j dt`` (Gauss–Legendre,
    exact for the polynomial integrand). The lumped row sums are ``∫ L_i`` by
    partition of unity; their positivity is checked where the lumped path uses them."""
    t = np.asarray(ts, dtype=float)
    if np.min(np.diff(np.sort(t))) < 1e-12:
        raise RuntimeError(
            "Line-trace interpolation nodes are not distinct — the per-node "
            "coordinate build is inconsistent with the field layout (issue #459).")
    degree = len(t) - 1
    xq, wq = np.polynomial.legendre.leggauss(degree + 1)
    xq = 0.5 * (xq + 1.0)
    wq = 0.5 * wq
    L = np.ones((len(t), len(xq)))
    for i, ti in enumerate(t):
        for j, tj in enumerate(t):
            if i != j:
                L[i] *= (xq - tj) / (ti - tj)
    return (L * wq) @ L.T


def _boundary_field_nodes(solver, boundary, field_id=0):
    """Interpolation nodes carrying `field_id` DOFs on `boundary`, one entry per NODE
    as ``(point, slot, coord)``. A DMPlex point can carry several nodes — a degree-3
    trace has two edge-interior nodes per edge point — and each keeps its OWN
    coordinate: keying both by the point's single coordinate collapses them in the
    de-smear and silently drops reactions (issue #459).
    Parallel-safe and COLLECTIVE (the per-node coordinate build interpolates the mesh
    coordinate field, and whether it is needed is agreed globally): a rank owning no
    part of the boundary still participates; ghost/shared nodes are included and their
    partial per-rank reactions are summed by coordinate in ``_desmear``."""
    dm = solver.dm
    dim = solver.mesh.dim
    lsec = dm.getLocalSection()
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)
    fS, fE = dm.getHeightStratum(1)
    ncomp = lsec.getFieldComponents(field_id)
    sis = _boundary_stratum_is(dm, solver.mesh, boundary)
    facets = [] if not (sis and sis.getSize() > 0) else [
        int(z) for z in sis.getIndices() if fS <= int(z) < fE]
    seen = set(); points = []
    for f in facets:
        for q in (int(c) for c in dm.getTransitiveClosure(f)[0]):
            if q in seen:
                continue
            fdof = lsec.getFieldDof(q, field_id)
            if fdof <= 0:
                continue
            seen.add(q)
            if fdof % ncomp:
                raise RuntimeError(
                    f"Field {field_id} carries {fdof} DOFs at point {q} with "
                    f"{ncomp} components — not a nodal (Lagrange) layout.")
            points.append((q, fdof // ncomp))
    nnodes_max = dm.comm.tompi4py().allreduce(
        max((m for _q, m in points), default=1), op=MPI.MAX)
    interior = _trace_interior_coords(solver, nnodes_max + 1) if nnodes_max >= 2 else {}
    out = []
    for q, m in points:
        if m == 1:
            out.append((q, 0, _point_coord(dm, dim, cvec, csec, v0, v1, q)))
        else:
            if q not in interior:
                # multi-node points other than mesh edges (e.g. the interior nodes of
                # a degree-4 face in 3D) have no per-node coordinate build yet
                raise NotImplementedError(
                    f"Boundary point {q} carries {m} interpolation nodes but only "
                    "edge-interior nodes have per-node coordinates (issue #459).")
            for slot, xc in enumerate(np.asarray(interior[q], dtype=float)[:m]):
                out.append((q, slot, xc[:dim]))
    return out, lsec, csec, cvec, v0, v1, interior


def _node_normals(solver, boundary, normal, nodes, dm, dim, cvec, csec, v0, v1):
    """Per-node outward unit normal (only needed to project a vector reaction).
    ``normal`` is None (geometric facet normal), a sympy 1×dim Matrix (analytic,
    lambdified), or a constant (dim,) vector."""
    # The geometric branch below now takes its orientation and its measure weight from
    # the shared `facet_measure_and_normal` (the #560/#561 rule), so it is no longer a
    # stale copy of the pre-#560 bisector.
    #
    # TODO(parallel): it still accumulates over THIS RANK's labelled facets only, so a
    # node on a partition seam would get a partial stencil — the #564 defect. It is
    # unreachable today (the only caller guards it with `if normal is not None`), which
    # is why it is not carrying its own cross-rank reduction: wire one in from
    # `rotated_bc._sum_facet_normals_across_ranks` before making this branch live.
    sym_fn = const = None
    if normal is not None:
        try:
            import sympy
            if isinstance(normal, sympy.Matrix):
                sym_fn = sympy.lambdify(list(solver.mesh.X),
                                        [normal[0, k] for k in range(dim)], "numpy")
        except Exception:
            sym_fn = None
        if sym_fn is None:
            const = np.asarray(normal, dtype=float).ravel()
    nmap = {}
    pts = {q for q, _s, _c in nodes}
    if normal is None:
        # accumulate area-weighted facet normals to the closure points; every node of
        # a point (e.g. both P3 edge-interior nodes) shares its point's facet normal
        sis = _boundary_stratum_is(dm, solver.mesh, boundary)
        facets = [] if not (sis and sis.getSize() > 0) else [int(z) for z in sis.getIndices()]
        fS, fE = dm.getHeightStratum(1)
        acc = {}
        for f in facets:
            if not (fS <= f < fE):
                continue
            measure, ne, _exterior = facet_measure_and_normal(dm, f)
            for q in (int(c) for c in dm.getTransitiveClosure(f)[0]):
                if q in pts:
                    acc[q] = acc.get(q, np.zeros(dim)) + measure * ne
        for q, s, _c in nodes:
            nn = acc.get(q, np.zeros(dim))
            nmap[(q, s)] = nn / (np.linalg.norm(nn) + 1e-30)
    else:
        for q, s, c in nodes:
            ne = np.asarray(sym_fn(*c), float).ravel() if sym_fn is not None else const.copy()
            nmap[(q, s)] = ne / (np.linalg.norm(ne) + 1e-30)
    return nmap


def _node_reactions(xs, R, dim, boundary):
    """Coordinate-keyed nodal reactions, refusing the #459 collapse: two reaction
    nodes sharing one coordinate key would silently overwrite each other."""
    nodeR = {_key(x, dim): float(r) for x, r in zip(xs, R)}
    if len(nodeR) != len(xs):
        raise RuntimeError(
            f"{len(xs)} boundary reaction nodes on {boundary!r} collapse onto "
            f"{len(nodeR)} distinct coordinate keys — per-node coordinates are not "
            "distinct (issue #459: a multi-node trace point needs true interpolation-"
            "node coordinates, not the point's single coordinate).")
    return nodeR


def _desmear(solver, boundary, xs, R, mass, remove_mean, partial_reaction=True,
             edge_node_coords=None):
    """De-smear per-node reaction loads R (aligned with xs) into a pointwise flux via the
    boundary mass. In 3D, coordinate-keyed reactions and trace elements are gathered to
    rank zero, which forms and solves the global system once; the flux values requested
    by each rank are then scattered in local ``xs`` order. Returns the flux at this
    rank's local nodes.

    ``partial_reaction`` controls how a boundary node shared across a partition cut is
    reconciled across ranks: ``True`` (default) SUMS each rank's contribution — correct
    when R is the RAW per-rank volume residual (``boundary_flux``, DM overlap=0); ``False``
    OVERWRITES (all ranks already agree) — correct when R comes from an ASSEMBLED global
    operator, e.g. the rotated free-slip reaction ``Q(A·u − b)`` (``rotated_bc``).

    ``edge_node_coords`` (2D, trace degree ≥ 3 only) is the ``_trace_interior_coords``
    map of per-edge interior node coordinates; ``boundary_flux`` passes the one it built
    so the element keys match ``xs`` exactly. ``None`` builds it on demand (collective)."""
    dm = solver.dm; dim = solver.mesh.dim; comm = dm.comm.tompi4py()
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)
    if mass not in ("auto", "lumped", "consistent", "p1"):
        raise ValueError("mass must be 'auto', 'lumped', 'consistent', or 'p1' "
                         "(P1-projected recovery on a 3D P2 trace).")
    if dim == 3:
        lsec = dm.getLocalSection()
        ncomp = lsec.getFieldComponents(0)
        f0, f1 = dm.getHeightStratum(1)
        e0, e1 = dm.getDepthStratum(1)

        def coord(q):
            return _point_coord(dm, dim, cvec, csec, v0, v1, q)

        nodeR = _node_reactions(xs, R, dim, boundary)
        sis = _boundary_stratum_is(dm, solver.mesh, boundary)
        facets = [] if not (sis and sis.getSize() > 0) else [
            int(q) for q in sis.getIndices() if f0 <= int(q) < f1
        ]
        local_elements = []

        for facet in facets:
            closure = [int(q) for q in dm.getTransitiveClosure(facet)[0]]
            vertices = [q for q in closure if v0 <= q < v1]
            edges = [q for q in closure if e0 <= q < e1]
            if len(vertices) != 3:
                raise NotImplementedError(
                    "3D boundary-flux recovery currently requires triangular facets."
                )
            if lsec.getFieldDof(facet, 0) > 0:
                raise NotImplementedError(
                    "3D boundary-flux recovery supports P1 or P2 triangular traces."
                )

            vertex_coords = [np.asarray(coord(q), dtype=float) for q in vertices]
            vertex_keys = [_key(value, dim) for value in vertex_coords]
            edge_midpoints = {}
            for edge in edges:
                edge_dof = lsec.getFieldDof(edge, 0)
                if edge_dof <= 0:
                    continue
                if edge_dof != ncomp:
                    raise NotImplementedError(
                        "3D boundary-flux recovery supports P1 or P2 triangular traces."
                    )
                edge_vertices = [
                    int(q)
                    for q in dm.getTransitiveClosure(edge)[0]
                    if v0 <= int(q) < v1
                ]
                if len(edge_vertices) == 2:
                    edge_key = frozenset(_key(coord(q), dim) for q in edge_vertices)
                    edge_midpoints[edge_key] = _key(coord(edge), dim)

            a, b, c = vertex_coords
            area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
            if not edge_midpoints:
                local_elements.append((1, tuple(vertex_keys), area))
                continue
            if len(edge_midpoints) != 3:
                raise NotImplementedError(
                    "3D boundary-flux recovery supports P1 or complete P2 triangular traces."
                )

            m01 = edge_midpoints[frozenset((vertex_keys[0], vertex_keys[1]))]
            m12 = edge_midpoints[frozenset((vertex_keys[1], vertex_keys[2]))]
            m20 = edge_midpoints[frozenset((vertex_keys[2], vertex_keys[0]))]
            local_elements.append(
                (
                    2,
                    tuple(vertex_keys) + (m01, m12, m20),
                    area,
                )
            )

        local_keys = [_key(x, dim) for x in xs]
        gathered = comm.gather((nodeR, local_elements, local_keys), root=0)
        flux_by_rank = None
        root_error = None

        if comm.rank == 0:
            try:
                requested_keys = [rank_keys for _rank_r, _rank_e, rank_keys in gathered]

                R_by = {}
                elements = {}
                for rank_values, rank_elements, _rank_keys in gathered:
                    for key, value in rank_values.items():
                        R_by[key] = R_by.get(key, 0.0) + value if partial_reaction else value
                    for order, nodes, area in rank_elements:
                        elements[(order, tuple(sorted(nodes)))] = (order, nodes, area)
                del gathered

                orders = {order for order, _nodes, _area in elements.values()}
                if len(orders) != 1:
                    raise RuntimeError(
                        f"Expected one trace order on boundary {boundary!r}, "
                        f"found {sorted(orders)}."
                    )
                order = orders.pop()
                if mass == "auto":
                    mass = "consistent" if order == 2 else "lumped"
                if order == 2 and mass == "lumped":
                    raise ValueError(
                        "A 3D P2 triangular trace has zero row-sum mass at its "
                        "vertices; use mass='consistent' (pointwise, carries the "
                        "vertex-integral checkerboard risk) or mass='p1' "
                        "(P1-projected, monotone — the choice for driving a P1 "
                        "surface field) for boundary-flux recovery."
                    )
                mid_owners = {}
                if mass == "p1":
                    if order != 2:
                        mass = "lumped"  # P1 trace: p1 IS lumped
                    else:
                        # Fold each P2 midpoint load onto its two P1 vertices, then
                        # de-smear with the monotone P1 lumped triangle mass.
                        new_elements = {}
                        for _order, nodes, area in elements.values():
                            vk = nodes[:3]
                            m01, m12, m20 = nodes[3:]
                            mid_owners[m01] = (vk[0], vk[1])
                            mid_owners[m12] = (vk[1], vk[2])
                            mid_owners[m20] = (vk[2], vk[0])
                            new_elements[(1, tuple(sorted(vk)))] = (1, vk, area)
                        folded = {}
                        for key, value in R_by.items():
                            if key in mid_owners:
                                va, vb = mid_owners[key]
                                folded[va] = folded.get(va, 0.0) + 0.5 * value
                                folded[vb] = folded.get(vb, 0.0) + 0.5 * value
                            else:
                                folded[key] = folded.get(key, 0.0) + value
                        R_by = folded
                        elements = new_elements
                        order = 1
                        mass = "lumped"

                keys = sorted(R_by)
                global_index = {key: i for i, key in enumerate(keys)}
                reaction = np.array([R_by[key] for key in keys], dtype=float)
                if mass == "lumped":
                    boundary_mass = np.zeros(len(keys), dtype=float)
                    for _order, nodes, area in elements.values():
                        for key in nodes:
                            boundary_mass[global_index[key]] += area / 3.0
                    missing = np.flatnonzero(boundary_mass <= 0.0)
                    if missing.size:
                        raise RuntimeError(
                            f"Boundary mass is zero at {missing.size} nodes on " f"{boundary!r}."
                        )
                    flux = reaction / boundary_mass
                elif mass == "consistent":
                    from scipy.sparse import coo_matrix
                    from scipy.sparse.linalg import spsolve

                    reference_mass = _P1_TRIANGLE_MASS if order == 1 else _P2_TRIANGLE_MASS
                    mass_scale = 12.0 if order == 1 else 180.0
                    nodes_per_element = reference_mass.shape[0]
                    entries_per_element = nodes_per_element**2
                    entry_count = len(elements) * entries_per_element
                    rows = np.empty(entry_count, dtype=np.int64)
                    cols = np.empty(entry_count, dtype=np.int64)
                    values = np.empty(entry_count, dtype=float)
                    cursor = 0
                    for _order, nodes, area in elements.values():
                        indices = np.fromiter(
                            (global_index[key] for key in nodes),
                            dtype=np.int64,
                            count=nodes_per_element,
                        )
                        next_cursor = cursor + entries_per_element
                        rows[cursor:next_cursor] = np.repeat(indices, nodes_per_element)
                        cols[cursor:next_cursor] = np.tile(indices, nodes_per_element)
                        values[cursor:next_cursor] = ((area / mass_scale) * reference_mass).ravel()
                        cursor = next_cursor
                    surface_mass = coo_matrix(
                        (values, (rows, cols)), shape=(len(keys), len(keys))
                    ).tocsr()
                    del rows, cols, values
                    surface_mass.sum_duplicates()
                    flux = np.asarray(spsolve(surface_mass, reaction), dtype=float)
                    boundary_mass = np.asarray(surface_mass @ np.ones(len(keys), dtype=float))
                    if not np.all(np.isfinite(flux)):
                        raise RuntimeError(
                            "Consistent boundary-mass solve failed on boundary " f"{boundary!r}."
                        )
                if remove_mean:
                    mean = float(np.dot(flux, boundary_mass) / np.sum(boundary_mass))
                    flux -= mean

                def value_at(key):
                    if key in global_index:
                        return flux[global_index[key]]
                    # P1-projected mode: a P2 midpoint reads the P1 interpolant.
                    va, vb = mid_owners[key]
                    return 0.5 * (flux[global_index[va]] + flux[global_index[vb]])

                flux_by_rank = [
                    np.array([value_at(key) for key in rank_keys], dtype=float)
                    for rank_keys in requested_keys
                ]
            except Exception as error:
                root_error = (type(error).__name__, str(error))

        root_error = comm.bcast(root_error, root=0)
        if root_error is not None:
            error_name, error_message = root_error
            error_type = {
                "ValueError": ValueError,
                "NotImplementedError": NotImplementedError,
                "RuntimeError": RuntimeError,
            }.get(error_name, RuntimeError)
            raise error_type(error_message)

        return np.asarray(comm.scatter(flux_by_rank, root=0), dtype=float)

    if dim != 2:
        raise NotImplementedError(
            f"Boundary-flux recovery is not implemented for mesh dimension {dim}."
        )
    if mass == "p1":
        mass = "lumped"    # P1 consumers read vertex values; vertex lumping is sound

    lsec = dm.getLocalSection()
    ncomp = lsec.getFieldComponents(0)
    e0, e1 = dm.getDepthStratum(1)
    def vcoord(q): return cvec[csec.getOffset(q) // dim]
    nodeR = _node_reactions(xs, R, dim, boundary)
    sis = _boundary_stratum_is(dm, solver.mesh, boundary)
    strat = [] if not (sis and sis.getSize() > 0) else [int(z) for z in sis.getIndices()]
    edges = [q for q in strat if e0 <= q < e1]
    # Interior-node count per edge from the SECTION (structural — the trace order is
    # never sniffed from coordinate keys): 0 → P1, 1 → P2, m → degree m+1. Whether the
    # per-node coordinate build is needed is agreed globally (it is collective).
    if edge_node_coords is None:
        m_max = comm.allreduce(
            max((lsec.getFieldDof(e, 0) // ncomp for e in edges), default=0), op=MPI.MAX)
        edge_node_coords = _trace_interior_coords(solver, m_max + 1) if m_max >= 2 else {}
    local_elems = []
    for e in edges:
        m = lsec.getFieldDof(e, 0) // ncomp
        a, b = (int(c) for c in dm.getCone(e))
        xa, xb = vcoord(a), vcoord(b)
        h = float(np.hypot(*(xb - xa)))
        if m >= 2:
            xin = np.asarray(edge_node_coords[e], dtype=float)[:m, :dim]
            keys_e = (_key(xa, dim), *(_key(x, dim) for x in xin), _key(xb, dim))
            # node parameters along the edge chord, measured from the actual node
            # coordinates so the element mass matches the basis in effect
            ts_e = (0.0, *(float(np.dot(x - xa, xb - xa) / (h * h)) for x in xin), 1.0)
        elif m == 1:
            cmid = _point_coord(dm, dim, cvec, csec, v0, v1, e)
            keys_e = (_key(xa, dim), _key(cmid, dim), _key(xb, dim))
            ts_e = (0.0, 0.5, 1.0)
        else:
            keys_e = (_key(xa, dim), _key(xb, dim))
            ts_e = (0.0, 1.0)
        local_elems.append((keys_e, ts_e, h))

    # Reconcile the nodal reaction across ranks by coordinate. partial_reaction=True: SUM
    # (raw per-rank residual, DM overlap=0, so a cut node holds only each rank's partial
    # → summing assembles the complete reaction, matching the rock-solid volume integral).
    # partial_reaction=False: OVERWRITE (already-assembled global reaction, all ranks agree
    # — summing would double-count shared nodes).
    R_by = {}
    for d in comm.allgather(nodeR):
        for k, v in d.items():
            R_by[k] = (R_by.get(k, 0.0) + v) if partial_reaction else v
    uniq = {}
    for lst in comm.allgather(local_elems):
        for keys_e, ts_e, h in lst:
            uniq[keys_e] = (ts_e, h)
    keys = sorted(R_by.keys()); gi = {k: i for i, k in enumerate(keys)}
    n = len(keys); Rg = np.zeros(n)
    for k, i in gi.items():
        Rg[i] = R_by[k]
    # A mixed trace (field DOFs on only part of the boundary's edges) means the field
    # layout is inconsistent with the trace — raise rather than guess (issue #413).
    orders = {len(keys_e) for keys_e in uniq}
    if len(orders) > 1:
        raise NotImplementedError(
            "2D boundary-flux recovery found different trace orders on parts of "
            "the boundary; mixed traces are not supported."
        )
    if mass == "auto":
        # The row-sum lumped de-smear is pointwise-exact for a linear flux only on
        # the symmetric P1/P2 node layouts. A degree >= 3 trace places its interior
        # nodes asymmetrically within each edge (PETSc's Gauss-Jacobi nodes), where
        # lumping is only O(h) pointwise — the consistent line mass is exact there
        # (up to the documented corner mixing, which it spreads over ~one element).
        mass = "lumped" if max(orders, default=2) <= 3 else "consistent"
    missing = {k for keys_e in uniq for k in keys_e if k not in gi}
    if missing:
        raise RuntimeError(
            f"{len(missing)} trace nodes on {boundary!r} have no reaction entry — "
            "the caller's node list does not cover the trace's interpolation nodes.")
    if mass == "lumped":
        mL = np.zeros(n)
        for keys_e, (ts_e, h) in uniq.items():
            if len(ts_e) == 2:
                w = (0.5, 0.5)
            elif len(ts_e) == 3:
                w = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)
            else:
                w = _line_mass_1d(ts_e).sum(axis=1)     # ∫ L_i by partition of unity
                if np.any(w <= 0.0):
                    raise ValueError(
                        f"The degree-{len(ts_e) - 1} line trace has non-positive "
                        "lumped row sums; use mass='consistent'.")
            for kk, wk in zip(keys_e, w):
                mL[gi[kk]] += h * wk
        sig = Rg / mL
    else:
        # consistent line mass — a dense (n×n) solve in the number of boundary nodes
        # (O(n^3)); fine for a 1D boundary (n ~ resolution) but prefer the default lumped
        # (O(n), monotone) for very large boundaries.
        M = np.zeros((n, n))
        Me2 = np.array([[4., 2, -1], [2, 16, 2], [-1, 2, 4]])
        Me1 = np.array([[2., 1], [1, 2]])
        for keys_e, (ts_e, h) in uniq.items():
            if len(ts_e) == 3:
                Mh = (h / 30.0) * Me2
            elif len(ts_e) == 2:
                Mh = (h / 6.0) * Me1
            else:
                Mh = h * _line_mass_1d(ts_e)
            idx = [gi[k] for k in keys_e]
            M[np.ix_(idx, idx)] += Mh
        sig = np.linalg.solve(M, Rg)
    if remove_mean:
        sig = sig - sig.mean()
    return np.array([sig[gi[_key(x, dim)]] for x in xs])


def boundary_flux(solver, boundary, mass="auto", remove_mean=False, normal=None):
    """See ``SolverBaseClass.boundary_flux``. Returns ``(xs, flux)`` for this rank's
    boundary nodes; scalar solver → normal flux, vector solver → traction (or its normal
    component if ``normal`` is given)."""
    dm = solver.dm; dim = solver.mesh.dim
    ra = np.asarray(solver._assemble_volume_reaction()).ravel()
    nodes, lsec, csec, cvec, v0, v1, edge_nodes = _boundary_field_nodes(
        solver, boundary, field_id=0)
    ncomp = lsec.getFieldComponents(0)
    xs = np.array([c for _q, _s, c in nodes]) if nodes else np.zeros((0, dim))

    if ncomp == 1:
        # one reaction per interpolation node: slot s indexes within the point's
        # field offset (two P3 edge-interior nodes → slots 0 and 1)
        R = np.array([ra[lsec.getFieldOffset(q, 0) + s] for q, s, _c in nodes]) \
            if nodes else np.zeros(0)
        flux = _desmear(solver, boundary, xs, R, mass, remove_mean,
                        edge_node_coords=edge_nodes)
        return xs, flux

    # vector reaction (traction sigma.n at each node); per-point DOFs are node-major
    # with components contiguous per node
    Rvec = np.array([ra[lsec.getFieldOffset(q, 0) + s * ncomp:
                        lsec.getFieldOffset(q, 0) + (s + 1) * ncomp]
                     for q, s, _c in nodes]) if nodes else np.zeros((0, ncomp))
    if normal is not None:
        # scalar NORMAL component sigma_nn = n.(sigma.n)
        nmap = _node_normals(solver, boundary, normal, nodes, dm, dim, cvec, csec, v0, v1)
        Rn = np.array([float(np.dot(nmap[(q, s)], Rvec[i]))
                       for i, (q, s, _c) in enumerate(nodes)]) if nodes else np.zeros(0)
        return xs, _desmear(solver, boundary, xs, Rn, mass, remove_mean,
                            edge_node_coords=edge_nodes)
    # full traction vector: de-smear each component independently
    cols = [_desmear(solver, boundary, xs, Rvec[:, k] if len(Rvec) else np.zeros(0),
                     mass, remove_mean, edge_node_coords=edge_nodes)
            for k in range(ncomp)]
    return xs, (np.column_stack(cols) if nodes else np.zeros((0, ncomp)))


def write_boundary_scalar_field(solver, field, value_by_key, dim):
    """Write ``value_by_key`` (coordinate-key → scalar) onto a scalar MeshVariable
    ``field`` at the matching nodes; interior nodes untouched. Returns ``field``.

    The field is written ONCE from a local numpy copy: a per-node write to ``var.data``
    fires the variable's write-callback each time, and the boundary-node count differs per
    rank (a rank may own none of the boundary), so per-node writes would desync the
    callback's collective and deadlock. Shared by the boundary-flux and rotated-free-slip
    (dynamic topography) field hand-offs."""
    fc = np.asarray(field.coords)
    newdata = np.asarray(field.data).copy()
    for i in range(fc.shape[0]):
        v = value_by_key.get(_key(fc[i], dim))
        if v is not None:
            newdata[i, 0] = v
    field.data[...] = newdata
    base = getattr(field, "_base_var", field)
    if hasattr(base, "_sync_lvec_to_gvec"):
        base._sync_lvec_to_gvec()
    if hasattr(base, "_canonical_data"):
        base._canonical_data = None
    solver.mesh._stale_lvec = True
    return field


def boundary_flux_field(solver, boundary, field, mass="auto",
                        remove_mean=False, scale=1.0, normal=None):
    r"""See ``SolverBaseClass.boundary_flux_field`` (the documented entry point;
    this free function is its implementation and shares its name). Writes
    ``scale * flux`` onto the scalar MeshVariable ``field`` at the boundary
    nodes (interior untouched).

    ``scale`` is a generic multiplier on the recovered flux. For dynamic
    topography it is the **negated reciprocal** of the buoyancy scale used by
    the expression-return paths: ``scale = -1 / buoyancy_scale``, where
    ``buoyancy_scale`` is :math:`\Delta\rho\, g` as taken by
    ``dynamic_topography`` / ``topography`` (there the division and the minus
    sign are internal). The two parameters are deliberately NOT aliased —
    treating them as one factor invites sign/reciprocal errors.
    """
    dim = solver.mesh.dim
    xs, flux = boundary_flux(solver, boundary, mass=mass, remove_mean=remove_mean, normal=normal)
    flux = np.asarray(flux)
    # a SCALAR field can only hold a scalar flux — a vector solver returns a per-node
    # traction VECTOR unless a `normal` is given to project it. Fail fast rather than
    # silently pairing the flattened vector with the nodes.
    if flux.ndim > 1 and flux.shape[1] != 1:
        raise ValueError(
            "boundary_flux_field target is a scalar MeshVariable but boundary_flux "
            "returned a vector (traction). Pass normal= to project onto the normal "
            "component, or use boundary_flux() directly for the full vector.")
    fmap = {_key(x, dim): scale * float(f) for x, f in zip(np.asarray(xs), flux.ravel())}
    return write_boundary_scalar_field(solver, field, fmap, dim)


def boundary_flux_to_field(*args, **kwargs):
    """Deprecated alias for :func:`boundary_flux_field` (renamed 2026-07 so the
    free function matches the solver method it implements; kept one cycle)."""
    import warnings
    warnings.warn(
        "boundary_flux_to_field is renamed; use boundary_flux_field(...)",
        DeprecationWarning, stacklevel=2)
    return boundary_flux_field(*args, **kwargs)
