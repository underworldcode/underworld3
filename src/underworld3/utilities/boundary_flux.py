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
assembled here in ``_desmear`` by SUMMING each rank's partial contribution by coordinate
(the same rock-solid gather used for the boundary mass — no hand-rolled global assembly).

``mass="auto"`` (default) uses a diagonal lumped mass where the trace basis admits
positive row sums (the 2D P2 line trace and the 3D P1 triangle trace), and the consistent
mass otherwise. A 3D P2 triangle has exactly zero row sum at every vertex, so its
pointwise recovery requires the consistent surface-mass solve.
``remove_mean=False`` (default) keeps the physical mean flux (the Nusselt number);
set ``remove_mean=True`` for a gauge-free field (e.g. dynamic topography).
"""
import numpy as np
from mpi4py import MPI


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


def _boundary_field_nodes(solver, boundary, field_id=0):
    """DMPlex points carrying `field_id` DOFs on `boundary`, with their coordinates.
    Parallel-safe: a rank owning no part of the boundary gets a NULL stratum IS
    (guarded); ghost/shared nodes are included and their partial per-rank reactions are
    summed by coordinate in ``_desmear`` to form the complete reaction."""
    dm = solver.dm
    dim = solver.mesh.dim
    lsec = dm.getLocalSection()
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)
    fS, fE = dm.getHeightStratum(1)
    sis = _boundary_stratum_is(dm, solver.mesh, boundary)
    if not (sis and sis.getSize() > 0):
        return [], lsec, csec, cvec, v0, v1
    facets = [int(z) for z in sis.getIndices()]
    seen = set(); out = []
    for f in facets:
        if not (fS <= f < fE):
            continue
        for q in (int(c) for c in dm.getTransitiveClosure(f)[0]):
            if q in seen or lsec.getFieldDof(q, field_id) <= 0:
                continue
            seen.add(q)
            out.append((q, _point_coord(dm, dim, cvec, csec, v0, v1, q)))
    return out, lsec, csec, cvec, v0, v1


def _node_normals(solver, boundary, normal, nodes, dm, dim, cvec, csec, v0, v1):
    """Per-node outward unit normal (only needed to project a vector reaction).
    ``normal`` is None (geometric facet normal), a sympy 1×dim Matrix (analytic,
    lambdified), or a constant (dim,) vector."""
    interior_ref = cvec.mean(axis=0)
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
    coord = {q: c for q, c in nodes}
    if normal is None:
        # accumulate area-weighted facet normals to the closure nodes
        sis = _boundary_stratum_is(dm, solver.mesh, boundary)
        facets = [] if not (sis and sis.getSize() > 0) else [int(z) for z in sis.getIndices()]
        fS, fE = dm.getHeightStratum(1)
        acc = {}
        for f in facets:
            if not (fS <= f < fE):
                continue
            _, cent, nrm = dm.computeCellGeometryFVM(f)
            ne = np.asarray(nrm, float); ne = ne / (np.linalg.norm(ne) + 1e-30)
            if np.dot(ne, np.asarray(cent) - interior_ref) < 0:
                ne = -ne
            for q in (int(c) for c in dm.getTransitiveClosure(f)[0]):
                if q in coord:
                    acc[q] = acc.get(q, np.zeros(dim)) + ne
        for q in coord:
            nn = acc.get(q, np.zeros(dim))
            nmap[q] = nn / (np.linalg.norm(nn) + 1e-30)
    else:
        for q, c in nodes:
            ne = np.asarray(sym_fn(*c), float).ravel() if sym_fn is not None else const.copy()
            nmap[q] = ne / (np.linalg.norm(ne) + 1e-30)
    return nmap


def _desmear(solver, boundary, xs, R, mass, remove_mean, partial_reaction=True):
    """De-smear per-node reaction loads R (aligned with xs) into a pointwise flux via the
    boundary mass, assembled globally by a coordinate-keyed allgather so every rank forms
    the identical system. Returns the flux at this rank's local nodes (xs order).

    ``partial_reaction`` controls how a boundary node shared across a partition cut is
    reconciled across ranks: ``True`` (default) SUMS each rank's contribution — correct
    when R is the RAW per-rank volume residual (``boundary_flux``, DM overlap=0); ``False``
    OVERWRITES (all ranks already agree) — correct when R comes from an ASSEMBLED global
    operator, e.g. the rotated free-slip reaction ``Q(A·u − b)`` (``rotated_bc``)."""
    dm = solver.dm; dim = solver.mesh.dim; comm = dm.comm.tompi4py()
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)
    if mass not in ("auto", "lumped", "consistent"):
        raise ValueError("mass must be 'auto', 'lumped', or 'consistent'.")
    if dim == 3:
        lsec = dm.getLocalSection()
        ncomp = lsec.getFieldComponents(0)
        f0, f1 = dm.getHeightStratum(1)
        e0, e1 = dm.getDepthStratum(1)

        def coord(q):
            return _point_coord(dm, dim, cvec, csec, v0, v1, q)

        nodeR = {_key(x, dim): float(r) for x, r in zip(xs, R)}
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

        R_by = {}
        for rank_values in comm.allgather(nodeR):
            for key, value in rank_values.items():
                R_by[key] = (
                    R_by.get(key, 0.0) + value if partial_reaction else value
                )

        elements = {}
        for rank_elements in comm.allgather(local_elements):
            for order, nodes, area in rank_elements:
                elements[(order, tuple(sorted(nodes)))] = (order, nodes, area)

        orders = {order for order, _nodes, _area in elements.values()}
        if len(orders) != 1:
            raise RuntimeError(
                f"Expected one trace order on boundary {boundary!r}, found {sorted(orders)}."
            )
        order = orders.pop()
        if mass == "auto":
            mass = "consistent" if order == 2 else "lumped"
        if order == 2 and mass == "lumped":
            raise ValueError(
                "A 3D P2 triangular trace has zero row-sum mass at its vertices; "
                "use mass='consistent' for pointwise boundary-flux recovery."
            )

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
                    f"Boundary mass is zero at {missing.size} nodes on {boundary!r}."
                )
            flux = reaction / boundary_mass
        elif mass == "consistent":
            from scipy.sparse import coo_matrix
            from scipy.sparse.linalg import spsolve

            rows = []
            cols = []
            values = []
            reference_mass = (
                _P1_TRIANGLE_MASS if order == 1 else _P2_TRIANGLE_MASS
            )
            mass_scale = 12.0 if order == 1 else 180.0
            for _order, nodes, area in elements.values():
                indices = [global_index[key] for key in nodes]
                element_mass = (area / mass_scale) * reference_mass
                for i, row in enumerate(indices):
                    for j, col in enumerate(indices):
                        rows.append(row)
                        cols.append(col)
                        values.append(element_mass[i, j])
            surface_mass = coo_matrix(
                (values, (rows, cols)), shape=(len(keys), len(keys))
            ).tocsr()
            surface_mass.sum_duplicates()
            flux = np.asarray(spsolve(surface_mass, reaction), dtype=float)
            boundary_mass = np.asarray(
                surface_mass @ np.ones(len(keys), dtype=float)
            )
            if not np.all(np.isfinite(flux)):
                raise RuntimeError(
                    f"Consistent boundary-mass solve failed on boundary {boundary!r}."
                )
        if remove_mean:
            mean = float(np.dot(flux, boundary_mass) / np.sum(boundary_mass))
            flux -= mean
        return np.array([flux[global_index[_key(x, dim)]] for x in xs])

    if dim != 2:
        raise NotImplementedError(
            f"Boundary-flux recovery is not implemented for mesh dimension {dim}."
        )
    if mass == "auto":
        mass = "lumped"

    e0, e1 = dm.getDepthStratum(1)
    def vcoord(q): return cvec[csec.getOffset(q) // dim]
    nodeR = {_key(x, dim): float(r) for x, r in zip(xs, R)}
    sis = _boundary_stratum_is(dm, solver.mesh, boundary)
    strat = [] if not (sis and sis.getSize() > 0) else [int(z) for z in sis.getIndices()]
    local_elems = []
    for e in [q for q in strat if e0 <= q < e1]:
        a, b = (int(c) for c in dm.getCone(e))
        cmid = _point_coord(dm, dim, cvec, csec, v0, v1, e)
        h = float(np.hypot(*(vcoord(b) - vcoord(a))))
        local_elems.append((_key(vcoord(a), dim), _key(cmid, dim), _key(vcoord(b), dim), h))

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
        for (ka, km, kb, h) in lst:
            uniq[(ka, km, kb)] = h
    keys = sorted(R_by.keys()); gi = {k: i for i, k in enumerate(keys)}
    n = len(keys); Rg = np.zeros(n)
    for k, i in gi.items():
        Rg[i] = R_by[k]
    # Trace order from the DATA, not an assumption: a P2 trace has a reaction DOF at
    # every edge point (the midpoint key is in R_by), a P1 trace has vertices only.
    # Assembling P2 line masses against a P1 trace died with a bare KeyError on the
    # missing midpoint (issue #413). A mix means the field layout is inconsistent
    # with the trace — raise rather than guess.
    mids_present = [km in R_by for (ka, km, kb) in uniq]
    if mids_present and any(mids_present) != all(mids_present):
        raise NotImplementedError(
            "2D boundary-flux recovery found edge-midpoint reactions on only part "
            "of the boundary; mixed P1/P2 traces are not supported."
        )
    trace_order = 2 if (mids_present and mids_present[0]) else 1
    if mass == "lumped":
        mL = np.zeros(n)
        for (ka, km, kb), h in uniq.items():
            if trace_order == 2:
                mL[gi[ka]] += h / 6.0; mL[gi[km]] += 2.0 * h / 3.0; mL[gi[kb]] += h / 6.0
            else:
                mL[gi[ka]] += h / 2.0; mL[gi[kb]] += h / 2.0
        sig = Rg / mL
    else:
        # consistent line mass — a dense (n×n) solve in the number of boundary nodes
        # (O(n^3)); fine for a 1D boundary (n ~ resolution) but prefer the default lumped
        # (O(n), monotone) for very large boundaries.
        M = np.zeros((n, n))
        Me2 = np.array([[4., 2, -1], [2, 16, 2], [-1, 2, 4]])
        Me1 = np.array([[2., 1], [1, 2]])
        for (ka, km, kb), h in uniq.items():
            if trace_order == 2:
                nodes = [gi[ka], gi[km], gi[kb]]; Mh = (h / 30.0) * Me2
            else:
                nodes = [gi[ka], gi[kb]]; Mh = (h / 6.0) * Me1
            for ii in range(len(nodes)):
                for jj in range(len(nodes)):
                    M[nodes[ii], nodes[jj]] += Mh[ii, jj]
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
    nodes, lsec, csec, cvec, v0, v1 = _boundary_field_nodes(solver, boundary, field_id=0)
    ncomp = lsec.getFieldComponents(0)
    xs = np.array([c for _q, c in nodes]) if nodes else np.zeros((0, dim))

    if ncomp == 1:
        R = np.array([ra[lsec.getFieldOffset(q, 0)] for q, _c in nodes]) if nodes else np.zeros(0)
        flux = _desmear(solver, boundary, xs, R, mass, remove_mean)
        return xs, flux

    # vector reaction (traction sigma.n at each node)
    Rvec = np.array([ra[lsec.getFieldOffset(q, 0):lsec.getFieldOffset(q, 0) + ncomp]
                     for q, _c in nodes]) if nodes else np.zeros((0, ncomp))
    if normal is not None:
        # scalar NORMAL component sigma_nn = n.(sigma.n)
        nmap = _node_normals(solver, boundary, normal, nodes, dm, dim, cvec, csec, v0, v1)
        Rn = np.array([float(np.dot(nmap[q], Rvec[i])) for i, (q, _c) in enumerate(nodes)]) \
            if nodes else np.zeros(0)
        return xs, _desmear(solver, boundary, xs, Rn, mass, remove_mean)
    # full traction vector: de-smear each component independently
    cols = [_desmear(solver, boundary, xs, Rvec[:, k] if len(Rvec) else np.zeros(0),
                     mass, remove_mean) for k in range(ncomp)]
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
