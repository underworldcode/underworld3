"""Consistent Boundary Flux (CBF) recovery — general across solvers.

The residual of the assembled interior (volume) FEM problem, read at an essential-BC
boundary node, is the *consistent* nodal flux there (Gresho et al.). De-smearing that
nodal reaction with the boundary mass gives a pointwise surface flux:

  * scalar diffusion / advection-diffusion  →  surface heat flux  -k dT/dn  (Nusselt),
  * Stokes                                   →  boundary traction sigma.n   (sigma_nn).

This module holds the parts that do not depend on the equation: the boundary-node
gathering, the boundary-mass de-smear (lumped / consistent), and the field hand-off.
The equation-specific bit — extracting the nodal reaction — is the solver method
``_assemble_volume_reaction`` (globally assembled, so shared boundary nodes are complete).

``mass="lumped"`` (default) uses the diagonal boundary mass: being an M-matrix it cannot
overshoot where the flux jumps (no Gibbs wiggle) and is a purely local division.
``remove_mean=False`` (default) keeps the physical mean flux (the Nusselt number);
set ``remove_mean=True`` for a gauge-free field (e.g. dynamic topography).
"""
import numpy as np
from mpi4py import MPI


def _key(c, dim):
    return tuple(round(float(t), 9) for t in np.asarray(c).ravel()[:dim])


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
    (guarded); ghost nodes are included (their reaction is completed by the global
    assembly in ``_assemble_volume_reaction``)."""
    dm = solver.dm
    dim = solver.mesh.dim
    lsec = dm.getLocalSection()
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)
    fS, fE = dm.getHeightStratum(1)
    bval = [b.value for b in solver.mesh.boundaries if b.name == boundary][0]
    sis = dm.getStratumIS(boundary, bval)
    if sis is None or sis.handle == 0:
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
        bval = [b.value for b in solver.mesh.boundaries if b.name == boundary][0]
        sis = dm.getStratumIS(boundary, bval)
        facets = [] if (sis is None or sis.handle == 0) else [int(z) for z in sis.getIndices()]
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


def _desmear(solver, boundary, xs, R, mass, remove_mean):
    """De-smear per-node reaction loads R (aligned with xs) into a pointwise flux via the
    boundary mass, assembled globally by a coordinate-keyed allgather so every rank forms
    the identical system. Returns the flux at this rank's local nodes (xs order)."""
    dm = solver.dm; dim = solver.mesh.dim; comm = dm.comm.tompi4py()
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)
    if dim != 2:
        # no line-mass geometry yet in 3D → global-mean lumped fallback
        tot = comm.allreduce(float(np.sum(R)), op=MPI.SUM)
        cnt = comm.allreduce(int(len(R)), op=MPI.SUM)
        m = tot / max(cnt, 1)
        return np.asarray(R) - (m if remove_mean else 0.0)

    e0, e1 = dm.getDepthStratum(1)
    def vcoord(q): return cvec[csec.getOffset(q) // dim]
    nodeR = {_key(x, dim): float(r) for x, r in zip(xs, R)}
    bval = [b.value for b in solver.mesh.boundaries if b.name == boundary][0]
    sis = dm.getStratumIS(boundary, bval)
    strat = [] if (sis is None or sis.handle == 0) else [int(z) for z in sis.getIndices()]
    local_elems = []
    for e in [q for q in strat if e0 <= q < e1]:
        a, b = (int(c) for c in dm.getCone(e))
        cmid = _point_coord(dm, dim, cvec, csec, v0, v1, e)
        h = float(np.hypot(*(vcoord(b) - vcoord(a))))
        local_elems.append((_key(vcoord(a), dim), _key(cmid, dim), _key(vcoord(b), dim), h))

    # SUM the nodal reaction across ranks by coordinate: with overlap=0 a boundary node
    # shared across a partition cut holds only each rank's partial cell contribution, so
    # summing them assembles the complete reaction (matches the rock-solid volume integral).
    R_by = {}
    for d in comm.allgather(nodeR):
        for k, v in d.items():
            R_by[k] = R_by.get(k, 0.0) + v
    uniq = {}
    for lst in comm.allgather(local_elems):
        for (ka, km, kb, h) in lst:
            uniq[(ka, km, kb)] = h
    keys = sorted(R_by.keys()); gi = {k: i for i, k in enumerate(keys)}
    n = len(keys); Rg = np.zeros(n)
    for k, i in gi.items():
        Rg[i] = R_by[k]
    if mass == "lumped":
        mL = np.zeros(n)
        for (ka, km, kb), h in uniq.items():
            mL[gi[ka]] += h / 6.0; mL[gi[km]] += 2.0 * h / 3.0; mL[gi[kb]] += h / 6.0
        sig = Rg / mL
    else:
        M = np.zeros((n, n))
        Me = np.array([[4., 2, -1], [2, 16, 2], [-1, 2, 4]])
        for (ka, km, kb), h in uniq.items():
            tri = [gi[ka], gi[km], gi[kb]]; Mh = (h / 30.0) * Me
            for ii in range(3):
                for jj in range(3):
                    M[tri[ii], tri[jj]] += Mh[ii, jj]
        sig = np.linalg.solve(M, Rg)
    if remove_mean:
        sig = sig - sig.mean()
    return np.array([sig[gi[_key(x, dim)]] for x in xs])


def boundary_flux(solver, boundary, mass="lumped", remove_mean=False, normal=None):
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


def boundary_flux_to_field(solver, boundary, field, mass="lumped",
                           remove_mean=False, scale=1.0, normal=None):
    """See ``SolverBaseClass.boundary_flux_field``. Writes ``scale * flux`` onto the
    scalar MeshVariable ``field`` at the boundary nodes (interior untouched)."""
    dim = solver.mesh.dim
    xs, flux = boundary_flux(solver, boundary, mass=mass, remove_mean=remove_mean, normal=normal)
    fmap = {_key(x, dim): scale * float(f) for x, f in zip(np.asarray(xs), np.asarray(flux).ravel())}
    fc = np.asarray(field.coords)
    # Write the field ONCE from a local copy: a per-node write to var.data fires the
    # write-callback each time, and the boundary-node count differs per rank (a rank may
    # own none of the boundary), so per-node writes would desync the callback and hang.
    newdata = np.asarray(field.data).copy()
    for i in range(fc.shape[0]):
        v = fmap.get(_key(fc[i], dim))
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
