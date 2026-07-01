"""Development version of underworld3.utilities.rotated_bc — reusable rotated
strong free-slip for the Stokes saddle. Productizes the validated prototypes:
build a per-node rotation Q from boundary normals, rotate the assembled saddle
Â=Q A Qᵀ / b̂=Q b, impose v_n=0 on the rotated normal rows, solve, rotate back
u=Qᵀû, remove the rigid-rotation gauge, and expose σ_nn as the constraint reaction.

Increment 1: box-flat (Q=identity on axis-aligned walls) must reproduce the native
essential free-slip solve bit-for-bit. Direct LU here; FMG wiring is a later step.
"""
import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc


# --------------------------------------------------------------------------- #
#  Rotation construction
# --------------------------------------------------------------------------- #
def _velocity_field_id(solver):
    return 0  # velocity is field 0 in the Stokes saddle


def _point_coord(dm, dim, cvec, csec, v0, v1, q):
    """Coordinate of a DMPlex point (vertex → its coord; higher point → mean of
    its closure vertices)."""
    if v0 <= q < v1:
        return cvec[csec.getOffset(q) // dim]
    clo = dm.getTransitiveClosure(q)[0]
    verts = [int(c) for c in clo if v0 <= c < v1]
    return np.mean([cvec[csec.getOffset(v) // dim] for v in verts], axis=0)


def _boundary_velocity_nodes(solver, boundary, normal=None):
    """DMPlex points carrying velocity DOFs on `boundary`, each with an outward
    unit normal. Dimension-general (2D edges, 3D faces).

    `normal` selects the normal source (per boundary):
      * None      — geometric facet normal from PETSc ``computeCellGeometryFVM``
                    (area-weighted, accumulated to the facet closure points).
      * sympy 1×dim Matrix — an analytic normal (function of mesh.X); evaluated
                    at each node's coordinate. Best for exact curved/planar faces
                    (radial ``X/|X|`` on a spherical cap; a constant on a planar
                    side of a regional spherical box).
      * (dim,) array — a constant normal vector.
    Returns ``[(point, n̂), ...]``.
    """
    dm = solver.dm
    dim = solver.mesh.dim
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)
    lsec = dm.getLocalSection()
    VEL = _velocity_field_id(solver)
    interior_ref = cvec.mean(axis=0)
    bval = [b.value for b in solver.mesh.boundaries if b.name == boundary][0]
    facets = [int(z) for z in dm.getStratumIS(boundary, bval).getIndices()]
    fS, fE = dm.getHeightStratum(1)              # facets (edges in 2D, faces in 3D)

    def coord(q):
        return _point_coord(dm, dim, cvec, csec, v0, v1, q)

    # analytic / constant normal specs → per-node evaluation
    sym_normal = None
    const_normal = None
    if normal is not None:
        try:
            import sympy
            if isinstance(normal, sympy.Matrix):
                sym_normal = normal
        except Exception:
            pass
        if sym_normal is None:
            const_normal = np.asarray(normal, dtype=float).ravel()

    nacc = {}
    pts = set()
    for f in facets:
        if not (fS <= f < fE):
            continue
        # facet outward normal
        if normal is None:
            _, cent, nrm = dm.computeCellGeometryFVM(f)
            ne = np.asarray(nrm, dtype=float)
            ne = ne / (np.linalg.norm(ne) + 1e-30)
            if np.dot(ne, np.asarray(cent) - interior_ref) < 0:
                ne = -ne
        # all velocity points on this facet (closure): verts + edges(3D) + the facet
        clo = dm.getTransitiveClosure(f)[0]
        for q in (int(c) for c in clo):
            if lsec.getFieldDof(q, VEL) <= 0:
                continue
            if normal is not None:
                cq = coord(q)
                if sym_normal is not None:
                    ne = np.array([float(sym_normal[0, k].subs(
                        dict(zip(solver.mesh.X, cq)))) for k in range(dim)])
                else:
                    ne = const_normal.copy()
                ne = ne / (np.linalg.norm(ne) + 1e-30)
            nacc[q] = nacc.get(q, np.zeros(dim)) + ne
            pts.add(q)
    out = []
    for q in pts:
        nrm = nacc[q] / (np.linalg.norm(nacc[q]) + 1e-30)
        out.append((q, nrm))
    return out


def build_rotation(solver, boundaries):
    """Global sparse rotation Q on the composite saddle vector: identity except a
    per-node (normal,tangential) block at each velocity node of `boundaries`.
    Returns (Q, Qt, normal_rows) where normal_rows are the global rows carrying
    the rotated NORMAL velocity component (v_n = n̂·v), to be strongly constrained.
    """
    dm = solver.dm
    N = dm.getGlobalVec().getSize()
    gsec = dm.getGlobalSection()
    lsec = dm.getLocalSection()
    l2g = dm.getLGMap()
    VEL = _velocity_field_id(solver)

    # gather all normals per velocity node across the boundaries. Each entry of
    # `boundaries` is a name (geometric normal) or a (name, normal) pair.
    node_normals = {}
    for spec in boundaries:
        name, normal = spec if isinstance(spec, tuple) else (spec, None)
        for q, nrm in _boundary_velocity_nodes(solver, name, normal=normal):
            node_normals.setdefault(q, []).append(nrm)

    dim = solver.mesh.dim
    Qd = sp.lil_matrix((N, N))
    Qd.setdiag(1.0)
    normal_rows = []
    for q, nrms in node_normals.items():
        lo = lsec.getFieldOffset(q, VEL)
        grows = [int(l2g.apply([lo + c])[0]) for c in range(dim)]
        if any(g < 0 for g in grows):
            continue
        # DIMENSION-GENERAL constraint frame. The accumulated normals span a
        # subspace S (rank r) that the velocity must be orthogonal to:
        #   r=1  → a face  : constrain v_n, (dim-1) tangential free
        #   r=2  → an edge : constrain 2 normal dirs, (dim-2) free (3D edge tangent)
        #   r=dim→ a corner: v = 0 (fully pinned)
        # Build an orthonormal frame E (dim×dim) whose first r rows span S and whose
        # last (dim-r) rows are the free tangential complement (SVD right vectors are
        # a complete orthonormal basis). Q's node block rows = E (rotated component
        # i = E[i]·v); constrain the first r rotated rows.
        M = np.array(nrms, dtype=float)
        # SVD → Vt rows are an orthonormal basis; the first r span the normals.
        _, sv, Vt = np.linalg.svd(M)
        r = int((sv > 1e-8 * (sv[0] if sv.size else 1.0)).sum())
        E = Vt                                   # (dim, dim) orthonormal frame
        for i in range(dim):
            for j in range(dim):
                Qd[grows[i], grows[j]] = E[i, j]
        normal_rows.extend(grows[:r])            # constrain the r normal-space rows
    Qc = Qd.tocsr()
    Q = PETSc.Mat().createAIJ(size=(N, N), csr=(Qc.indptr, Qc.indices, Qc.data))
    Q.assemble()
    Qt = Q.transpose(PETSc.Mat())
    return Q, Qt, sorted(set(normal_rows))


# --------------------------------------------------------------------------- #
#  The rotated solve
# --------------------------------------------------------------------------- #
def solve_rotated_freeslip(solver, boundaries, remove_rotation_gauge=True, verbose=False):
    """Assemble + solve the rotated strong-free-slip Stokes saddle. Fills the
    solver's velocity/pressure fields with the (rotated-back, gauge-removed)
    solution. Returns a dict with the rotation Q and reaction data for σ_nn.

    Called from ``SNES_Stokes_SaddlePt.solve`` after ``_build`` (so the SNES/DM
    exist); when used standalone it builds the solver itself."""
    if getattr(solver, "snes", None) is None:
        solver._setup_pointwise_functions()
        solver._setup_discretisation()
        solver._setup_solver()
    dm = solver.dm
    snes = solver.snes

    Q, Qt, normal_rows = build_rotation(solver, boundaries)

    # A = exact Jacobian at 0 (linear), b = -F(0)
    U0 = dm.getGlobalVec(); U0.set(0.0)
    J = snes.getJacobian()[0]; snes.computeJacobian(U0, J)
    Aorig = J.copy()
    F0 = dm.getGlobalVec(); snes.computeFunction(U0, F0)
    b = F0.copy(); b.scale(-1.0)

    # rotate: Â = Q A Qᵀ, b̂ = Q b
    Ahat = Aorig.ptap(Qt)
    bhat = b.duplicate(); Q.mult(b, bhat)

    # pin one pressure DOF (datum) — row only, keeps B^T coupling
    gsec = dm.getGlobalSection()
    PRE = 1; pin = None; pS, pE = gsec.getChart()
    for q in range(pS, pE):
        if gsec.getFieldDof(q, PRE) > 0 and gsec.getFieldOffset(q, PRE) >= 0:
            pin = gsec.getFieldOffset(q, PRE); break

    # constrain rotated normal rows (v_n=0) + pin
    Ahat.zeroRowsColumns(normal_rows, diag=1.0)
    Ahat.zeroRows([pin], diag=1.0)
    ba = bhat.getArray(); ba[normal_rows] = 0.0; ba[pin] = 0.0; bhat.setArray(ba)

    # direct LU (FMG wiring later)
    ksp = PETSc.KSP().create(); ksp.setOperators(Ahat); ksp.setType("preonly")
    pc = ksp.getPC(); pc.setType("lu"); pc.setFactorSolverType("mumps")
    Uhat = dm.getGlobalVec(); ksp.solve(bhat, Uhat)

    # rotate back u = Qᵀ û → fields
    U = dm.getGlobalVec(); Qt.mult(Uhat, U)
    for name, var in solver.fields.items():
        sg = U.getSubVector(solver._subdict[name][0])
        solver._subdict[name][1].globalToLocal(sg, var.vec)
        U.restoreSubVector(solver._subdict[name][0], sg)

    # Remove the rigid-rotation gauge ONLY when it is a genuine null space of the
    # constrained problem — i.e. when a rigid rotation satisfies every rotated
    # v_n=0 constraint (true on closed circular/spherical free-slip boundaries;
    # FALSE on straight walls, where rotation violates v_n=0 and the constraint
    # itself pins it — projecting there would corrupt the solution).
    removed = False
    if remove_rotation_gauge and _rotation_is_nullspace(solver, Q, normal_rows):
        _remove_rotation_gauge(solver)
        removed = True

    return {"Q": Q, "Qt": Qt, "A": Aorig, "b": b, "U": U, "Uhat": Uhat,
            "normal_rows": normal_rows, "boundaries": list(boundaries),
            "rotation_gauge_removed": removed, "ksp_reason": ksp.getConvergedReason()}


def _rotation_is_nullspace(solver, Q, normal_rows, tol=1e-8):
    """True iff the rigid-body rotation t=(-y,x) satisfies all rotated v_n=0
    constraints — i.e. Q·t is ~0 on every constrained normal row."""
    if solver.mesh.dim != 2 or not normal_rows:
        return False
    v = solver.Unknowns.u
    c = v.coords
    tloc = np.column_stack([-c[:, 1], c[:, 0]])
    # scatter t into a composite global vector, rotate, check the normal rows
    dm = solver.dm
    tg = dm.getGlobalVec(); tg.set(0.0)
    vname = "velocity"
    sg = tg.getSubVector(solver._subdict[vname][0])
    # write t into the velocity meshvar, push to the sub global vec
    saved = v.data.copy()
    v.data[...] = tloc
    solver._subdict[vname][1].localToGlobal(v.vec, sg)
    tg.restoreSubVector(solver._subdict[vname][0], sg)
    v.data[...] = saved
    tr = tg.duplicate(); Q.mult(tg, tr)
    tra = tr.getArray()
    viol = np.linalg.norm(tra[normal_rows]) / (np.linalg.norm(tra) + 1e-30)
    return viol < tol


def boundary_normal_traction(solver, boundary, info, consistent_mass=True):
    """Consistent boundary normal traction σ_nn on `boundary` from the constraint
    reaction of the last rotated-free-slip solve.

    The reaction is r = Q(A·u − b): in the rotated frame its value at each node's
    NORMAL row is the integrated nodal traction ∫_Γ(σ·n̂)φ_i. Map those rows to the
    boundary P2 nodes and (optionally) solve the consistent P2 boundary mass to get
    a pointwise σ_nn. Returned mean-removed (the ρg·h gauge).
    """
    dm = solver.dm
    dim = solver.mesh.dim
    Q = info["Q"]; A = info["A"]; b = info["b"]; U = info["U"]
    # Cartesian residual r_c = A u − b, rotated: r = Q r_c
    rc = A.createVecLeft(); A.mult(U, rc); rc.axpy(-1.0, b)
    r = rc.duplicate(); Q.mult(rc, r)
    ra = r.getArray()

    lsec = dm.getLocalSection(); l2g = dm.getLGMap(); VEL = _velocity_field_id(solver)
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)
    normal = dict((nm, nrm) for nm, nrm in
                  [(s if isinstance(s, tuple) else (s, None)) for s in info["boundaries"]]).get(boundary)
    nodes = _boundary_velocity_nodes(solver, boundary, normal=normal)
    # each face node: normal row = the frame row spanning its (single) normal = grows[0]
    xs = []; Rn = []; pts = []
    for q, nrm in nodes:
        lo = lsec.getFieldOffset(q, VEL)
        row = int(l2g.apply([lo])[0])            # frame row 0 = normal (single-boundary face)
        if row < 0:
            continue
        xs.append(_point_coord(dm, dim, cvec, csec, v0, v1, q))
        Rn.append(ra[row]); pts.append(q)
    xs = np.array(xs); Rn = np.array(Rn)
    if not consistent_mass or dim != 2:
        sig = -Rn
        return xs, sig - sig.mean()
    # 2D consistent P2 boundary mass from DMPlex top-edge connectivity
    sig = _consistent_mass_2d(solver, boundary, normal, pts, Rn)
    return xs, sig


def _consistent_mass_2d(solver, boundary, normal, pts, Rn):
    """Solve M_Γ σ = −R with the consistent P2 line mass on `boundary` (2D)."""
    dm = solver.dm; dim = 2
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0); e0, e1 = dm.getDepthStratum(1)
    def vcoord(q): return cvec[csec.getOffset(q) // dim]
    bval = [b.value for b in solver.mesh.boundaries if b.name == boundary][0]
    edges = [q for q in (int(z) for z in dm.getStratumIS(boundary, bval).getIndices())
             if e0 <= q < e1]
    idx = {p: k for k, p in enumerate(pts)}
    n = len(pts); M = np.zeros((n, n))
    for e in edges:
        a, b = (int(c) for c in dm.getCone(e))
        if a not in idx or b not in idx or e not in idx:
            continue
        h = float(np.hypot(*(vcoord(b) - vcoord(a))))
        Me = (h / 30.0) * np.array([[4., 2, -1], [2, 16, 2], [-1, 2, 4]])
        tri = [idx[a], idx[e], idx[b]]
        for ii in range(3):
            for jj in range(3):
                M[tri[ii], tri[jj]] += Me[ii, jj]
    sig = np.linalg.solve(M, -np.asarray(Rn))
    return sig - sig.mean()


def _remove_rotation_gauge(solver):
    """Project the rigid-body rotation t=(-y,x) out of the converged velocity
    (Cartesian). Purely tangential → cannot introduce wall throughflow."""
    v = solver.Unknowns.u
    c = v.coords
    t = np.column_stack([-c[:, 1], c[:, 0]])
    coef = np.sum(v.data * t) / np.sum(t * t)
    v.data[...] = v.data - coef * t
    return coef
