"""Development version of underworld3.utilities.rotated_bc — reusable rotated
strong free-slip for the Stokes saddle. Productizes the validated prototypes:
build a per-node rotation Q from boundary normals, rotate the assembled saddle
Â=Q A Qᵀ / b̂=Q b, impose v_n=0 on the rotated normal rows, solve, rotate back
u=Qᵀû, remove the rigid-rotation gauge, and expose σ_nn as the constraint reaction.

Increment 1: box-flat (Q=identity on axis-aligned walls) must reproduce the native
essential free-slip solve bit-for-bit. Direct LU here; FMG wiring is a later step.
"""
import numpy as np
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
    # In parallel a rank may own NO part of this boundary → getStratumIS returns a
    # null IS; calling getIndices() on it segfaults. Guard and return no local nodes.
    sis = dm.getStratumIS(boundary, bval)
    if sis is None or sis.handle == 0:
        return []
    facets = [int(z) for z in sis.getIndices()]
    fS, fE = dm.getHeightStratum(1)              # facets (edges in 2D, faces in 3D)

    def coord(q):
        return _point_coord(dm, dim, cvec, csec, v0, v1, q)

    # analytic / constant normal specs. An analytic (sympy) normal is LAMBDIFIED
    # once into a fast numpy callable — per-node sympy .subs() is orders of magnitude
    # slower and, in parallel, serialises on the rank that owns the boundary (the
    # others idle), which looked like a hang.
    sym_fn = None
    const_normal = None
    if normal is not None:
        try:
            import sympy
            if isinstance(normal, sympy.Matrix):
                sym_fn = sympy.lambdify(list(solver.mesh.X),
                                        [normal[0, k] for k in range(dim)], "numpy")
        except Exception:
            sym_fn = None
        if sym_fn is None:
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
                if sym_fn is not None:
                    cq = coord(q)
                    ne = np.asarray(sym_fn(*cq), dtype=float).ravel()
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
    lsec = dm.getLocalSection()
    l2g = dm.getLGMap()
    VEL = _velocity_field_id(solver)
    dim = solver.mesh.dim

    # gather all normals per velocity node across the boundaries. Each entry of
    # `boundaries` is a name (geometric normal) or a (name, normal) pair.
    node_normals = {}
    for spec in boundaries:
        name, normal = spec if isinstance(spec, tuple) else (spec, None)
        for q, nrm in _boundary_velocity_nodes(solver, name, normal=normal):
            node_normals.setdefault(q, []).append(nrm)

    # Distributed Q with the assembled operator's ROW layout. Q is identity except a
    # per-node dim×dim orthonormal block; because a node's dim velocity components
    # live on a SINGLE DMPlex point owned by ONE rank, each block is entirely within
    # that rank's diagonal portion — no off-rank columns. Each rank sets ONLY its
    # owned rows (ghost copies of a shared boundary node are skipped: their global
    # rows fall outside [rstart, rend)).
    A = solver.snes.getJacobian()[0]
    rstart, rend = A.getOwnershipRange()
    nloc = rend - rstart
    N = A.getSize()[0]
    Q = PETSc.Mat().create(comm=dm.comm)
    Q.setSizes(((nloc, N), (nloc, N)))
    Q.setType("aij")
    Q.setPreallocationNNZ((dim, 0))
    Q.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    for i in range(rstart, rend):
        Q.setValue(i, i, 1.0)                    # identity default (owned rows)

    normal_rows = []
    for q, nrms in node_normals.items():
        lo = lsec.getFieldOffset(q, VEL)
        grows = [int(l2g.apply([lo + c])[0]) for c in range(dim)]
        if any(g < 0 for g in grows):
            continue
        if not (rstart <= grows[0] < rend):      # not owned by this rank → skip
            continue
        # DIMENSION-GENERAL constraint frame: the accumulated normals span a rank-r
        # subspace the velocity must be orthogonal to (r=1 face → constrain v_n;
        # r=2 edge → constrain 2, free the edge tangent; r=dim corner → v=0). The SVD
        # right vectors are a complete orthonormal frame E; Q's node block rows = E
        # (rotated component i = E[i]·v); constrain the first r rotated rows.
        M = np.array(nrms, dtype=float)
        _, sv, Vt = np.linalg.svd(M)
        r = int((sv > 1e-8 * (sv[0] if sv.size else 1.0)).sum())
        for i in range(dim):
            for j in range(dim):
                Q.setValue(grows[i], grows[j], float(Vt[i, j]))
        normal_rows.extend(grows[:r])            # constrain the r normal-space rows
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

    # Assemble the operator FIRST so its parallel row layout is final before we build
    # Q against it (A = exact Jacobian at 0 — linear; b = -F(0)).
    snes.setUp()
    U0 = dm.getGlobalVec(); U0.set(0.0)
    J = snes.getJacobian()[0]; snes.computeJacobian(U0, J)
    Aorig = J.copy()
    F0 = dm.getGlobalVec(); snes.computeFunction(U0, F0)
    b = F0.copy(); b.scale(-1.0)

    Q, Qt, normal_rows = build_rotation(solver, boundaries)

    # rotate: Â = Q A Qᵀ, b̂ = Q b
    Ahat = Aorig.ptap(Qt)
    bhat = b.duplicate(); Q.mult(b, bhat)

    # pin one pressure DOF (datum) — row only, keeps B^T coupling
    gsec = dm.getGlobalSection()
    PRE = 1; pin = None; pS, pE = gsec.getChart()
    for q in range(pS, pE):
        if gsec.getFieldDof(q, PRE) > 0 and gsec.getFieldOffset(q, PRE) >= 0:
            pin = gsec.getFieldOffset(q, PRE); break

    # constrain rotated normal rows (v_n=0): zero the matrix rows/cols (identity
    # diagonal) AND the RHS at those rows — zeroRowsColumns does NOT touch the RHS,
    # so a nonzero b there would leak straight into the solution (û_i = b_i / 1),
    # independent of the solver/tolerance.
    Ahat.zeroRowsColumns(normal_rows, diag=1.0)
    ba = bhat.getArray(); ba[normal_rows] = 0.0; bhat.setArray(ba)

    # ITERATIVE by default (LU is almost never right): a self-contained fieldsplit-
    # Schur solve whose velocity block is geometric FMG on the custom prolongation
    # when a hierarchy is registered (set_custom_fmg), else GAMG. Direct LU only when
    # explicitly opted in via solver._rotated_use_lu.
    if getattr(solver, "_rotated_use_lu", False):
        Ahat.zeroRows([pin], diag=1.0)
        ba = bhat.getArray(); ba[pin] = 0.0; bhat.setArray(ba)
        ksp = PETSc.KSP().create(); ksp.setOperators(Ahat); ksp.setType("preonly")
        pc = ksp.getPC(); pc.setType("lu"); pc.setFactorSolverType("mumps")
        Uhat = dm.getGlobalVec(); ksp.solve(bhat, Uhat)
        ksp_reason = ksp.getConvergedReason()
    else:
        Uhat, ksp_reason = _solve_rotated_iterative(
            solver, Ahat, bhat, Q, Qt, normal_rows, verbose=verbose)

    # rotate back u = Qᵀ û
    U = dm.getGlobalVec(); Qt.mult(Uhat, U)

    # Remove the rigid-rotation gauge ONLY when it is a genuine null space of the
    # constrained problem (closed circular/spherical free-slip); on straight walls
    # the constraint pins the rotation, and projecting would corrupt the solution.
    # Done on the GLOBAL vector U with PETSc dots (parallel-correct ownership — a
    # local nodal sum would double-count shared nodes at rank boundaries).
    removed = False
    if remove_rotation_gauge and _rotation_is_nullspace(solver, Q, normal_rows):
        tg = _rigid_rotation_global(solver)
        coef = U.dot(tg) / (tg.dot(tg) + 1e-30)
        U.axpy(-coef, tg)
        removed = True

    # scatter U → velocity/pressure fields
    for name, var in solver.fields.items():
        sg = U.getSubVector(solver._subdict[name][0])
        solver._subdict[name][1].globalToLocal(sg, var.vec)
        U.restoreSubVector(solver._subdict[name][0], sg)

    return {"Q": Q, "Qt": Qt, "A": Aorig, "b": b, "U": U, "Uhat": Uhat,
            "normal_rows": normal_rows, "boundaries": list(boundaries),
            "rotation_gauge_removed": removed, "ksp_reason": ksp_reason}


def _solve_rotated_iterative(solver, Ahat, bhat, Q, Qt, normal_rows, verbose=False):
    """Solve the rotated saddle with a SELF-CONTAINED fieldsplit-Schur KSP on the
    rotated operator. The velocity block is geometric FMG on the CUSTOM prolongation
    (PR#290, rotated) when a hierarchy is registered (``set_custom_fmg``), else GAMG.

    A plain rotated Mat has no DM field info, so UW3's DM-coupled fieldsplit cannot
    split it — we build the split from EXPLICIT velocity/pressure index sets. For the
    custom-FMG case the velocity sub-PC gets our prolongation via ``setMGInterpolation``
    (needs no DM); the rotated block A_vv = Q_v A_vv Q_vᵀ is formed from Âhat
    automatically and only the FINE prolongation is rotated (Galerkin coarse ops
    auto-correct). NO direct solve of the fine system."""
    from underworld3.utilities import custom_mg
    dm = solver.dm
    vel_is = solver._subdict["velocity"][0]
    pres_is = solver._subdict["pressure"][0]

    custom_Pl = None
    if getattr(solver, "_custom_mg", None) is not None:
        vis = np.asarray(vel_is.getIndices())
        g2blk = {int(g): k for k, g in enumerate(vis)}
        Qv = Q.createSubMatrix(vel_is, vel_is)
        nrows_blk = sorted({g2blk[g] for g in normal_rows if g in g2blk})
        Ps = solver._custom_mg["hierarchy"].build(solver)
        Pfine = Qv.matMult(Ps[-1]); Pfine.zeroRows(nrows_blk, diag=0.0)
        custom_Pl = list(Ps[:-1]) + [Pfine]

    # rotated coupled null space (pressure-const ⊕ Q·rotation) on the operator
    nsp = _rotated_nullspace(solver, Q, normal_rows)
    if nsp is not None:
        Ahat.setNullSpace(nsp); Ahat.setTransposeNullSpace(nsp); nsp.remove(bhat)

    pfx = "rotfs_"
    opts = PETSc.Options()
    cfg = {
        "ksp_type": "fgmres", "ksp_rtol": str(float(solver.tolerance)), "ksp_max_it": "300",
        "pc_type": "fieldsplit", "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full", "pc_fieldsplit_schur_precondition": "selfp",
        "fieldsplit_vel_ksp_type": "preonly",
        "fieldsplit_pres_ksp_type": "fgmres", "fieldsplit_pres_ksp_rtol": "1e-6",
        "fieldsplit_pres_ksp_max_it": "200", "fieldsplit_pres_pc_type": "jacobi",
    }
    if custom_Pl is None:                             # GAMG velocity block
        cfg["fieldsplit_vel_pc_type"] = "gamg"
    for k, v in cfg.items():
        opts[pfx + k] = v
    ksp = PETSc.KSP().create(comm=dm.comm); ksp.setOptionsPrefix(pfx)
    ksp.setOperators(Ahat)
    pc = ksp.getPC(); pc.setType("fieldsplit")
    pc.setFieldSplitIS(("vel", vel_is), ("pres", pres_is))
    ksp.setFromOptions()
    pc.setUp()
    if custom_Pl is not None:                         # geometric FMG via custom P
        vel_pc = pc.getFieldSplitSubKSP()[0].getPC()
        A_vv, P_vv = vel_pc.getOperators()
        vel_pc.reset(); vel_pc.setOperators(A_vv, P_vv)
        custom_mg._configure_pcmg(vel_pc, custom_Pl)
        vel_pc.setUp()

    Uhat = Ahat.createVecRight(); Uhat.set(0.0)
    ksp.solve(bhat, Uhat)
    # An identity constraint row in an ITERATIVE solve only drives its residual
    # (= û_i) below tolerance, so û_i ~ tol, not exactly 0. Because zeroRowsColumns
    # made these DOFs fully decoupled (row AND column zeroed), û_i affects no other
    # equation → setting them to exactly 0 here makes the strong v_n=0 BC exact
    # independent of the iterative tolerance, without perturbing the rest.
    rs, re = Uhat.getOwnershipRange()
    loc = np.asarray([g - rs for g in normal_rows if rs <= g < re], dtype=np.int64)
    ua = Uhat.getArray(); ua[loc] = 0.0; Uhat.setArray(ua)
    if verbose:
        from underworld3 import mpi
        kind = "custom-FMG" if custom_Pl is not None else "GAMG"
        mpi.pprint(f"[rotated_bc] velocity block = {kind}; outer KSP "
                   f"{ksp.getConvergedReason()} in {ksp.getIterationNumber()} its")
    return Uhat, ksp.getConvergedReason()


def _rotated_nullspace(solver, Q, normal_rows):
    """Coupled Stokes null space in the rotated frame: constant pressure, plus the
    rigid rotation Q·(-y,x) when it is a genuine null space of the constraints.
    Returns a PETSc.NullSpace on the composite vector, or None.

    Each vector is ZEROED at the constrained normal rows so it is exactly compatible
    with the strong v_n=0 constraint. Without this, a rotation-mode vector that is
    only ~O(h²)-small at the constrained rows (curved boundary: the normal in Q is
    sampled at the chord midpoint, the rotation at the true P2 node) would, when
    PETSc projects the solution onto the null space, inject a spurious wall-normal
    component — a strong BC must be exact, independent of the iterative solve.
    """
    dm = solver.dm
    vecs = []
    # constant pressure (Q = identity on pressure → unchanged)
    if getattr(solver, "_petsc_use_pressure_nullspace", False):
        pv = dm.getGlobalVec(); pv.set(0.0)
        pis = solver._subdict["pressure"][0]
        sp = pv.getSubVector(pis); sp.set(1.0); pv.restoreSubVector(pis, sp)
        pv.normalize(); vecs.append(pv)
    # rigid rotation (rotated), only if it satisfies the constraints
    if solver.mesh.dim == 2 and _rotation_is_nullspace(solver, Q, normal_rows):
        tg = _rigid_rotation_global(solver)
        tr = tg.duplicate(); Q.mult(tg, tr)
        vecs.append(tr)
    if not vecs:
        return None
    # Make every null-space vector EXACTLY compatible with the strong v_n=0
    # constraint (zero at the constrained rows), then orthonormalise. This is what
    # keeps the wall-normal velocity exact under an iterative solve.
    rs, re = vecs[0].getOwnershipRange()
    loc = np.asarray([g - rs for g in normal_rows if rs <= g < re], dtype=np.int64)
    for w in vecs:
        wa = w.getArray(); wa[loc] = 0.0; w.setArray(wa)
    ortho = []
    for w in vecs:
        for u in ortho:
            w.axpy(-w.dot(u), u)
        nrm = w.norm()
        if nrm > 1e-14:
            w.scale(1.0 / nrm); ortho.append(w)
    return PETSc.NullSpace().create(constant=False, vectors=ortho, comm=dm.comm)


def _rotation_is_nullspace(solver, Q, normal_rows, tol=1e-8):
    """True iff the rigid-body rotation t=(-y,x) satisfies all rotated v_n=0
    constraints — i.e. Q·t is ~0 on every constrained normal row.

    COLLECTIVE: every rank runs the same global-vector ops. Do NOT early-return on
    a per-rank ``not normal_rows`` — in parallel a rank may own no boundary node
    (empty normal_rows) while others do, and an early return there would desync the
    collective norms below and deadlock."""
    if solver.mesh.dim != 2:
        return False
    tg = _rigid_rotation_global(solver)
    tr = tg.duplicate(); Q.mult(tg, tr)
    full = tr.norm()                              # parallel norm
    # norm of tr restricted to the constrained rows: zero everything else, then .norm()
    rs, re = tr.getOwnershipRange()
    loc = np.asarray([g - rs for g in normal_rows if rs <= g < re], dtype=np.int64)
    trc = tr.duplicate(); trc.set(0.0)
    tra = trc.getArray(); tga = tr.getArray()
    tra[loc] = tga[loc]; trc.setArray(tra)
    viol = trc.norm() / (full + 1e-30)            # parallel (collective on all ranks)
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
    sis = dm.getStratumIS(boundary, bval)
    strat = [] if (sis is None or sis.handle == 0) else [int(z) for z in sis.getIndices()]
    edges = [q for q in strat if e0 <= q < e1]
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


def _rigid_rotation_global(solver):
    """The Cartesian rigid-body rotation t=(-y,x) as a composite GLOBAL vector
    (velocity DOFs only, zero pressure). Parallel-safe: built via localToGlobal on
    the velocity sub-DM, so shared nodes are handled by PETSc, not double-counted."""
    dm = solver.dm
    v = solver.Unknowns.u
    c = v.coords
    saved = v.data.copy()
    v.data[...] = np.column_stack([-c[:, 1], c[:, 0]])
    tg = dm.getGlobalVec(); tg.set(0.0)
    vis = solver._subdict["velocity"][0]
    sg = tg.getSubVector(vis)
    solver._subdict["velocity"][1].localToGlobal(v.vec, sg)
    tg.restoreSubVector(vis, sg)
    v.data[...] = saved
    return tg
