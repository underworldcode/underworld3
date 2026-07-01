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
from mpi4py import MPI

# Monotonic counter so each rotated solve gets a UNIQUE PETSc options prefix. With a
# fixed prefix, sequential rotated solves (e.g. two solvers in one script, or one
# solver in a time-stepping loop) share and re-set the same global-options keys; the
# per-solve prefix plus the delValue cleanup below keeps each solve's options local
# and stops the global database growing / emitting "unused option" warnings.
_ROT_SOLVE_COUNT = 0


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
    # zeroRowsColumns takes GLOBAL row indices (correct); the RHS write must use
    # OWNERSHIP-RELATIVE local indices (bhat.getArray() is this rank's local slice,
    # so indexing it with global rows overflows on any rank whose ownership does not
    # start at 0 — the np>1 crash that masqueraded as a hang).
    Ahat.zeroRowsColumns(normal_rows, diag=1.0)
    brs, bre = bhat.getOwnershipRange()
    bloc = np.asarray([g - brs for g in normal_rows if brs <= g < bre], dtype=np.int64)
    ba = bhat.getArray(); ba[bloc] = 0.0; bhat.setArray(ba)

    # ITERATIVE by default (LU is almost never right): a self-contained fieldsplit-
    # Schur solve whose velocity block is geometric FMG on the custom prolongation
    # when a hierarchy is registered (set_custom_fmg), else GAMG. Direct LU only when
    # explicitly opted in via solver._rotated_use_lu.
    if getattr(solver, "_rotated_use_lu", False):
        # NOTE: the pressure `pin` is a naive per-rank global search (parallel-unsafe;
        # LU is opt-in only — see the follow-up in rotated_bc). The RHS write below
        # still uses ownership-relative indexing so it does not overflow the local slice.
        Ahat.zeroRows([pin], diag=1.0)
        if pin is not None and brs <= pin < bre:
            ba = bhat.getArray(); ba[pin - brs] = 0.0; bhat.setArray(ba)
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

    # Parity with the normal solve's post-scatter sync (pyx: after the field copy-back):
    # refresh the enhanced-variable gvec cache and drop the canonical-data cache so
    # downstream consumers (var.data / var.array / checkpoint / stats) don't read a
    # stale value; and mark the mesh local vector stale.
    solver.mesh._stale_lvec = True
    for name, var in solver.fields.items():
        target_var = getattr(var, "_base_var", var)
        if hasattr(target_var, "_sync_lvec_to_gvec"):
            target_var._sync_lvec_to_gvec()
        if hasattr(target_var, "_canonical_data"):
            target_var._canonical_data = None

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

    # UNIQUE prefix per solve (see _ROT_SOLVE_COUNT) so sequential rotated solves
    # do not share global-options state; the keys are removed after the solve.
    global _ROT_SOLVE_COUNT
    _ROT_SOLVE_COUNT += 1
    pfx = f"rotfs{_ROT_SOLVE_COUNT}_"
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
    try:
        ksp = PETSc.KSP().create(comm=dm.comm); ksp.setOptionsPrefix(pfx)
        ksp.setOperators(Ahat)
        pc = ksp.getPC(); pc.setType("fieldsplit")
        pc.setFieldSplitIS(("vel", vel_is), ("pres", pres_is))
        ksp.setFromOptions()
        pc.setUp()
        if custom_Pl is not None:                     # geometric FMG via custom P
            vel_pc = pc.getFieldSplitSubKSP()[0].getPC()
            A_vv, P_vv = vel_pc.getOperators()
            vel_pc.reset(); vel_pc.setOperators(A_vv, P_vv)
            custom_mg._configure_pcmg(vel_pc, custom_Pl)
            vel_pc.setUp()

        Uhat = Ahat.createVecRight(); Uhat.set(0.0)
        ksp.solve(bhat, Uhat)
    finally:
        # all options consumed by setFromOptions/setUp/solve — drop them so the
        # global database stays clean (and bounded under time-stepping).
        for k in cfg:
            try:
                opts.delValue(pfx + k)
            except Exception:
                pass
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


def boundary_normal_traction(solver, boundary, info, mass="lumped"):
    """Boundary normal traction σ_nn on `boundary` from the constraint reaction of the
    last rotated-free-slip solve. Returned mean-removed (the ρg·h gauge), as
    ``(xs, sigma)`` with one entry per boundary velocity node on this rank.

    σ_nn is recovered from the CARTESIAN nodal reaction r_c = A·u − b: the nodal load
    is R_i = n̂_i · r_c(node_i), where n̂_i is THIS boundary's outward normal at node i.
    Projecting the Cartesian reaction onto the boundary normal (rather than reading the
    rotated frame's normal row) is corner-correct — at a node shared with another
    rotated-free-slip boundary the rotated frame's first row is a mix of both walls'
    normals, but n̂·r_c is the true normal traction for this boundary. The pointwise
    σ_nn is the boundary-mass de-smear of R (2D).

    ``mass`` selects the de-smear:
      * ``"lumped"`` (default) — the diagonal (row-sum) boundary mass. Being an M-matrix
        it CANNOT overshoot at a stress discontinuity (no Gibbs wiggle where the traction
        jumps, e.g. across a viscosity contrast), it is a purely local division (no global
        mass solve → trivially parallel), and it is marginally more accurate than the
        consistent mass on SolCx. Recommended for driving a free surface, where an
        overshoot at a sharp feature injects a spurious surface-velocity pulse.
      * ``"consistent"`` — the full consistent P2 line mass. Marginally sharper on smooth
        tractions but overshoots at discontinuities.

    Parallel-safe: r_c is scattered to a local vector (ghosts included) and read by LOCAL
    section offset; the boundary mass is assembled globally by a coordinate-keyed
    allgather of the boundary elements, so every rank produces the same de-smear and the
    mean-removal gauge is global.
    """
    dm = solver.dm
    dim = solver.mesh.dim
    A = info["A"]; b = info["b"]; U = info["U"]
    # Cartesian nodal reaction r_c = A u − b (global), scattered to local incl. ghosts
    rc = A.createVecLeft(); A.mult(U, rc); rc.axpy(-1.0, b)
    rcl = dm.getLocalVec(); dm.globalToLocal(rc, rcl); rca = np.asarray(rcl.getArray())

    lsec = dm.getLocalSection(); VEL = _velocity_field_id(solver)
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)
    normal = dict((nm, nrm) for nm, nrm in
                  [(s if isinstance(s, tuple) else (s, None)) for s in info["boundaries"]]).get(boundary)
    nodes = _boundary_velocity_nodes(solver, boundary, normal=normal)
    xs = []; Rn = []; pts = []
    for q, nrm in nodes:
        lo = lsec.getFieldOffset(q, VEL)
        rcv = rca[lo:lo + dim]                    # Cartesian reaction at this node (local)
        xs.append(_point_coord(dm, dim, cvec, csec, v0, v1, q))
        Rn.append(float(np.dot(nrm, rcv)))        # R_i = n̂·r_c  (corner-correct)
        pts.append(q)
    dm.restoreLocalVec(rcl)
    xs = np.array(xs); Rn = np.array(Rn)
    if dim != 2:
        # no line-mass geometry in 3D yet → crude global-mean-removed load
        comm = dm.comm.tompi4py()
        tot = comm.allreduce(float(Rn.sum()), op=MPI.SUM)
        cnt = comm.allreduce(int(Rn.size), op=MPI.SUM)
        return xs, (-Rn) - (-tot / max(cnt, 1))
    sig = _recover_sigma_nn_2d(solver, boundary, pts, Rn, xs, mass=mass)
    return xs, sig


def dynamic_topography_field(solver, boundary, info, field, buoyancy_scale=1.0, mass="lumped"):
    """Populate a scalar MeshVariable ``field`` with the dynamic topography
    :math:`h = -(\\sigma_{nn}-\\overline{\\sigma_{nn}})/(\\Delta\\rho\\,g)` on ``boundary``,
    recovered from the rotated-free-slip constraint reaction (lumped by default —
    monotone, no Gibbs overshoot at a stress jump). Interior nodes are left untouched.
    Returns ``field``.

    This is the hand-off to the free-surface machinery: the 3-number topography
    integrator drives node motion from a surface field, so σ_nn is written onto the
    boundary nodes of a P1 (or higher) scalar field it can read / BdIntegral. Parallel-
    safe: the recovery is partition-independent and the write is local (each rank fills
    its own boundary nodes, matched by coordinate to the recovery output).
    """
    dim = solver.mesh.dim
    xs, sig = boundary_normal_traction(solver, boundary, info, mass=mass)

    def key(c):
        return tuple(round(float(t), 9) for t in np.asarray(c).ravel()[:dim])

    # σ_nn is already mean-removed (the ρg·h gauge); topography h = -σ_nn / (Δρ g)
    hmap = {key(x): -float(s) / buoyancy_scale for x, s in zip(np.asarray(xs), np.asarray(sig))}
    fc = np.asarray(field.coords)
    # Build the new nodal values in a LOCAL numpy copy, then assign the field ONCE.
    # A per-element write to var.data fires the variable's write-callback each time; the
    # number of boundary nodes differs per rank (a rank may own none of the boundary),
    # so per-element writes would desync any collective in the callback and deadlock.
    newdata = np.asarray(field.data).copy()
    for i in range(fc.shape[0]):
        h = hmap.get(key(fc[i]))
        if h is not None:
            newdata[i, 0] = h
    field.data[...] = newdata
    # refresh the field's gvec cache so symbolic/BdIntegral reads see the update
    base = getattr(field, "_base_var", field)
    if hasattr(base, "_sync_lvec_to_gvec"):
        base._sync_lvec_to_gvec()
    if hasattr(base, "_canonical_data"):
        base._canonical_data = None
    solver.mesh._stale_lvec = True
    return field


def _recover_sigma_nn_2d(solver, boundary, pts, Rn, xs, mass="lumped"):
    """De-smear the nodal reaction loads R into a pointwise σ_nn on `boundary` (2D) with
    either the LUMPED (diagonal, monotone) or the CONSISTENT P2 line mass.

    Parallel-safe by construction: each rank emits its local boundary ELEMENTS as
    self-contained coordinate-keyed records (the three P2 node keys + the element length);
    an allgather assembles the SAME global boundary mass on every rank (elements
    de-duplicated by key, so a facet shared across a partition cut is counted once). Every
    rank forms the identical de-smear and returns σ at its own local nodes, so the result
    — and the mean-removal gauge — is partition-independent."""
    dm = solver.dm; dim = 2; comm = dm.comm.tompi4py()
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0); e0, e1 = dm.getDepthStratum(1)
    def vcoord(q): return cvec[csec.getOffset(q) // dim]
    def key(c): return (round(float(c[0]), 9), round(float(c[1]), 9))

    # local node -> reaction load, keyed by coordinate (ghosts included via pts/Rn)
    nodeR = {key(x): float(R) for x, R in zip(xs, Rn)}

    # local boundary elements as coordinate-keyed records: (ka, kmid, kb, h)
    bval = [bb.value for bb in solver.mesh.boundaries if bb.name == boundary][0]
    sis = dm.getStratumIS(boundary, bval)
    strat = [] if (sis is None or sis.handle == 0) else [int(z) for z in sis.getIndices()]
    edges = [q for q in strat if e0 <= q < e1]
    local_elems = []
    for e in edges:
        a, bb = (int(c) for c in dm.getCone(e))
        ca, cb = vcoord(a), vcoord(bb)
        cmid = _point_coord(dm, dim, cvec, csec, v0, v1, e)
        h = float(np.hypot(*(cb - ca)))
        local_elems.append((key(ca), key(cmid), key(cb), h))

    # gather elements + node loads across ranks (1D boundary → cheap)
    all_elems = comm.allgather(local_elems)
    all_nodeR = comm.allgather(nodeR)
    R_by_key = {}
    for d in all_nodeR:
        R_by_key.update(d)                       # same key → same value on every rank
    # de-duplicate elements (a shared facet appears on >1 rank)
    uniq = {}
    for lst in all_elems:
        for (ka, km, kb, h) in lst:
            uniq[(ka, km, kb)] = h
    # global node numbering
    keys = sorted(R_by_key.keys())
    gidx = {k: i for i, k in enumerate(keys)}
    n = len(keys); R = np.zeros(n)
    for k, i in gidx.items():
        R[i] = R_by_key[k]

    if mass == "lumped":
        # diagonal P2 line mass (row sums of the consistent mass): h*[1/6, 2/3, 1/6] for
        # (vertexA, mid-edge, vertexB). Monotone → no overshoot at a stress jump.
        mL = np.zeros(n)
        for (ka, km, kb), h in uniq.items():
            mL[gidx[ka]] += h / 6.0
            mL[gidx[km]] += 2.0 * h / 3.0
            mL[gidx[kb]] += h / 6.0
        sig_g = -R / mL
    else:
        # consistent P2 line mass M σ = −R
        M = np.zeros((n, n))
        Me = np.array([[4., 2, -1], [2, 16, 2], [-1, 2, 4]])
        for (ka, km, kb), h in uniq.items():
            tri = [gidx[ka], gidx[km], gidx[kb]]
            Mh = (h / 30.0) * Me
            for ii in range(3):
                for jj in range(3):
                    M[tri[ii], tri[jj]] += Mh[ii, jj]
        sig_g = np.linalg.solve(M, -R)
    sig_g = sig_g - sig_g.mean()                 # global gauge → partition-independent
    # return σ at THIS rank's local nodes, in the input (xs/pts) order
    return np.array([sig_g[gidx[key(x)]] for x in xs])


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
