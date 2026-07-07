"""Rotated strong free-slip for the Stokes saddle (the implementation behind
``solver.add_rotated_freeslip_bc``): build a per-node rotation Q from boundary
normals, rotate the assembled saddle Â=Q A Qᵀ / b̂=Q b, impose v_n=0 on the
rotated normal rows, solve, rotate back u=Qᵀû, remove the rigid-rotation gauge,
and expose σ_nn as the constraint reaction.

The rotated saddle is solved by a self-contained fieldsplit-Schur KSP by default: the
velocity block is geometric FMG on the custom prolongation (``set_custom_fmg``) when a
hierarchy is registered (the PREFERRED route), else GAMG tuned to the native path's
settings; the Schur complement is preconditioned by the native 1/mu pressure mass
(the Pmat p-p block, exactly the standard path's ``schur_precondition=a11``), with a
constant-pressure nullspace on the inner Schur solve for enclosed domains; direct
MUMPS LU is opt-in via ``solver._rotated_use_lu``.
σ_nn / dynamic topography reuse the shared Consistent-Boundary-Flux de-smear in
``underworld3.utilities.boundary_flux``.
"""
import numpy as np
from petsc4py import PETSc

# Shared Consistent-Boundary-Flux machinery (the σ_nn recovery is a rotated-frame reading
# of the same primitive): the consolidated-label boundary stratum, the lumped/consistent
# boundary-mass de-smear, and the scalar-field hand-off all live in `boundary_flux`.
from underworld3.utilities.boundary_flux import (
    _boundary_stratum_is, _desmear, write_boundary_scalar_field)

# Monotonic counter so each rotated solve gets a UNIQUE PETSc options prefix. With a
# fixed prefix, sequential rotated solves (e.g. two solvers in one script, or one
# solver in a time-stepping loop) share and re-set the same global-options keys; the
# per-solve prefix plus the delValue cleanup below keeps each solve's options local
# and stops the global database growing / emitting "unused option" warnings.
_ROT_SOLVE_COUNT = 0


def _warn_if_ksp_diverged(ksp, kind):
    """Emit a rank-0 warning if a KSP finished on a KSP_DIVERGED_* reason
    (negative converged-reason). ``ksp.solve`` never raises on divergence, so
    without this the linear rotated-freeslip solvers here would return a
    partial answer to the caller silently — which is exactly what let the 3D
    rotation-nullspace bug (#306) go unnoticed until it produced clearly-wrong
    physics in a downstream test."""
    reason = int(ksp.getConvergedReason())
    # Convention: > 0 == converged, < 0 == diverged, 0 == KSP_CONVERGED_ITERATING
    # (should not happen after ksp.solve() returns; warn if it does). Matches
    # petsc_generic_snes_solvers.pyx:2979.
    if reason > 0:
        return
    its = int(ksp.getIterationNumber())
    try:
        rnorm = float(ksp.getResidualNorm())
    except Exception:
        rnorm = float("nan")
    from underworld3 import mpi
    mpi.pprint(f"[rotated_bc] WARNING: {kind} KSP did NOT converge "
               f"(reason={reason}, iterations={its}, |r|={rnorm:.3e}); "
               f"proceeding with the last iterate.")


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
    # Boundary facets via the consolidated "UW_Boundaries" label (per-boundary labels do
    # not survive mesh adaptation); raises a clear error for an unknown boundary name.
    # In parallel a rank may own NO part of this boundary → a null IS; guard and return
    # no local nodes (calling getIndices() on a null IS would segfault).
    sis = _boundary_stratum_is(dm, solver.mesh, boundary)
    if not (sis and sis.getSize() > 0):
        return []
    facets = [int(z) for z in sis.getIndices()]
    fS, fE = dm.getHeightStratum(1)              # facets (edges in 2D, faces in 3D)

    def coord(q):
        return _point_coord(dm, dim, cvec, csec, v0, v1, q)

    # analytic / constant normal specs. An analytic (sympy) normal is LAMBDIFIED
    # once into a fast numpy callable — per-node sympy .subs() is orders of magnitude
    # slower and, in parallel, serialises on the rank that owns the boundary (the
    # others idle), which looked like a hang. Components are UNWRAPPED first so a
    # normal written in curvilinear terms (e.g. ``mesh.CoordinateSystem.unit_e_0``,
    # whose UWexpressions hide the Cartesian coordinates behind ``r`` etc.) reduces
    # to pure mesh.X before lambdify — otherwise the generated function references
    # the curvilinear symbol by bare name and dies with a NameError at call time.
    sym_fn = None
    const_normal = None
    if normal is not None:
        try:
            import sympy
            if isinstance(normal, sympy.Matrix):
                from underworld3.function.expressions import unwrap
                comps = [sympy.sympify(unwrap(normal[0, k], keep_constants=False,
                                              return_self=False))
                         for k in range(dim)]
                stray = set().union(*[c.free_symbols for c in comps]) \
                    - set(solver.mesh.X)
                if stray:
                    raise ValueError(
                        f"analytic normal for boundary '{boundary}' contains "
                        f"symbols {sorted(map(str, stray))} that are not mesh "
                        "coordinates — express it in mesh.X (e.g. radial X/|X|).")
                sym_fn = sympy.lambdify(list(solver.mesh.X), comps, "numpy")
        except ValueError:
            raise
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
    # Q against it (A = exact Jacobian at 0 — linear; b = -F(0)). The Pmat is
    # assembled alongside: its p-p block is the native 1/mu pressure mass (the
    # DS JacobianPreconditioner term) that preconditions the Schur complement —
    # petsc4py's computeJacobian(x, J) would silently pass J as its own Pmat and
    # the mass block would never be assembled.
    snes.setUp()
    U0 = dm.getGlobalVec(); U0.set(0.0)
    J, Jp = snes.getJacobian()[:2]
    snes.computeJacobian(U0, J, Jp)
    Aorig = J.copy()
    F0 = dm.getGlobalVec(); snes.computeFunction(U0, F0)
    b = F0.copy(); b.scale(-1.0)
    dm.restoreGlobalVec(U0); dm.restoreGlobalVec(F0)   # borrowed temporaries → return to pool

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

    # constrain rotated normal rows (v_n=0): zero the matrix rows/cols AND the RHS
    # at those rows — zeroRowsColumns does NOT touch the RHS, so a nonzero b there
    # would leak straight into the solution (û_i = b_i / diag), independent of the
    # solver/tolerance.
    # The constraint diagonal is set to the mean |diag(A_vv)| rather than 1.0: unit
    # diagonals amid O(eta/h^2) viscous entries put a spectrum outlier on EVERY
    # rotated boundary node (the whole surface on a sphere), which poisons
    # diagonal-based Schur approximations and MG smoothing exactly in the boundary
    # strip. Any positive diagonal is exact — the solution rows are explicitly
    # zeroed after the solve.
    # zeroRowsColumns takes GLOBAL row indices (correct); the RHS write must use
    # OWNERSHIP-RELATIVE local indices (bhat.getArray() is this rank's local slice,
    # so indexing it with global rows overflows on any rank whose ownership does not
    # start at 0 — the np>1 crash that masqueraded as a hang).
    Ahat.zeroRowsColumns(normal_rows, diag=_velocity_diag_scale(Ahat, solver))
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
        Uhat = dm.createGlobalVec(); ksp.solve(bhat, Uhat)   # returned in info → own it
        ksp_reason = ksp.getConvergedReason(); ksp_its = ksp.getIterationNumber()
        _warn_if_ksp_diverged(ksp, kind="rotated direct-LU")
    else:
        Mp = _pressure_mass_schur_pmat(solver)
        Uhat, ksp_reason, ctx = _solve_rotated_iterative(
            solver, Ahat, bhat, Q, Qt, normal_rows, verbose=verbose, Mp=Mp)
        ksp_its = ctx["ksp"].getIterationNumber()
        _destroy_rotated_ksp_ctx(ctx)

    # rotate back u = Qᵀ û  (U is returned in info → create, don't borrow from the pool)
    U = dm.createGlobalVec(); Qt.mult(Uhat, U)

    removed = _finalize_rotated_solution(solver, U, Q, normal_rows, remove_rotation_gauge)

    return {"Q": Q, "Qt": Qt, "A": Aorig, "b": b, "U": U, "Uhat": Uhat,
            "normal_rows": normal_rows, "boundaries": list(boundaries),
            "rotation_gauge_removed": removed, "ksp_reason": ksp_reason,
            "ksp_its": ksp_its}


def _finalize_rotated_solution(solver, U, Q, normal_rows, remove_rotation_gauge):
    """Remove the rigid-rotation gauge (if it is a genuine null space of the
    constrained problem), scatter the composite global vector ``U`` into the
    velocity/pressure fields, and refresh the enhanced-variable caches. Shared by
    the linear one-shot and the nonlinear driver. Returns whether the gauge was
    removed."""
    dm = solver.dm
    # Remove the rigid-rotation gauge — every mode that is a genuine null space of
    # the constrained problem (closed circular free-slip: one; full spherical
    # shell: all three); on straight walls the constraint pins the rotations, and
    # projecting would corrupt the solution.
    # Done on the GLOBAL vector U with PETSc dots (parallel-correct ownership — a
    # local nodal sum would double-count shared nodes at rank boundaries). The
    # surviving modes are orthonormalised (Gram-Schmidt) before projection — the
    # three 3D modes are not mutually orthogonal on a general mesh.
    # COLLECTIVE: all ranks walk the same mode list, same order.
    removed = False
    if remove_rotation_gauge:
        live = []
        for tg in _rigid_rotation_modes(solver):
            if _mode_satisfies_constraints(solver, Q, normal_rows, tg):
                live.append(tg.copy())          # owned copy — pool vec goes back below
            dm.restoreGlobalVec(tg)
        ortho = []
        for w in live:
            for q in ortho:
                w.axpy(-w.dot(q), q)
            nrm = w.norm()
            if nrm > 1e-14:
                w.scale(1.0 / nrm); ortho.append(w)
            else:
                w.destroy()
        for q in ortho:
            U.axpy(-U.dot(q), q)
            q.destroy()
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
    return removed


def _zero_rows_local(vec, normal_rows):
    """Zero ``vec`` at the global rows ``normal_rows`` using ownership-relative
    local indices (indexing the local slice with global rows overflows on any rank
    whose ownership does not start at 0 — the np>1 crash class)."""
    rs, re = vec.getOwnershipRange()
    loc = np.asarray([g - rs for g in normal_rows if rs <= g < re], dtype=np.int64)
    a = vec.getArray(); a[loc] = 0.0; vec.setArray(a)


def _gather_fields_to_global(solver):
    """Composite global vector built from the solver's current velocity/pressure
    field values (the warm-start initial guess for the nonlinear driver)."""
    dm = solver.dm
    U = dm.createGlobalVec(); U.set(0.0)
    for name, var in solver.fields.items():
        sg = U.getSubVector(solver._subdict[name][0])
        solver._subdict[name][1].localToGlobal(var.vec, sg)
        U.restoreSubVector(solver._subdict[name][0], sg)
    return U


def solve_rotated_freeslip_nonlinear(solver, boundaries, remove_rotation_gauge=True,
                                     verbose=False, zero_init_guess=True, picard=0,
                                     rtol=None, atol=1.0e-11, stol=1.0e-8, max_it=50):
    """Nonlinear rotated strong-free-slip solve: a manual outer Newton/Picard loop
    that rotates the residual F(u), the Jacobian J(u) and the v_n=0 constraint EVERY
    iteration, reusing the validated self-contained rotated fieldsplit-Schur solve
    (``_solve_rotated_iterative``, incl. custom geometric FMG / GAMG velocity block
    and the rotated coupled null space) for each Newton increment.

    Why a manual loop rather than ``snes.solve()``: the rotated operator ``Q A Qᵀ``
    (a ``ptap`` result) carries no DM field information, so PETSc's DM-coupled
    fieldsplit + geometric-MG cannot precondition it (SUBPC_ERROR); the increment
    must be solved by the IS-based self-contained fieldsplit. Driving that from a
    manual Newton loop keeps the whole validated linear machinery and imposes the
    strong constraint exactly at every iterate.

    Each iteration (unknown carried in the CARTESIAN frame ``u``; the increment is
    solved in the rotated frame ``û = Q u``):
      * ``F = computeFunction(u)``  → Cartesian residual (native essential BCs on
        other boundaries already applied by the DM);
      * ``F̂ = Q F``, zero ``F̂`` at the constrained normal rows (the constraint
        residual is 0 there); converge on ‖F̂‖;
      * ``Ĵ = Q J(u) Qᵀ`` with ``zeroRowsColumns(normal_rows)``;
      * solve ``Ĵ δ̂ = −F̂``, ``δ = Qᵀ δ̂``, with a ‖F̂‖ backtracking line search;
      * ``u += α δ`` and re-impose ``v_n = 0`` exactly.

    The tangent used by ``computeJacobian`` is the solver's own (``consistent_jacobian``
    → Picard / Newton / continuation), so the rotated loop inherits the same tangent
    the standard path would use. The converged Cartesian residual is stashed as the
    constraint reaction for σ_nn recovery (``boundary_normal_traction``).

    Tangent / warmup handling (mirrors the standard ``solve`` path so nothing is
    silently ignored):

    * ``consistent_jacobian == "continuation"`` — a two-phase Picard→Newton solve via
      the ``constants[]``-routed blend α: phase 1 holds α=0 (frozen/Picard tangent) to
      the loose ``solver.newton_switch_rtol`` (and at least ``picard`` iterations if
      given) to enter Newton's basin; phase 2 sets α=1 (consistent Newton tangent) and
      drives to the requested tolerance. α is restored to 0 afterwards.
    * ``consistent_jacobian is True`` (pure Newton) with ``picard > 0`` — a Picard
      warmup needs the frozen tangent, which the pure-Newton compile does not carry, so
      this **raises** ``NotImplementedError`` pointing to ``"continuation"`` rather than
      silently ignoring the request.
    * ``consistent_jacobian is False`` (default, frozen/Picard) — the whole solve is
      already the frozen tangent, so ``picard`` is inherently satisfied (matches the
      standard path, whose post-warmup Newton phase also uses the frozen tangent here).
    """
    if getattr(solver, "snes", None) is None:
        solver._setup_pointwise_functions(); solver._setup_discretisation(); solver._setup_solver()
    dm = solver.dm
    snes = solver.snes
    snes.setUp()
    if rtol is None:
        rtol = float(solver.tolerance)

    # Resolve the tangent / warmup policy (see the docstring). ``continuation`` runs a
    # staged α=0 → α=1 solve; pure Newton + picard>0 is unsupported (no frozen tangent
    # to warm up with) and errors loudly; frozen (default) already satisfies a warmup.
    mode = getattr(solver, "consistent_jacobian", False)
    continuation = (mode == "continuation")
    if picard and not continuation and mode is True:
        raise NotImplementedError(
            f"rotated free-slip: a Picard->Newton warmup (picard={picard}) with the "
            "consistent Newton tangent (consistent_jacobian=True) requires "
            "consistent_jacobian='continuation' — with pure Newton the frozen (Picard) "
            "tangent needed for the warmup is not compiled in. Use "
            "consistent_jacobian='continuation' (staged Picard then Newton), or drop "
            "picard to run pure Newton.")
    switch_rtol = max(float(getattr(solver, "newton_switch_rtol", 1.0e-2)), rtol)
    if continuation:
        solver._set_newton_alpha(0.0)            # start in the Picard phase

    # Q, the custom-FMG prolongation and the coupled null space depend only on the
    # geometry / normals (NOT the solution), so build them ONCE and reuse each step.
    Q, Qt, normal_rows = build_rotation(solver, boundaries)
    custom_Pl = _build_rotated_custom_Pl(solver, Q, normal_rows)
    nsp = _rotated_nullspace(solver, Q, normal_rows)

    # initial guess (cartesian, composite): warm-start from the fields or zero, then
    # impose v_n=0 exactly on it so the iteration starts feasible.
    if zero_init_guess:
        u = dm.createGlobalVec(); u.set(0.0)
    else:
        u = _gather_fields_to_global(solver)
    uh = u.duplicate(); Q.mult(u, uh); _zero_rows_local(uh, normal_rows); Qt.mult(uh, u)
    uh.destroy()                             # transient projection buffer

    J, Jp = snes.getJacobian()[:2]
    pres_is = solver._subdict["pressure"][0]
    Fc = dm.createGlobalVec()
    reaction = dm.createGlobalVec()          # the final (un-zeroed) Cartesian residual

    # Reused across Newton iterations: the rotated operator (ptap-with-result), the
    # 1/mu pressure-mass Schur pmat (values refreshed in place), the constraint
    # diagonal scale (frozen at the first tangent — only the magnitude matters),
    # and the KSP/PC context (fieldsplit ISs, FMG hierarchy, GAMG setup survive).
    Ahat = None; Mp = None; ctx = None; diag_scale = None; lin_its = []

    def rotated_residual(uvec, keep_cartesian=False):
        snes.computeFunction(uvec, Fc)
        if keep_cartesian:
            Fc.copy(reaction)                # stash the Cartesian reaction for σ_nn
        Fh = Fc.duplicate(); Q.mult(Fc, Fh); _zero_rows_local(Fh, normal_rows)
        return Fh

    r0 = None; last_reason = 0; iters = 0; converged = False
    phase = "picard" if continuation else "newton"
    for iters in range(max_it):
        Fhat = rotated_residual(u, keep_cartesian=True)
        rnorm = Fhat.norm()
        if r0 is None:
            r0 = rnorm
        if verbose:
            from underworld3 import mpi
            mpi.pprint(f"[rotated_bc] nonlinear iter {iters:2d}  |F̂|={rnorm:.6e}  "
                       f"rel={rnorm/(r0+1e-300):.3e}  [{phase}]")
        # residual convergence (relative to the initial residual, plus an absolute
        # floor so an already-converged warm start does not chase machine noise).
        if rnorm <= rtol * r0 + atol:
            converged = True; Fhat.destroy(); break
        # Continuation: switch the frozen (Picard, α=0) tangent to the consistent
        # (Newton, α=1) tangent once the residual has dropped into Newton's basin (the
        # loose newton_switch_rtol) and at least `picard` Picard iterations have run.
        if continuation and phase == "picard" and rnorm <= switch_rtol * r0 and iters >= picard:
            solver._set_newton_alpha(1.0); phase = "newton"
            if verbose:
                from underworld3 import mpi
                mpi.pprint(f"[rotated_bc] continuation: Picard→Newton at iter {iters} "
                           f"(rel |F̂| {rnorm/(r0+1e-300):.2e})")
        snes.computeJacobian(u, J, Jp)       # Jp carries the 1/mu mass (Schur pmat)
        if Ahat is None:
            Ahat = J.ptap(Qt)
        else:
            J.ptap(Qt, result=Ahat)          # same nonzero pattern → in-place refresh
        if ctx is None:
            Mp = _pressure_mass_schur_pmat(solver)
        elif Mp is not None:
            Jp.createSubMatrix(pres_is, pres_is, submat=Mp)   # viscosity may be u-dependent
        if diag_scale is None:
            diag_scale = _velocity_diag_scale(Ahat, solver)
        Ahat.zeroRowsColumns(normal_rows, diag=diag_scale)
        bhat = Fhat.copy(); bhat.scale(-1.0)
        dhat, last_reason, ctx = _solve_rotated_iterative(
            solver, Ahat, bhat, Q, Qt, normal_rows,
            custom_Pl=custom_Pl, nsp=nsp, Mp=Mp, verbose=False, ctx=ctx)
        lin_its.append(ctx["ksp"].getIterationNumber())
        d = dm.createGlobalVec(); Qt.mult(dhat, d)
        # step-norm convergence (SNES_CONVERGED_SNORM): a tiny Newton step means we
        # are at the solution — the exit for a warm start that is already converged
        # (otherwise the relative test above, with a tiny r0, chatters near machine
        # level). ‖u‖=0 on a cold start ⇒ this never fires prematurely (d is large).
        if d.norm() <= stol * (u.norm() + 1e-30):
            converged = True
            dhat.destroy(); d.destroy(); bhat.destroy(); Fhat.destroy()
            break
        # backtracking line search on ‖F̂‖ (full Newton/Picard step first). Cheap
        # insurance far from the solution; α=1 is accepted immediately near it. If no
        # step reduces the residual, the iteration has stalled (typically already at
        # the solution) → stop rather than accept a non-decreasing step.
        alpha = 1.0; improved = False
        for _ls in range(8):
            utry = u.copy(); utry.axpy(alpha, d)
            uth = utry.duplicate(); Q.mult(utry, uth); _zero_rows_local(uth, normal_rows); Qt.mult(uth, utry)
            uth.destroy()
            Ftry = rotated_residual(utry)
            if Ftry.norm() < rnorm:
                u.destroy(); u = utry; improved = True; Ftry.destroy(); break
            utry.destroy(); Ftry.destroy(); alpha *= 0.5
        dhat.destroy(); d.destroy(); bhat.destroy(); Fhat.destroy()
        if not improved:
            break

    # Restore a clean frozen (Picard) tangent for any subsequent solve (next time
    # step), matching the standard _continuation_solve which leaves alpha at 0.
    if continuation:
        solver._set_newton_alpha(0.0)

    # The loop can exhaust max_it or stall in the line search (`not improved`) without
    # meeting the residual / step-norm criteria. Warn — as the standard SNES path does
    # on divergence — so an unconverged iterate left in the fields is not silent.
    if not converged:
        from underworld3 import mpi
        rel = (rnorm / (r0 + 1e-300)) if r0 is not None else float("nan")
        mpi.pprint(f"[rotated_bc] WARNING: nonlinear rotated free-slip did NOT converge "
                   f"in {iters + 1} iterations (rel |F̂| = {rel:.2e}); the fields hold "
                   f"the last (unconverged) iterate.")

    Fc.destroy()                             # residual output buffer (reaction is kept for info)
    _destroy_rotated_ksp_ctx(ctx)            # KSP/PC + the owned Schur pmat
    if Ahat is not None:
        Ahat.destroy()                       # the reused rotated operator
    removed = _finalize_rotated_solution(solver, u, Q, normal_rows, remove_rotation_gauge)

    return {"Q": Q, "Qt": Qt, "reaction": reaction, "U": u,
            "normal_rows": normal_rows, "boundaries": list(boundaries),
            "rotation_gauge_removed": removed, "ksp_reason": last_reason,
            "nonlinear_iterations": iters, "converged": converged,
            "ksp_its": lin_its,
            "continuation_switched": continuation and phase == "newton"}


def _build_rotated_custom_Pl(solver, Q, normal_rows):
    """The rotated custom-FMG prolongation list [*coarse, Q_v·P_fine] for the
    velocity block, or None if no hierarchy is registered. Depends only on Q and
    the mesh (NOT the solution), so the nonlinear driver builds it ONCE and reuses
    it across Newton iterations (the prolongation build is the expensive part)."""
    if getattr(solver, "_custom_mg", None) is None:
        return None
    vel_is = solver._subdict["velocity"][0]
    vis = np.asarray(vel_is.getIndices())
    g2blk = {int(g): k for k, g in enumerate(vis)}
    Qv = Q.createSubMatrix(vel_is, vel_is)
    nrows_blk = sorted({g2blk[g] for g in normal_rows if g in g2blk})
    Ps = solver._custom_mg["hierarchy"].build(solver)
    Pfine = Qv.matMult(Ps[-1]); Pfine.zeroRows(nrows_blk, diag=0.0)
    return list(Ps[:-1]) + [Pfine]


def _pressure_mass_schur_pmat(solver):
    """The native 1/mu pressure-mass Schur preconditioner block, extracted from the
    solver's assembled Pmat — the SAME p-p block the standard path uses via
    ``pc_fieldsplit_schur_precondition=a11`` (the DS JacobianPreconditioner term
    ``_pp_G0``). Q is identity on pressure, so the block needs no rotation.
    Returns None (→ selfp fallback) when the Pmat is not distinct from the operator
    or the block was not assembled. The CALLER must have assembled the Pmat
    (``snes.computeJacobian(x, J, Jp)``) and owns the returned Mat."""
    A, P = solver.snes.getJacobian()[:2]
    if P is None or P.handle == A.handle:
        return None
    pres_is = solver._subdict["pressure"][0]
    Mp = P.createSubMatrix(pres_is, pres_is)
    if Mp.norm() == 0.0:                               # not assembled → useless
        Mp.destroy()
        return None
    return Mp


def _velocity_diag_scale(Ahat, solver):
    """Mean |diag| of the (rotated) velocity block — the representative diagonal
    for constraint rows (see the zeroRowsColumns call sites). Collective."""
    d = Ahat.getDiagonal()
    vis = solver._subdict["velocity"][0]
    sub = d.getSubVector(vis)
    n = sub.getSize()
    s = sub.norm(PETSc.NormType.NORM_1) / max(n, 1)
    d.restoreSubVector(vis, sub)
    d.destroy()
    return float(s) if s > 0.0 else 1.0


def _destroy_rotated_ksp_ctx(ctx):
    """Release the reusable rotated-KSP context (KSP and the owned Schur pmat)."""
    if ctx is None:
        return
    if ctx.get("ksp") is not None:
        ctx["ksp"].destroy()
    if ctx.get("Mp") is not None:
        ctx["Mp"].destroy()


def _solve_rotated_iterative(solver, Ahat, bhat, Q, Qt, normal_rows, verbose=False,
                             custom_Pl=None, nsp=None, Mp=None, ctx=None):
    """Solve the rotated saddle with a SELF-CONTAINED fieldsplit-Schur KSP on the
    rotated operator. The velocity block is geometric FMG on the CUSTOM prolongation
    (PR#290, rotated) when a hierarchy is registered (``set_custom_fmg``), else GAMG
    (tuned to the native path's settings).

    A plain rotated Mat has no DM field info, so UW3's DM-coupled fieldsplit cannot
    split it — we build the split from EXPLICIT velocity/pressure index sets. For the
    custom-FMG case the velocity sub-PC gets our prolongation via ``setMGInterpolation``
    (needs no DM); the rotated block A_vv = Q_v A_vv Q_vᵀ is formed from Âhat
    automatically and only the FINE prolongation is rotated (Galerkin coarse ops
    auto-correct). NO direct solve of the fine system.

    Preconditioning parity with the native solve (the Schur-iteration fix):
      * ``Mp`` — the 1/mu pressure-mass block from the native Pmat
        (``_pressure_mass_schur_pmat``) is installed as a USER Schur preconditioner,
        exactly what the standard path uses via ``schur_precondition=a11``. Without
        it (Mp=None) the Schur pre falls back to selfp, which degrades badly on
        curved/deformed boundaries and variable viscosity.
      * the pressure sub-solve mirrors the native FGMRES+GASM at the solver
        tolerance; the constant-pressure nullspace is attached to the Schur
        complement for enclosed domains (the hand-built IS fieldsplit does not
        inherit it from the operator).

    ``custom_Pl`` / ``nsp`` may be PREBUILT (nonlinear driver: build once, reuse each
    Newton step); when None they are built here (linear one-shot).

    Returns ``(Uhat, reason, ctx)``. Passing ``ctx`` back in reuses the KSP/PC
    across Newton iterations — the fieldsplit ISs, Schur USER pmat and FMG
    prolongations survive; only the operator-values refresh is paid. The caller
    must keep ``Ahat``/``Mp`` the SAME Mat objects (values updated in place) and
    release the context with ``_destroy_rotated_ksp_ctx`` when done."""
    from underworld3.utilities import custom_mg
    dm = solver.dm
    vel_is = solver._subdict["velocity"][0]
    pres_is = solver._subdict["pressure"][0]

    if ctx is None:
        if custom_Pl is None:
            custom_Pl = _build_rotated_custom_Pl(solver, Q, normal_rows)

        # rotated coupled null space (pressure-const ⊕ Q·rotation) on the operator
        if nsp is None:
            nsp = _rotated_nullspace(solver, Q, normal_rows)
        if nsp is not None:
            Ahat.setNullSpace(nsp); Ahat.setTransposeNullSpace(nsp)

        # UNIQUE prefix per KSP (see _ROT_SOLVE_COUNT) so concurrent rotated solves
        # do not share global-options state; the keys are removed after setup.
        global _ROT_SOLVE_COUNT
        _ROT_SOLVE_COUNT += 1
        pfx = f"rotfs{_ROT_SOLVE_COUNT}_"
        opts = PETSc.Options()
        tol = float(solver.tolerance)
        cfg = {
            "ksp_type": "fgmres", "ksp_rtol": str(tol), "ksp_max_it": "300",
            "pc_type": "fieldsplit", "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            # native-parity pressure sub-solve (pyx Stokes defaults): FGMRES at the
            # solver tolerance; GASM on the 1/mu mass, jacobi if only selfp exists.
            "fieldsplit_pres_ksp_type": "fgmres",
            "fieldsplit_pres_ksp_rtol": str(tol),
            "fieldsplit_pres_ksp_max_it": "200",
        }
        if Mp is not None:
            cfg["fieldsplit_pres_pc_type"] = "gasm"
        else:
            cfg["pc_fieldsplit_schur_precondition"] = "selfp"
            cfg["fieldsplit_pres_pc_type"] = "jacobi"
        if custom_Pl is None:
            # GAMG fallback velocity block, tuned to native parity (pyx Stokes
            # defaults). NOTE: the custom-FMG route is the preferred velocity
            # block — this applies only when no hierarchy is registered.
            cfg.update({
                "fieldsplit_vel_ksp_type": "fgmres",
                "fieldsplit_vel_ksp_rtol": str(tol * 0.1),
                "fieldsplit_vel_ksp_max_it": "200",
                "fieldsplit_vel_pc_type": "gamg",
                "fieldsplit_vel_pc_gamg_type": "agg",
                "fieldsplit_vel_pc_gamg_repartition": "true",
                "fieldsplit_vel_pc_mg_type": "additive",
                "fieldsplit_vel_pc_gamg_agg_nsmooths": "2",
                "fieldsplit_vel_mg_levels_ksp_max_it": "3",
                "fieldsplit_vel_mg_levels_ksp_converged_maxits": "true",
            })
        else:
            # full-MG cycle per Schur application, by design
            cfg["fieldsplit_vel_ksp_type"] = "preonly"
        for k, v in cfg.items():
            opts[pfx + k] = v
        try:
            ksp = PETSc.KSP().create(comm=dm.comm); ksp.setOptionsPrefix(pfx)
            ksp.setOperators(Ahat)
            pc = ksp.getPC(); pc.setType("fieldsplit")
            pc.setFieldSplitIS(("vel", vel_is), ("pres", pres_is))
            ksp.setFromOptions()
            if Mp is not None:
                pc.setFieldSplitSchurPreType(
                    PETSc.PC.FieldSplitSchurPreType.USER, Mp)
            pc.setUp()
            if custom_Pl is not None:                 # geometric FMG via custom P
                vel_pc = pc.getFieldSplitSubKSP()[0].getPC()
                A_vv, P_vv = vel_pc.getOperators()
                vel_pc.reset(); vel_pc.setOperators(A_vv, P_vv)
                custom_mg._configure_pcmg(vel_pc, custom_Pl)
                # The Galerkin-coarsened ROTATED velocity block inherits every
                # rigid-rotation nullspace mode of the constrained problem (a
                # closed circle: one; a spherical shell: three) — the default
                # redundant/LU coarse solve hits a zero pivot (SUBPC_ERROR,
                # outer reason -11). SVD is nullspace-robust and the coarse
                # level is small; same choice as the native spherical FMG setups.
                vopts = PETSc.Options()
                vpfx = vel_pc.getOptionsPrefix() or ""
                vopts.setValue(vpfx + "mg_coarse_pc_type", "svd")
                vopts.delValue(vpfx + "mg_coarse_redundant_pc_type")
                vel_pc.setFromOptions()
                vel_pc.setUp()
                vopts.delValue(vpfx + "mg_coarse_pc_type")
            # Constant-pressure nullspace on the Schur COMPLEMENT (enclosed
            # domains): the IS-built fieldsplit does not propagate the coupled
            # nullspace to the inner Schur solve, which is otherwise singular
            # and grinds against its iteration cap every outer iteration.
            cns = None
            if getattr(solver, "_petsc_use_pressure_nullspace", False):
                S = pc.getFieldSplitSubKSP()[1].getOperators()[0]
                cns = PETSc.NullSpace().create(constant=True, comm=dm.comm)
                S.setNullSpace(cns)
        finally:
            # all options consumed by setFromOptions/setUp — drop them so the
            # global database stays clean (and bounded under time-stepping).
            for k in cfg:
                try:
                    opts.delValue(pfx + k)
                except Exception:
                    pass
        ctx = {"ksp": ksp, "pc": pc, "Mp": Mp, "nsp": nsp, "cns": cns,
               "custom_Pl": custom_Pl, "pfx": pfx}
    else:
        ksp = ctx["ksp"]
        nsp = ctx["nsp"]
        # Same Mat objects, new values (ptap-with-result / createSubMatrix-with-
        # submat) — poke the KSP so PCSetUp refreshes on the changed operator.
        ksp.setOperators(Ahat)

    if nsp is not None:
        nsp.remove(bhat)                              # project EVERY rhs

    Uhat = Ahat.createVecRight(); Uhat.set(0.0)
    ksp.solve(bhat, Uhat)
    _warn_if_ksp_diverged(ksp, kind="rotated fieldsplit-Schur")
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
        kind = "custom-FMG" if ctx["custom_Pl"] is not None else "GAMG"
        schur = "1/mu-mass" if ctx["Mp"] is not None else "selfp"
        mpi.pprint(f"[rotated_bc] velocity block = {kind}; Schur pre = {schur}; "
                   f"outer KSP {ksp.getConvergedReason()} in "
                   f"{ksp.getIterationNumber()} its")
    return Uhat, ksp.getConvergedReason(), ctx


def _rotated_nullspace(solver, Q, normal_rows):
    """Coupled Stokes null space in the rotated frame: constant pressure, plus every
    rigid rotation Q·mode that is a genuine null space of the constraints — one
    mode (-y,x) in 2D, up to THREE (e_k×r) in 3D (all three on a spherical shell
    with free-slip inner and outer boundaries; leaving them off makes the outer
    Krylov grind against a near-singular operator and pollutes the solution with
    arbitrary rotation content). Returns a PETSc.NullSpace on the composite
    vector, or None.

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
        pv = dm.createGlobalVec(); pv.set(0.0)     # persists inside the returned NullSpace
        pis = solver._subdict["pressure"][0]
        sp = pv.getSubVector(pis); sp.set(1.0); pv.restoreSubVector(pis, sp)
        pv.normalize(); vecs.append(pv)
    # rigid rotations (rotated), each only if it satisfies the constraints.
    # COLLECTIVE: all ranks walk the same mode list, same order.
    for tg in _rigid_rotation_modes(solver):
        if _mode_satisfies_constraints(solver, Q, normal_rows, tg):
            tr = tg.duplicate(); Q.mult(tg, tr)    # tr persists in the NullSpace
            vecs.append(tr)
        dm.restoreGlobalVec(tg)                    # tg transient → return to pool
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


def _mode_satisfies_constraints(solver, Q, normal_rows, tg, tol=1e-8):
    """True iff the rigid-body mode ``tg`` satisfies all rotated v_n=0
    constraints — i.e. Q·tg is ~0 on every constrained normal row. (A closed
    circular boundary admits its one rotation; a full spherical shell admits all
    three; straight/partial walls pin them.)

    COLLECTIVE: every rank runs the same global-vector ops. Do NOT early-return on
    a per-rank ``not normal_rows`` — in parallel a rank may own no boundary node
    (empty normal_rows) while others do, and an early return there would desync the
    collective norms below and deadlock."""
    tr = tg.duplicate(); Q.mult(tg, tr)
    full = tr.norm()                              # parallel norm
    # norm of tr restricted to the constrained rows: zero everything else, then .norm()
    rs, re = tr.getOwnershipRange()
    loc = np.asarray([g - rs for g in normal_rows if rs <= g < re], dtype=np.int64)
    trc = tr.duplicate(); trc.set(0.0)
    tra = trc.getArray(); tga = tr.getArray()
    tra[loc] = tga[loc]; trc.setArray(tra)
    viol = trc.norm() / (full + 1e-30)            # parallel (collective on all ranks)
    tr.destroy(); trc.destroy()                   # transient duplicates
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
    # Cartesian nodal reaction r_c = F(u) at the converged state. The nonlinear
    # driver stashes it directly (the final ``computeFunction`` residual); the linear
    # one-shot reconstructs it as A·u−b (with A=J(0), b=−F(0), F affine ⇒ A·u−b=F(u)).
    if info.get("reaction") is not None:
        rc = info["reaction"]; own_rc = False    # owned by info — do NOT destroy
    else:
        A = info["A"]; b = info["b"]; U = info["U"]
        rc = A.createVecLeft(); A.mult(U, rc); rc.axpy(-1.0, b); own_rc = True
    rcl = dm.getLocalVec(); dm.globalToLocal(rc, rcl); rca = np.asarray(rcl.getArray())

    lsec = dm.getLocalSection(); VEL = _velocity_field_id(solver)
    csec = dm.getCoordinateSection()
    cvec = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dim)
    v0, v1 = dm.getDepthStratum(0)
    normal = dict((nm, nrm) for nm, nrm in
                  [(s if isinstance(s, tuple) else (s, None)) for s in info["boundaries"]]).get(boundary)
    nodes = _boundary_velocity_nodes(solver, boundary, normal=normal)
    xs = []; Rn = []
    for q, nrm in nodes:
        lo = lsec.getFieldOffset(q, VEL)
        rcv = rca[lo:lo + dim]                    # Cartesian reaction at this node (local)
        xs.append(_point_coord(dm, dim, cvec, csec, v0, v1, q))
        Rn.append(float(np.dot(nrm, rcv)))        # R_i = n̂·r_c  (corner-correct)
    dm.restoreLocalVec(rcl)
    if own_rc:
        rc.destroy()                          # reconstructed A·u−b (linear path) — free it
    xs = np.array(xs); Rn = np.array(Rn)
    # σ_nn = −(nodal reaction), de-smeared by the SHARED boundary-mass primitive and
    # mean-removed (the ρg·h gauge). partial_reaction=False: the reaction here comes from
    # the ASSEMBLED global operator Q(A·u−b), already complete at every node (ranks agree
    # at a partition-cut node — must OVERWRITE, not sum, else shared nodes double-count).
    return xs, _desmear(solver, boundary, xs, -Rn, mass,
                        remove_mean=True, partial_reaction=False)


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
    # σ_nn is already mean-removed (the ρg·h gauge); topography h = -σ_nn / (Δρ g).
    def key(c):
        return tuple(round(float(t), 9) for t in np.asarray(c).ravel()[:dim])
    hmap = {key(x): -float(s) / buoyancy_scale for x, s in zip(np.asarray(xs), np.asarray(sig))}
    # shared bulk field write (parallel-safe: write ONCE, not per node)
    return write_boundary_scalar_field(solver, field, hmap, dim)


def _rigid_rotation_modes(solver):
    """The Cartesian rigid-body rotation mode(s) as composite GLOBAL vectors
    (velocity DOFs only, zero pressure): ``[(-y,x)]`` in 2D, ``[e_k×r]`` for
    k=x,y,z in 3D — a spherical shell with free-slip on both boundaries has all
    THREE. Parallel-safe: built via localToGlobal on the velocity sub-DM, so
    shared nodes are handled by PETSc, not double-counted. The vectors are
    borrowed from the DM pool — every one must go back via
    ``dm.restoreGlobalVec``. COLLECTIVE: all ranks build the same mode list."""
    dm = solver.dm
    v = solver.Unknowns.u
    c = v.coords
    saved = v.data.copy()
    if solver.mesh.dim == 2:
        fields = [np.column_stack([-c[:, 1], c[:, 0]])]
    else:
        x, y, z = c[:, 0], c[:, 1], c[:, 2]
        zero = np.zeros_like(x)
        fields = [np.column_stack([zero, -z, y]),      # e_x × r
                  np.column_stack([z, zero, -x]),      # e_y × r
                  np.column_stack([-y, x, zero])]      # e_z × r
    vis = solver._subdict["velocity"][0]
    modes = []
    for f in fields:
        v.data[...] = f
        tg = dm.getGlobalVec(); tg.set(0.0)
        sg = tg.getSubVector(vis)
        solver._subdict["velocity"][1].localToGlobal(v.vec, sg)
        tg.restoreSubVector(vis, sg)
        modes.append(tg)
    v.data[...] = saved
    return modes
