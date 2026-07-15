"""Monge–Ampère (BFO convex-branch) equidistribution machinery, the
linear OT improvement step, and the mover sub-solver wiring that the
anisotropic mover also reuses. See the package docstring for the
module map.
"""

import numpy as np

import underworld3 as uw

from .graph import (_tri_cells, _signed_areas,
                    _cap_step_to_edge_fraction, _backtracked_move,
                    _reweight_displacement_radial_tangential,
                    _global_sum, _global_max, _global_mean)


# ======================================================================
#  Monge–Ampère mesh-equidistribution machinery (PRESERVED, not the
#  default metric path). Exhaustively investigated 2026-05-16: every
#  FE-MA-potential variant (linear / recovered-Hessian smoothed &
#  variational / BFO convex-branch + damping / outer composition)
#  caps at deep/near ≈ 1.07 for an 8× target vs an exact ~10× — see
#  the project memory and scripts/ma_*.py. Kept because (a) the
#  "bit-identical across variants" result suggests a common missing
#  ingredient worth understanding, and (b) the elastic-spring
#  redistribution may work as a *preconditioner* for the MA solve
#  (a graded starting mesh might let MA escape the weak branch) —
#  an open investigation. Call _monge_ampere_mover() directly to use.
# ======================================================================

# Cached MA solver state keyed by (mesh-id, pinned-labels, topology):
# the φ Poisson, the variational Hessian-recovery solver, ∇φ
# projector, the ρ_cur proxy field. Rebuilt on a topology change.
_WINSLOW_CACHE: dict = {}

# Sign of the BFO source vs UW3's SNES_Poisson convention
# (SNES_Poisson F0 = -f, strong form Δφ = -ps.f). With this sign the
# validated linear first iterate Δφ = (c/ρ_tgt - 1) grades the right
# way (nodes toward high target density).
_EQUIDIST_SIGN = -1.0

_HESSIAN_CLASS = None


# Cached state for the OT-improvement-step path (one weighted-
# Poisson per call). Keyed like the other movers; same lifetime.
_OT_CACHE: dict = {}


def _use_direct_solver(solver, singular=False, elliptic=True):
    r"""Force a cached MA sub-solver onto a sparse **direct** factorisation
    (MUMPS LU) instead of the UW3 default GMRES + GAMG.

    **Parallel safety (parallel-singular-corruption, 2026-05-27):** MUMPS
    (the only parallel LU in this build) corrupts the heap when the same
    factorisation path is exercised over *repeated* solves at np >= 3 — a
    probabilistic SEGV/SIGBUS or MPI deadlock (the UW3 default GMRES+GAMG, which
    never calls MUMPS, is clean). Since the movers re-solve in a Picard/outer
    loop, MUMPS is unusable in parallel here. Under MPI this function therefore
    falls back to the MUMPS-free iterative path (:func:`_use_iterative_solver`);
    the direct MUMPS path below is kept for the validated **serial** efficiency
    lever. ``elliptic`` is forwarded to the iterative fallback (φ-Poisson →
    GAMG; mass systems → CG+Jacobi).

    Why this is the dominant MA-efficiency lever (profiled 2026-05-17,
    res-16 Annulus, AMP=8, warm re-call): the Picard loop fixes the
    mesh, so the φ-Poisson Laplacian and the Hessian-recovery SPD mass
    matrix are *constant operators* re-solved ~40× with only the RHS
    changing. With GAMG, every ``solve()`` pays a full multigrid
    **setup** (the constant near-nullspace re-attach forces it) — the
    Hessian solve alone was ~0.93 s/iter ≈ 37 s. These problems are
    tiny (≲10⁴ DOF); MUMPS factorises in milliseconds and the per-iter
    cost collapses to a back-substitution. A direct solve is also
    *exact* (machine precision, tighter than the GMRES rtol), so the
    Picard fixed point — hence the grading/quality — is unchanged.

    ``singular=True`` (the pure-Neumann φ Poisson): MUMPS null-pivot
    detection (ICNTL(24)=1) handles the rank-1-deficient operator; the
    ``constant_nullspace`` hook still removes the constant mode from
    the RHS/solution, so the result is the same consistent solution
    the iterative path produced — but it also eliminates the
    GAMG-on-pure-Neumann ``DIVERGED_LINEAR_SOLVE`` re-solve pathology.
    """
    # Parallel: MUMPS-repeated corrupts the heap (see docstring) — use the
    # MUMPS-free iterative path instead. Serial keeps the fast direct solve.
    if uw.mpi.size > 1:
        _use_iterative_solver(solver, singular=singular, elliptic=elliptic)
        return

    o = solver.petsc_options
    # These three sub-problems are *linear* (φ Poisson with the Hessian
    # source frozen; the SPD Hessian-recovery mass system; the ∇φ
    # projection) → one KSP solve, no Newton line-search / 2nd iterate
    # (which was doubling work and emitting spurious
    # ``DIVERGED_LINEAR_SOLVE`` after 2 iters).
    o["snes_type"] = "ksponly"
    # ksponly does exactly ONE linear KSP solve (no Newton). Default
    # snes_max_it leaves snes->iter=0, so if a converged-reason
    # viewer is on (a user's global -snes_converged_reason, an outer
    # debug flag, …) PETSc mislabels the *successful* linear solve
    # as "DIVERGED_MAX_IT iterations 0" and floods the log with
    # phantom failures. snes_max_it=1 ⇒ the single solve counts as
    # one converged iteration ⇒ reason = CONVERGED, not a fake
    # DIVERGED. Numerically inert (the KSP solve is identical) —
    # purely stops these linear sub-solves masquerading as failures.
    o["snes_max_it"] = 1
    # The Picard loop fixes the mesh, so the operator is **constant**
    # across the ~40 inner solves — only the RHS changes. Lag the
    # Jacobian (compute once, reuse) and the preconditioner (factorise
    # once, reuse): every subsequent inner solve collapses to a MUMPS
    # back-substitution. A fresh ``solver.solve()`` after
    # ``_deform_mesh`` rebuilds the SNES (is_setup=False) so the lag
    # counter resets and the operator is correctly re-factorised on the
    # first solve of the next call — the reuse is confined to the loop
    # where the mesh genuinely does not move.
    o["snes_lag_jacobian"] = -2
    o["snes_lag_preconditioner"] = -2
    o["ksp_type"] = "preonly"
    o["pc_type"] = "lu"
    o["pc_factor_mat_solver_type"] = "mumps"
    if singular:
        o["mat_mumps_icntl_24"] = 1   # null-pivot detection
        o["mat_mumps_icntl_25"] = 0   # one solution of the singular sys
    # GAMG-only keys are inert once pc_type≠gamg; drop them so the
    # effective option set is exactly what is documented.
    for k in ("pc_gamg_type", "pc_gamg_repartition", "pc_mg_type",
              "pc_gamg_agg_nsmooths", "mg_levels_ksp_max_it",
              "mg_levels_ksp_converged_maxits"):
        try:
            o.delValue(k)
        except Exception:
            pass


def _use_iterative_solver(solver, singular=False, elliptic=True):
    r"""Parallel-scalable alternative to ``_use_direct_solver``: keep
    the *same factor/setup-once-reuse pattern* (the real efficiency
    lever) but with an **iterative** PC so it scales beyond the
    serial / modest-size regime where sparse direct factorisation is
    viable (this PETSc build has only MUMPS + serial builtin LU — no
    hypre / SuperLU_DIST).

    The Picard loop fixes the mesh ⇒ the operator is constant across
    the ~25 inner solves; ``snes_lag_jacobian=-2`` /
    ``snes_lag_preconditioner=-2`` build the PC **once per
    ``_monge_ampere_mover`` call** and reuse it for every inner solve
    (the GAMG hierarchy / Jacobi diagonal is *not* rebuilt per
    iteration — that per-iter GAMG re-setup was the original ~0.9 s
    Hessian cost). ``_deform_mesh`` resets ``is_setup`` so the lag
    counter resets and the PC is correctly rebuilt on the next call's
    first solve. Combined with a Krylov **warm start** from the
    previous Picard φ (caller passes ``zero_init_guess=False``), the
    inner solves are a handful of CG iterations on an already-built
    hierarchy.

    ``elliptic=True`` (the φ-Poisson Laplacian): CG + GAMG with the
    constant near-nullspace (already attached via
    ``constant_nullspace`` — GAMG needs it for the pure-Neumann
    operator). ``elliptic=False`` (the SPD Hessian-recovery / ∇φ mass
    systems): a mass matrix is spectrally trivial — CG + Jacobi
    converges in a few iterations with **no** hierarchy setup, fully
    parallel; GAMG there would be wasted setup.

    Numerics: an iterative solve to a tight ``ksp_rtol`` reproduces
    the BFO Picard fixed point — hence the grading — to well within
    its 4-dp precision (validated against the direct path); it is a
    *cost/parallelism* change, not a formulation change.
    """
    o = solver.petsc_options
    o["snes_type"] = "ksponly"
    # See _use_direct_solver: snes_max_it=1 stops a converged-reason
    # viewer mislabelling these linear ksponly sub-solves as
    # "DIVERGED_MAX_IT iterations 0". Numerically inert.
    o["snes_max_it"] = 1
    o["snes_lag_jacobian"] = -2
    o["snes_lag_preconditioner"] = -2
    # Krylov choice is per-operator (set in the branches below):
    #  * elliptic φ-Poisson → FGMRES. The UW3 DMPlex-FEM assembly +
    #    Neumann/nullspace handling does not guarantee an *exactly*
    #    symmetric operator, and the GAMG **SOR smoother is
    #    non-symmetric**, so the preconditioner is non-SPD — CG's
    #    assumptions are violated (it only "worked" here by
    #    robustness margin). FGMRES tolerates a non-symmetric
    #    operator *and* a varying/non-symmetric preconditioner.
    #  * mass systems (Hessian recovery, ∇φ projection) → CG: a
    #    consistent mass matrix with a Jacobi PC is provably SPD and
    #    symmetric, so CG is correct and the cheapest option.
    # Inner solve inside an outer BFO Picard — it tolerates inexact
    # inner solves (inexact-Picard); 1e-7 is far tighter than the
    # Picard increment near convergence (~1e-4) so the fixed point —
    # hence the grading — is unchanged, at a fraction of the iters a
    # direct-path-matching 1e-10 would need.
    o["ksp_rtol"] = 1.0e-7
    o["ksp_atol"] = 1.0e-12
    o["pc_factor_mat_solver_type"] = ""   # not a direct solve
    try:
        o.delValue("pc_factor_mat_solver_type")
        o.delValue("mat_mumps_icntl_24")
        o.delValue("mat_mumps_icntl_25")
    except Exception:
        pass
    if elliptic:
        # P3 pure-Neumann Laplacian: plain agg-GAMG with a weak
        # Jacobi/Chebyshev smoother needs ~280 iters here. A stronger
        # SOR smoother with more sweeps + smoothed aggregation cuts
        # that ~4×; the hierarchy is still built only once per call
        # (lagged), so the extra setup is amortised over the ~25
        # reused inner solves. SOR ⇒ non-symmetric PC ⇒ FGMRES.
        o["ksp_type"] = "fgmres"
        o["ksp_gmres_restart"] = 100      # > the ~75-iter solve
        o["pc_type"] = "gamg"
        o["pc_gamg_type"] = "agg"
        o["pc_gamg_agg_nsmooths"] = 1
        o["pc_gamg_threshold"] = 0.02
        o["mg_levels_ksp_type"] = "richardson"
        o["mg_levels_pc_type"] = "sor"
        o["mg_levels_ksp_max_it"] = 4
        # GAMG coarse solve. MUMPS (parallel LU) corrupts the heap over repeated
        # parallel solves (parallel-singular-corruption) — so in parallel use a
        # MUMPS-free coarse: `redundant` replicates the (tiny) coarse grid to
        # every rank and solves it with a dense SVD, which is robust on the
        # singular pure-Neumann coarse and never calls MUMPS (verified clean +
        # convergent at np=5). Serial keeps the fast MUMPS coarse.
        for k in ("mg_coarse_pc_factor_mat_solver_type",
                  "mg_coarse_redundant_pc_type"):
            try: o.delValue(k)
            except Exception: pass
        if uw.mpi.size > 1:
            o["mg_coarse_pc_type"] = "redundant"
            o["mg_coarse_redundant_pc_type"] = "svd"
        else:
            o["mg_coarse_pc_type"] = "lu"
            o["mg_coarse_pc_factor_mat_solver_type"] = "mumps"
    else:
        o["ksp_type"] = "cg"              # consistent mass = SPD
        o["pc_type"] = "jacobi"           # mass matrix → trivial
        for k in ("ksp_gmres_restart", "pc_gamg_type",
                  "pc_gamg_agg_nsmooths", "pc_gamg_threshold",
                  "mg_levels_ksp_type", "mg_levels_pc_type",
                  "mg_levels_ksp_max_it", "mg_coarse_pc_type",
                  "mg_coarse_pc_factor_mat_solver_type",
                  "mg_coarse_redundant_pc_type"):
            try:
                o.delValue(k)
            except Exception:
                pass


def _solver_wiring(linear_solver):
    """Option-wiring function for a cached mover sub-solver.

    ``linear_solver="gamg"`` wires the iterative, parallel-scalable
    option set (:func:`_use_iterative_solver`); anything else wires the
    serial MUMPS factor-once set (:func:`_use_direct_solver`, which
    itself falls back to the iterative path under MPI — the
    MUMPS-heap-corruption guard)."""
    if linear_solver == "gamg":
        def _wire(s, singular=False, elliptic=True):
            _use_iterative_solver(s, singular, elliptic)
    else:
        def _wire(s, singular=False, elliptic=True):
            _use_direct_solver(s, singular, elliptic)
    return _wire


def _warm_start_krylov(linear_solver):
    """True when a mover's inner solves should WARM-START the Krylov
    iteration from the previous solution (pass ``zero_init_guess=False``):

    * the GAMG path always — the hierarchy is built once (lagged) and
      the solution changes slowly under relaxation, so the warm start
      leaves only a handful of Krylov iterations per inner solve;
    * the "direct" path under MPI — :func:`_use_direct_solver` silently
      routes to the iterative solver there (MUMPS-heap-corruption
      guard), so the warm start pays exactly as it does for GAMG.

    The serial direct path is an exact factorisation, indifferent to
    the initial guess."""
    return (linear_solver == "gamg"
            or (linear_solver == "direct" and uw.mpi.size > 1))


def _patch_volumes(tris, coords, n_verts, vol_field=None):
    """Per-vertex dual-patch area: a node's share (1/3) of every
    incident triangle's |area|. ρ_cur ∝ 1/patch for the (opt-in,
    n_outer>1) outer MA composition; at equidistribution
    ``patch · ρ_tgt`` is uniform.

    This quantity is exactly the **lumped P1 mass diagonal** ``M_ii = ∫ N_i dV``.
    The hand-rolled local sum below is serial-exact, but **under-counts shared
    vertices on rank-partition boundaries in parallel** — each rank only adds its
    own incident triangles and never sums the neighbouring rank's. So in parallel
    we assemble it through the FE mass matrix instead (``_lumped_vertex_volumes``),
    where PETSc does the cross-rank ``localToGlobal(ADD)`` for us. Requires the
    P1 ``vol_field``; falls back to the local sum when it is not supplied.
    """
    if vol_field is not None and uw.mpi.size > 1:
        return _lumped_vertex_volumes(vol_field)
    area = np.abs(_signed_areas(coords, tris)) / 3.0
    patch = np.zeros(n_verts, dtype=np.float64)
    for k in range(3):
        np.add.at(patch, tris[:, k], area)
    patch[patch <= 0.0] = patch[patch > 0.0].mean()
    return patch


def _lumped_vertex_volumes(vol_field):
    r"""Parallel-correct per-vertex dual-patch volume = the lumped P1 mass
    diagonal ``M_ii = ∫ N_i dV`` of ``vol_field``'s (P1, continuous, scalar)
    space, assembled via the FE mass matrix so the cross-rank sum over shared
    partition-boundary vertices is done by PETSc — unlike the hand-rolled local
    sum in :func:`_patch_volumes`, which under-counts those vertices in parallel.

    Identity: by partition of unity (``Σ_j N_j ≡ 1``) the consistent mass matrix
    has row sums ``Σ_j M_ij = ∫ N_i Σ_j N_j = ∫ N_i dV``, i.e. the lumped diagonal
    is ``M·1``.

    TODO(petsc4py): PETSc has a purpose-built
    ``DMCreateMassMatrixLumped(dm, &llm, &lm)`` that returns this lumped diagonal
    directly (with the cross-rank ADD built in), but petsc4py (3.25) does not bind
    it yet — only the *consistent* ``DM.createMassMatrix`` is exposed, hence the
    ``M·1`` below. Replace this body with ``subdm.createMassMatrixLumped()`` once
    petsc4py exposes that DM method.

    Returns a per-vertex numpy array in ``vol_field``'s local DOF ordering (the
    same depth-0 vertex ordering the movers use for ``vol_field.array``).
    """
    mesh = vol_field.mesh
    indexset, subdm = mesh.dm.createSubDM(vol_field.field_id)
    M = subdm.createMassMatrix(subdm)      # consistent P1 mass (FE-assembled, parallel-correct)
    ones = M.createVecRight()
    ones.set(1.0)
    lumped = M.createVecLeft()
    M.mult(ones, lumped)                   # M·1 = row sums = lumped diagonal
    lvec = subdm.getLocalVec()
    subdm.globalToLocal(lumped, lvec, addv=False)
    out = np.asarray(lvec.array).copy()
    subdm.restoreLocalVec(lvec)
    for obj in (M, ones, lumped, indexset, subdm):
        try:
            obj.destroy()
        except Exception:
            pass
    pos = out > 0.0
    if not pos.all():
        out[~pos] = out[pos].mean()
    return out


def _hessian_recovery_class():
    r"""Lazily build (and memoise) the variationally-consistent
    Hessian-recovery solver class.

    Recovers ``H_ij ≈ ∂²φ/∂x_i∂x_j`` from an external scalar field
    ``φ`` by the *weak* (integrated-by-parts) form — the plan's
    :math:`R_H`: ``∫H_ij τ_ij + ∫(∂φ/∂x_i)(∂τ_ij/∂x_j) = 0`` ⇒
    ``H_ij = ∂²φ/∂x_i∂x_j``. Only **first** derivatives of ``φ``
    appear (UW3 forbids second derivatives of mesh-variable
    functions); the operator is the SPD mass matrix (no nullspace).
    Defined lazily to avoid an import cycle (meshing→systems/cython).
    """
    global _HESSIAN_CLASS
    if _HESSIAN_CLASS is not None:
        return _HESSIAN_CLASS

    import sympy
    from underworld3.cython.generic_solvers import SNES_MultiComponent
    from underworld3.utilities._api_tools import Template

    class _HessianRecovery(SNES_MultiComponent):
        def __init__(self, mesh, phi_field, degree=2, verbose=False):
            self._phi = phi_field
            super().__init__(
                mesh, n_components=mesh.cdim * mesh.cdim,
                degree=degree, verbose=verbose)
            self._smoothing = sympy.sympify(0)
            self._constitutive_model = (
                uw.constitutive_models.Constitutive_Model(
                    self.Unknowns))

        def _hessian_source(self):
            cdim = self.mesh.cdim
            X = self.mesh.CoordinateSystem.X
            phi = self._phi.sym[0]
            rows = []
            for i in range(cdim):
                for j in range(cdim):
                    row = [sympy.Integer(0)] * cdim
                    row[j] = phi.diff(X[i])
                    rows.append(row)
            return sympy.Matrix(rows)

        F0 = Template(
            r"f_0\left(\mathbf{u}\right)",
            lambda self: self.u.sym,
            "Hessian-recovery mass term: f_0 = H.")

        F1 = Template(
            r"\mathbf{F}_1\left(\mathbf{u}\right)",
            lambda self: self._hessian_source(),
            "Hessian-recovery weak source: F_1 = e_j ∂φ/∂x_i.")

    _HESSIAN_CLASS = _HessianRecovery
    return _HESSIAN_CLASS


def _monge_ampere_mover(mesh, metric, pinned_labels, verbose,
                      n_outer=1, n_picard=25, relax=1.0,
                      step_frac=None, picard_relax=0.4,
                      outer_tol=1.0e-3, boundary_slip=False,
                      linear_solver="direct", phi_degree=2,
                      move_anisotropy=None,
                      target_side_rho=False):
    r"""Metric-driven mesh equidistribution — Benamou–Froese–Oberman
    convex-branch Monge–Ampère (PRESERVED; not the default path).

    Solves ``det(I+D²φ)=g``, ``g=c·ρ_cur/ρ_tgt``, by a damped Picard
    on the convex-branch source
    ``Δφ = √((φxx−φyy)²+4φxy²+4g) − 2`` (the +√ selects the Brenier
    branch), with the variationally-consistent recovered Hessian
    (``_hessian_recovery_class``) and the pure-Neumann
    ``constant_nullspace`` φ Poisson. ``n_outer>1`` composes maps
    (recompute ρ_cur from patch volumes each step). Moves nodes by
    ∇φ with a coherent global signed-area backtrack.

    Efficiency (2026-05-17): the φ Poisson and the SPD Hessian-recovery
    mass system are *constant operators* within the Picard loop (the
    mesh is fixed; only the RHS changes). ``_use_direct_solver`` puts
    both on MUMPS LU with a lagged (compute-once) factorisation, so the
    inner iterations are back-substitutions — see that function's
    docstring. ``n_picard`` defaults to 25: the deep/near grading is
    flat from iter ≈20 (4-dp identical at AMP 8 & 20), so 40 was pure
    overhead. Net: ~10× faster, grading/quality bit-for-bit unchanged.

    ``phi_degree`` defaults to **2** (was 3). The deep/near grading
    is set by the φ *order*, not the solver: P2 ≡ P3 to ~3 dp across
    AMP 0/2/8/20 (matches the recorded baseline; AMP=0 no-op exact;
    no tangle) while P2 halves the cost (smaller matrices — also
    helps the direct factorisation scale). P1 is **not**
    grading-equivalent (≈1.40 vs 1.71 at AMP=8 — ~18 % weaker); P2
    is the floor. ``linear_solver="gamg"`` is an experimental,
    documented-fragile parallel prototype (P3 was a major GAMG
    confound; even at P2 GAMG re-solve is erratic — see the design
    doc); ``"direct"`` (MUMPS, MPI-parallel) is the validated path.

    Grading: redistribution with a fixed node count reaches deep/near
    ≈1.5–1.8× for an 8–20× density target (the exact OT ~10× needs
    *more nodes* — a topology change, not this smoother). ``n_outer=1``
    is the safe default (AMP=0 exact no-op, never tangles). See the
    project memory + scripts/ma_*.py / ma_cost_grading.py.
    """
    import sympy

    pinned_labels = tuple(pinned_labels)
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    cStart, cEnd = dm.getHeightStratum(0)
    cone_size = dm.getConeSize(cStart) if cEnd > cStart else 0
    if linear_solver not in ("direct", "gamg"):
        raise ValueError(
            f"linear_solver must be 'direct' or 'gamg', "
            f"got {linear_solver!r}")
    phi_degree = int(phi_degree)
    aux_degree = max(1, phi_degree - 1)   # ∇φ / recovered-Hessian
    key = (id(mesh), pinned_labels,
           pEnd - pStart, cEnd - cStart, cone_size,
           linear_solver, phi_degree)

    cdim = mesh.cdim

    cache = _WINSLOW_CACHE.get(key)
    if cache is None:
        _wire = _solver_wiring(linear_solver)
        phi = uw.discretisation.MeshVariable(
            f"winslow_phi_{id(mesh)}", mesh,
            vtype=uw.VarType.SCALAR, degree=phi_degree,
            continuous=True)
        ps = uw.systems.Poisson(mesh, phi)
        ps.constitutive_model = uw.constitutive_models.DiffusionModel
        ps.constitutive_model.Parameters.diffusivity = 1.0
        ps.constant_nullspace = True
        _wire(ps, singular=True, elliptic=True)
        hsolver = _hessian_recovery_class()(
            mesh, phi, degree=aux_degree, verbose=False)
        hsolver.tolerance = 1.0e-6
        _wire(hsolver, elliptic=False)
        vol_field = uw.discretisation.MeshVariable(
            f"winslow_vol_{id(mesh)}", mesh,
            vtype=uw.VarType.SCALAR, degree=1, continuous=True)
        gradphi = uw.discretisation.MeshVariable(
            f"winslow_gphi_{id(mesh)}", mesh,
            vtype=uw.VarType.VECTOR, degree=aux_degree,
            continuous=True)
        gproj = uw.systems.Vector_Projection(mesh, gradphi)
        gproj.smoothing = 0.0
        _wire(gproj, elliptic=False)
        _WINSLOW_CACHE[key] = (
            phi, ps, gradphi, gproj, hsolver, vol_field)
    else:
        phi, ps, gradphi, gproj, hsolver, vol_field = cache

    X = mesh.CoordinateSystem.X
    grad_phi = sympy.Matrix(
        [phi.sym[0].diff(X[i]) for i in range(cdim)]).T
    Hf = hsolver.u.sym
    Hmat = sympy.Matrix(cdim, cdim,
                        lambda i, j: Hf[i * cdim + j])
    gproj.uw_function = grad_phi
    omega = float(picard_relax)

    for outer in range(n_outer):
        dm = mesh.dm
        tris = _tri_cells(dm)
        pStart, pEnd = dm.getDepthStratum(0)
        n_verts = pEnd - pStart
        old_coords = np.asarray(mesh.X.coords).copy()
        _cdim = mesh.cdim

        # Boundary tangential slip via the mesh-owned contract
        # (boundary-slip-strategy.md): MA's natural Neumann BC (∇φ·n̂=0) makes
        # ∇φ tangential at the boundary, so slip vertices slide along their
        # surface (radial ring / box face / facet) and snap back; non-slip,
        # junction, and degenerate-normal vertices pin. Replaces the inline
        # per-ring / box-edge snap (the 'ring'/'box' hint is now inferred from
        # the registered bounding surfaces).
        is_pinned, _project = mesh.boundary_slip(
            boundary_slip, reference_coords=old_coords,
            boundary_labels=pinned_labels)

        if tris is not None and n_outer > 1:
            patch = _patch_volumes(tris, old_coords, n_verts, vol_field)
            patch /= float(np.mean(patch))
        else:
            patch = np.ones(n_verts, dtype=np.float64)
        _va = vol_field.array
        _va[...] = patch.reshape(_va.shape)

        rho_t = np.asarray(
            uw.function.evaluate(metric, old_coords)).reshape(-1)
        b = rho_t * patch
        inv_sqrt_b_mean = _global_mean(np.mean(1.0 / np.sqrt(b)))
        c = 1.0 / (inv_sqrt_b_mean ** 2)

        # Target-side ρ evaluation: substitute X[i] → X[i] +
        # gradphi.sym[i] so ρ is queried at the moving target
        # x + ∇φ(x), not the source x. Removes the phase error
        # where refinement-by-size is transported away from the
        # feature location by ∇φ. gradphi.sym values are updated
        # each Picard iter (gproj.solve below) so the source self-
        # consistently tracks the current map estimate.
        if target_side_rho:
            metric_target = metric.subs(
                [(X[i], X[i] + gradphi.sym[i])
                 for i in range(cdim)])
        else:
            metric_target = metric
        g = c / (metric_target * vol_field.sym[0])
        if cdim == 2:
            Hxx = Hf[0]
            Hxy = (Hf[1] + Hf[2]) / 2
            Hyy = Hf[3]
            f_src = sympy.sqrt(
                (Hxx - Hyy) ** 2 + 4 * Hxy ** 2 + 4 * g) - 2
        else:
            f_src = (g - 1.0) - Hmat.det()
        ps.f = sympy.Matrix([[_EQUIDIST_SIGN * f_src]])

        hsolver.u.array[...] = 0.0

        zero_init_guess = not _warm_start_krylov(linear_solver)
        prev_change = None
        # If target-side ρ is on, gradphi needs to be tracking the
        # current φ inside the Picard loop (it's used by ps.f via
        # the X→X+gradphi substitution). Initialise to zero so the
        # first ps.solve sees ρ at source (= identity map estimate).
        if target_side_rho:
            gradphi.array[...] = 0.0
        for it in range(n_picard):
            phi_prev = np.asarray(phi.array).copy()
            ps.solve(zero_init_guess=zero_init_guess)
            phi.array[...] = ((1.0 - omega) * phi_prev
                              + omega * np.asarray(phi.array))
            hsolver.solve()
            if target_side_rho:
                gproj.solve()   # update target-side ρ for next iter
            change = float(np.abs(
                np.asarray(phi.array) - phi_prev).max())
            change = _global_max(change)
            if prev_change is not None and change < 1.0e-6:
                break
            prev_change = change

        if not target_side_rho:
            gproj.solve()
        disp = np.asarray(
            uw.function.evaluate(gradphi.sym, old_coords)
        ).reshape(old_coords.shape)

        # Directional move-weighting (opt-in; default None ⇒ unchanged).
        if move_anisotropy is not None and cdim == 2:
            disp = _reweight_displacement_radial_tangential(
                disp, old_coords, move_anisotropy)

        step = _cap_step_to_edge_fraction(
            relax * disp, dm, old_coords, step_frac)

        free = ~is_pinned
        # _project: slip → ring (∥ only)
        new_coords, scale = _backtracked_move(
            old_coords, step, free, tris, _project)

        mesh._deform_mesh(new_coords)

        d = float(np.linalg.norm(
            new_coords - old_coords, axis=1).max())
        if uw.mpi.size > 1:
            d = _global_sum(d ** 2) ** 0.5
        if verbose:
            uw.pprint(
                f"  equidistribute MA outer {outer+1}/{n_outer}: "
                f"c={c:.4f}  scale={scale:.3f}  max|Δx|={d:.3e}")
        if d < outer_tol:
            break


def _ot_improvement_step(mesh, metric, pinned_labels, verbose,
                             n_outer=1, relax=1.0,
                             step_frac=0.3,
                             outer_tol=1.0e-4,
                             boundary_slip=False,
                             linear_solver="direct", phi_degree=2):
    r"""OT-improvement step: one (or a few) weighted-Poisson
    equidistribution flow iterations.

    Solves on the *current* mesh

    .. math::

        \nabla\!\cdot(\rho\,\nabla\phi)
            \;=\;-\,\rho\,\log\!\bigl(V\rho/K\bigr),
        \quad K=\exp(\langle\rho\log(V\rho)\rangle/\langle\rho\rangle),
        \quad \nabla\phi\cdot\hat{n}=0,

    and moves nodes by ``relax · ∇φ``. ``V_i`` is the dual patch
    area at vertex ``i``; the source vanishes identically at
    equidistribution ``V_i\,\rho_i\equiv K``.

    Semantics: this is a *single OT improvement step* w.r.t. the
    current mesh — the input mesh has no special status (it is
    whatever you currently have). Calling it again from the
    deformed mesh applies another improvement step. Compose
    freely with spring / smoothing / anisotropic.

    Differences from ``_monge_ampere_mover`` (the convex-branch
    BFO Picard):

    * Linear: one weighted-Poisson per outer iter, no inner
      Picard, no Hessian recovery, no convex-branch radical.
    * The source uses the *current* mesh's patch volumes; the
      formulation is identically zero at equidistribution, so
      iterations are self-stabilising (no over-correction).
    * ρ at the current node positions (no source-vs-target
      asymmetry; the iteration is on the current mesh, ρ is at
      its physical position).

    Parameters mirror ``_monge_ampere_mover`` where they apply.
    ``n_outer`` composes outer improvement steps; the source
    drives toward zero so the per-iter motion naturally
    diminishes.
    """
    import sympy

    pinned_labels = tuple(pinned_labels)
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    cStart, cEnd = dm.getHeightStratum(0)
    cone_size = dm.getConeSize(cStart) if cEnd > cStart else 0
    if linear_solver not in ("direct", "gamg"):
        raise ValueError(
            f"linear_solver must be 'direct' or 'gamg', "
            f"got {linear_solver!r}")
    phi_degree = int(phi_degree)
    aux_degree = max(1, phi_degree - 1)
    cdim = mesh.cdim
    if cdim != 2:
        raise NotImplementedError(
            "_ot_improvement_step: 2D meshes only for now.")

    # Boundary slip uses the projected boundary-normal field
    # (mesh.Gamma_P1). This is reliable only for *radial* coordinate
    # systems (cylindrical / spherical / geographic), where mesh.Gamma is
    # the coordinate-derived radial field and evaluates cleanly at vertices.
    # For Cartesian boundaries the vertex-evaluated facet normal is
    # degenerate (0/0), so we pin the boundary instead of slipping with a
    # garbage normal. 'ring'/'box'/'axes' are legacy aliases for slip-on.
    from underworld3.meshing._ot_adapt import _is_radial_coords as _isr
    if isinstance(boundary_slip, str):
        _slip_req = boundary_slip.strip().lower() in (
            "ring", "box", "axes", "axis", "true", "on", "1")
    else:
        _slip_req = bool(boundary_slip)
    _slip_on = _slip_req and _isr(mesh)
    if _slip_on:
        # Create / refresh the projected normals ONCE here, before the OT
        # Poisson solver's DM is built — creating the _n_proj MeshVariable
        # mid-mover would stale that DM handle (project_uw3_smoother_footguns).
        try:
            mesh._update_projected_normals()
        except Exception:
            _slip_on = False

    key = (id(mesh), pinned_labels,
           pEnd - pStart, cEnd - cStart, cone_size,
           linear_solver, phi_degree)

    cache = _OT_CACHE.get(key)
    if cache is None:
        _wire = _solver_wiring(linear_solver)
        phi = uw.discretisation.MeshVariable(
            f"ot_phi_{id(mesh)}", mesh,
            vtype=uw.VarType.SCALAR, degree=phi_degree,
            continuous=True)
        ps = uw.systems.Poisson(mesh, phi)
        ps.constitutive_model = uw.constitutive_models.DiffusionModel
        # weighted diffusion: D(x) = ρ(x). Updated each outer iter
        # via the symbolic metric expression (evaluated at the
        # current mesh's quad pts).
        ps.constitutive_model.Parameters.diffusivity = metric
        ps.constant_nullspace = True
        _wire(ps, singular=True, elliptic=True)
        vol_field = uw.discretisation.MeshVariable(
            f"ot_vol_{id(mesh)}", mesh,
            vtype=uw.VarType.SCALAR, degree=1, continuous=True)
        gradphi = uw.discretisation.MeshVariable(
            f"ot_gphi_{id(mesh)}", mesh,
            vtype=uw.VarType.VECTOR, degree=aux_degree,
            continuous=True)
        gproj = uw.systems.Vector_Projection(mesh, gradphi)
        gproj.smoothing = 0.0
        _wire(gproj, elliptic=False)
        X = mesh.CoordinateSystem.X
        gradphi_sym = sympy.Matrix(
            [phi.sym[0].diff(X[i]) for i in range(cdim)]).T
        gproj.uw_function = gradphi_sym
        _OT_CACHE[key] = (phi, ps, gradphi, gproj, vol_field)
    else:
        phi, ps, gradphi, gproj, vol_field = cache

    zero_init_guess = not _warm_start_krylov(linear_solver)

    for outer in range(n_outer):
        dm = mesh.dm
        tris = _tri_cells(dm)
        pStart, pEnd = dm.getDepthStratum(0)
        n_verts = pEnd - pStart
        old_coords = np.asarray(mesh.X.coords).copy()
        _cdim = mesh.cdim

        # Boundary tangential slip via the mesh-owned contract
        # (boundary-slip-strategy.md). Slip stays gated to radial meshes via
        # ``_slip_on`` (a Cartesian boundary pins — the vertex-evaluated facet
        # normal is degenerate there, see above); on a radial mesh the
        # registered radial surfaces do the tangent slide + |r| restore.
        is_pinned, _project = mesh.boundary_slip(
            boundary_slip if _slip_on else False,
            reference_coords=old_coords, boundary_labels=pinned_labels)

        # --- compute V (patch volumes) on current mesh ---------
        if tris is None:
            patch = np.ones(n_verts, dtype=np.float64)
        else:
            patch = _patch_volumes(tris, old_coords, n_verts, vol_field)
        # Normalise so the mean over the domain is the cell mean.
        patch_mean = _global_mean(np.mean(patch))
        # Write current V values into the MeshVariable.
        _va = vol_field.array
        _va[...] = (patch / max(patch_mean, 1e-30)).reshape(_va.shape)

        # --- compute K = exp(<ρ log(Vρ)> / <ρ>) ----------------
        rho_at_y = np.asarray(uw.function.evaluate(
            metric, old_coords)).reshape(-1)
        Vrho = (patch / max(patch_mean, 1e-30)) * rho_at_y
        # weighted geometric mean (zero-mean Neumann compat
        # condition) — guard against Vrho≤0:
        Vrho_pos = np.clip(Vrho, 1e-30, None)
        wnum = float(np.sum(rho_at_y * np.log(Vrho_pos)))
        wden = float(np.sum(rho_at_y))
        wnum = _global_sum(wnum)
        wden = _global_sum(wden)
        ln_K = wnum / max(wden, 1e-30)
        K_val = float(np.exp(ln_K))

        # --- source: f = -ρ · log(V·ρ / K) ---------------------
        # SNES_Poisson convention: F0 = -f, strong form ∇·(D∇u)
        # = -ps.f. We want ∇·(ρ∇φ) = -ρ·log(V·ρ/K) ⇒ ps.f =
        # ρ·log(V·ρ/K).
        f_src = metric * sympy.log(
            metric * vol_field.sym[0] / sympy.Float(K_val))
        ps.f = sympy.Matrix([[f_src]])

        # --- solve weighted Poisson ----------------------------
        ps.solve(zero_init_guess=zero_init_guess)
        gproj.solve()
        disp = np.asarray(uw.function.evaluate(
            gradphi.sym, old_coords)
        ).reshape(old_coords.shape)

        step = _cap_step_to_edge_fraction(
            float(relax) * disp, dm, old_coords, step_frac)

        # --- coherent global signed-area backtrack -------------
        free = ~is_pinned
        new_coords, scale = _backtracked_move(
            old_coords, step, free, tris, _project)

        mesh._deform_mesh(new_coords)

        d = float(np.linalg.norm(
            new_coords - old_coords, axis=1).max())
        if uw.mpi.size > 1:
            d = _global_sum(d ** 2) ** 0.5

        # Per-iter "imbalance" diagnostic — std of log(V·ρ/K).
        imb = float(np.std(np.log(Vrho_pos) - ln_K))
        if uw.mpi.size > 1:
            imb_sq = _global_sum(imb * imb)
            cnt = int(_global_sum(Vrho_pos.size))
            imb = (imb_sq / max(cnt, 1)) ** 0.5

        if verbose:
            uw.pprint(
                f"  OT-improve outer {outer+1}/{n_outer}: "
                f"K={K_val:.4f}  imb={imb:.3e}  "
                f"scale={scale:.3f}  max|Δx|={d:.3e}")
        if d < outer_tol:
            break
