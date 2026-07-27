"""The Huang–Kamenski variational MMPDE mover (recommended for
production adaptive meshing; triangle and tetrahedral meshes). See the
package docstring for the module map.
"""

import math
import warnings

import numpy as np

import underworld3 as uw

from .graph import (_tri_cells, _tet_cells, _signed_areas, _signed_volumes,
                    _owned_cell_mask,
                    _owned_vertex_mask, _min_incident_edge_nd,
                    _global_sum, _global_min, _global_max, _global_mean)


def _spd_sanitise(M):
    """Project a batch of metric tensors ``(N, d, d)`` onto SPD.

    The MMPDE functional uses fractional powers (``sqrt(detM)``,
    ``detM**((1-p)/2)``, ``S**q``) that are defined only for a
    symmetric-positive-definite metric. FE extrapolation outside the current
    mesh can hand back a non-SPD tensor; this projection floors its
    eigenvalues at a small RELATIVE level so a genuine SPD metric passes
    through unchanged while garbage becomes a benign "coarsen here" tensor
    (tiny positive eigenvalues) instead of a NaN in the energy.

    The floor is relative to the batch's largest finite eigenvalue with an
    absolute minimum of 1e-8, so the pass-through guarantee holds for the
    O(1)-normalised metrics the mover uses; a metric batch whose valid
    eigenvalues sit far below unit scale would be floored too.
    """
    # Symmetrise: the metric is symmetric by construction, so for a valid
    # tensor this is an exact no-op (M_ij == M_ji bit-for-bit).
    Ms = 0.5 * (M + np.swapaxes(M, -1, -2))
    if Ms.shape[0] == 0:
        return Ms                                   # rank owns no cells
    try:
        w, Vc = np.linalg.eigh(Ms)
    except np.linalg.LinAlgError:
        # Degenerate-input eigh behaviour is LAPACK-path dependent: the 2x2
        # kernel returns quiet NaNs, the general kernel raises — and a
        # batched eigh raises for the WHOLE batch if any one tensor fails.
        # Retry per tensor so one degenerate tensor cannot take down its
        # neighbours; a tensor that still fails is marked non-finite and
        # rebuilt as the isotropic fallback below (#352).
        w = np.empty(Ms.shape[:-1])
        Vc = np.empty_like(Ms)
        for i, Mi in enumerate(Ms):
            try:
                w[i], Vc[i] = np.linalg.eigh(Mi)
            except np.linalg.LinAlgError:
                w[i] = np.nan
                Vc[i] = np.eye(Ms.shape[-1])
    wmax = float(np.nanmax(w[np.isfinite(w)], initial=-np.inf))
    if not np.isfinite(wmax):
        # Every eigenvalue on this rank is NaN/inf: the metric carries no
        # information here, so anchor the floor at the unit scale — the
        # projection below then rebuilds every tensor as the same benign
        # isotropic fallback instead of propagating NaN (#352).
        wmax = 1.0
    floor = max(wmax, 1.0) * 1.0e-8
    # Per-tensor SPD test: a cell is "bad" only if one of its OWN
    # eigenvalues is non-finite or below the floor. Project just those
    # cells; every already-SPD tensor is returned untouched (bit-identical
    # to the symmetrised input), so one bad point cannot perturb the rest.
    bad = ~np.isfinite(w).all(axis=1) | (w.min(axis=1) < floor)
    if not bad.any():
        return Ms
    out = Ms.copy()
    wf = np.clip(np.nan_to_num(w[bad], nan=floor, posinf=wmax, neginf=floor),
                 floor, None)
    # The eigenvector basis itself can be NaN for a fully-degenerate input
    # tensor; fall back to the identity basis there so the rebuilt tensor
    # is the isotropic floor metric, not NaN (#352).
    Vb = Vc[bad]
    nan_basis = ~np.isfinite(Vb).all(axis=(1, 2))
    if nan_basis.any():
        Vb = Vb.copy()
        Vb[nan_basis] = np.eye(Ms.shape[-1])
    out[bad] = np.einsum('nij,nj,nkj->nik', Vb, wf, Vb)
    return out



def _mmpde_mover(mesh, metric, pinned_labels, verbose,
                   n_outer=150, p=1.5, theta=1.0 / 3.0, tau=1.0,
                   step_frac=0.2, area_floor_frac=0.01,
                   boundary_slip=False, outer_tol=1.0e-7, tol=1.0e-3,
                   stol=None, stol_k=3,
                   fd_eps=1.0e-6, metric_eval="rbf", rbf_k=None,
                   accel="cg", momentum=0.0,
                   resolution_ratio=None, reference="mesh",
                   **_unknown_kwargs):
    r"""Anisotropic variational moving-mesh adaptation (Huang–Kamenski
    MMPDE; the direct simplex discretization of JCP 301 (2015) 322,
    arXiv:1410.7872). Triangle (2D) and tetrahedral (3D) meshes,
    parallel-safe. The per-element algebra is written over ``cdim``
    throughout — the two dimensions differ only in the cell list, the
    signed-measure primitive and the ``d!`` volume factor.

    Generates the physical mesh as the image of a **fixed computational
    (reference) mesh** under the inverse coordinate map, minimizing
    Huang's functional ``G = theta*sqrt(detM)*S**q + (1-2theta)*d**q *
    r**p * detM**((1-p)/2)`` with ``q = d*p/2``, ``S = tr(J Minv J^T)``,
    ``J = Ehat @ inv(E)``, ``r = det J``.
    Because `G → ∞` as `det𝕁 → 0` the map is non-folding (Math. Comp. 87
    (2018) 1887); because it is the inverse map of a convex computational
    domain it genuinely *clusters and aligns* to `M` — a thin strip on a
    fault, not an isotropic centre-of-gravity blob, and not a
    non-clustering decoupled Winslow smooth (the scalar Monge-Ampère and
    anisotropic-Winslow movers it superseded were retired 2026-07). See
    ``docs/developer/design/anisotropic-mmpde-mover.md``.

    ``metric`` is the SPD `d×d` metric tensor: a sympy `Matrix` (function
    of ``mesh.CoordinateSystem.X``) or a ``VarType.TENSOR`` /
    ``SYM_TENSOR`` :class:`MeshVariable`. Build it small **across** a
    feature (along its normal) and base along it, localized near the
    feature (e.g. `M = I + (R²-1)·exp(-(d_seg/W)²)·n nᵀ`).

    Parallel safety (release gate: `np>=2` must match serial): the
    per-element `d×d` algebra is rank-local (batched ``numpy.linalg``);
    the **velocity assembly** `Σ_{K∋i}|K|v^K_i` is summed over **owned
    cells** into the coordinate DM Vec with ``localToGlobal(ADD_VALUES)``
    + ``globalToLocal`` (cross-rank ghost reduction — not ``np.add.at``
    into a global array); the per-node step and the energy/area
    line-search predicates are computed from owned/assembled data with
    collective ``allreduce`` so every rank takes the same accept/backtrack
    branch; only owned vertices move and ghosts are halo-synced each trial
    so the final ``_deform_mesh`` is consistent.

    Time integration: gradient flow `dx_i/dt = (P_i/τ)Σ|K|v`,
    `P_i = detM(x_i)^{(p-1)/2}` (scale-free), explicit Euler with a
    per-node step cap (``step_frac``·min-incident-edge) and an **energy
    line-search backtrack** (accept only if no fold *and* `I_h`
    decreases) so the descent is monotone. ``n_outer`` Euler steps.

    The steepest-descent direction is accelerated by ``accel`` (default
    ``"cg"``, nonlinear conjugate gradient, parameter-free): this cuts the
    outer-iteration count ~13× on the first (uniform→radial) adapt vs plain
    descent and makes adapt-every-step affordable. ``"heavyball"`` /
    ``"hb-restart"`` use Polyak momentum with coefficient ``momentum`` (default
    0.9 for those modes); ``"none"`` is plain descent. The line-search keeps
    every accelerator fold-safe. (Previously controlled by the ``MMPDE_ACCEL`` /
    ``MMPDE_MOMENTUM`` environment variables, now removed — pass as kwargs, e.g.
    ``method_kwargs={"accel": "cg"}`` through ``smooth_mesh_interior``.)

    ``reference`` selects what the fixed computational mesh — the thing
    the physical mesh is the image OF — actually is, and so what the
    shape term measures deviation *from*:

    * ``"mesh"`` (default, the **redistribution** frame): the reference
      is the mesh at the first call. This is the classical MMPDE and the
      right frame for moving an existing well-shaped mesh onto a metric.
      Its fixed point is the mesh it started from: with ``M = I`` the
      identity map already minimises the functional, so calling the
      mover to "tidy up" a mesh moves nothing at all.
    * ``"ideal"`` (the **relaxation** frame): the reference is a single
      regular simplex, scaled to the reference mesh's typical cell
      volume. Deviation is then measured from *equilateral-in-the-metric*
      rather than from the supplied mesh, so distortion the mesh arrived
      with is no longer self-justifying. This is the frame to use after
      a topological refinement (``mesh.adapt``), where node positions
      were fixed by inherited combinatorics — bisection's needles and
      centroid refinement's slivers — rather than by any geometric
      criterion. Same metric, same grading, same non-folding guarantee;
      only the shape target moves.

    ``resolution_ratio`` is accepted but unused: the strategy dispatch in
    ``_smooth_mesh_interior_bare`` injects it into ``method_kwargs`` for
    every mover, and MMPDE's clustering intensity comes from the metric
    tensor itself. Any other unexpected keyword is warned about (READ-11:
    it is a caller typo, not a tunable) rather than silently swallowed.
    """
    import sympy
    from petsc4py import PETSc
    if _unknown_kwargs:
        warnings.warn(
            f"_mmpde_mover: ignoring unknown keyword argument(s) "
            f"{sorted(_unknown_kwargs)} — not MMPDE tunables (typo?)",
            stacklevel=2)
    pinned_labels = tuple(pinned_labels)
    cdim = mesh.cdim
    if cdim not in (2, 3):
        # Guard here, before any metric parsing or DM work, so the caller
        # gets an honest message rather than a failure deeper in the mover.
        raise NotImplementedError(
            "MMPDE mesh movement supports 2D (triangle) and 3D "
            f"(tetrahedral) simplex meshes. Got a mesh with cdim={cdim}.")
    p = float(p); theta = float(theta); tau = float(tau)
    q = cdim * p / 2.0
    dq = float(cdim) ** q
    parallel = uw.mpi.size > 1

    # --- metric as evaluable sympy entries -------------------------
    # Accept a full d×d SPD tensor (sympy Matrix or tensor MeshVariable) OR a
    # scalar density rho — the latter is coerced to the isotropic tensor rho*I,
    # so mmpde takes the same metric forms as the ma/ot/anisotropic movers.
    Msym = metric.sym if isinstance(metric, uw.discretisation.MeshVariable) else metric
    if not isinstance(Msym, sympy.MatrixBase):
        Msym = sympy.sympify(Msym)
    if not isinstance(Msym, sympy.MatrixBase):        # bare scalar expression
        Msym = sympy.eye(cdim) * Msym
    elif Msym.shape == (1, 1):                        # 1x1 (scalar MeshVariable)
        Msym = sympy.eye(cdim) * Msym[0, 0]
    if Msym.shape != (cdim, cdim):
        raise ValueError(
            f"_mmpde_mover metric must be {cdim}x{cdim} (or a scalar "
            f"density), got {Msym.shape}")

    def _eval_M_analytic(pts):
        """Exact Eulerian metric via sympy evaluate → (n, cdim, cdim).
        Correct but slow (sympy symbolic processing dominates the cost)."""
        n = pts.shape[0]
        out = np.empty((n, cdim, cdim))
        for a in range(cdim):
            for b in range(cdim):
                e = Msym[a, b]
                if getattr(e, "free_symbols", None):
                    out[:, a, b] = np.asarray(
                        uw.function.evaluate(e, pts)).reshape(-1)
                else:
                    out[:, a, b] = float(e)
        return out

    # `_eval_M` is (re)bound below once `ref` is known: either the exact
    # analytic path, or a bake-once + Shepard/RBF interpolation from the
    # FIXED reference cloud (Eulerian — the metric is a function of space,
    # so we interpolate from a static cloud to the moving centroids, NOT a
    # Lagrangian nodal field). RBF is ~10× faster per eval and smooths the
    # analytic endpoint "elbow" kink; the metric is a guide field so the
    # interpolation error costs no correctness (the line-search on I_h
    # keeps the move valid for whatever M it is handed).
    _eval_M = _eval_M_analytic

    def _dM_dx(cen):
        """∂M/∂x at centroids via centred FD on the analytic metric →
        (n, cdim, cdim, cdim) indexed [cell, a, b, component]."""
        n = cen.shape[0]
        d = np.zeros((n, cdim, cdim, cdim))
        for c in range(cdim):
            sh = np.zeros(cdim); sh[c] = fd_eps
            Mp = _eval_M(cen + sh)
            Mm = _eval_M(cen - sh)
            d[:, :, :, c] = (Mp - Mm) / (2.0 * fd_eps)
        return d

    # --- topology / parallel scaffolding ---------------------------
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    n_verts = pEnd - pStart
    # cdim in (2, 3) is guaranteed by the guard at the top; the rest of
    # the mover is written over cdim and needs only the right cell list,
    # signed-measure primitive and d! volume factor.
    cells_all = _tri_cells(dm) if cdim == 2 else _tet_cells(dm)
    signed_vol = _signed_areas if cdim == 2 else _signed_volumes
    if cells_all is None:
        # TODO(BUG): rank-LOCAL early return — under MPI a rank with zero
        # local cells (or a non-simplex local patch) returns here while the
        # other ranks enter the collective energy/line-search reductions
        # below, hanging the job. Pre-existing 2D posture, inherited by 3D
        # (2026-07 review note); the fix is a collective emptiness check.
        return
    fact = float(math.factorial(cdim))         # d! → |K| = |detE|/d!
    owned_cell = _owned_cell_mask(dm)
    cells_own = cells_all[owned_cell]
    is_owned_v = _owned_vertex_mask(dm)

    coord_dm = dm.getCoordinateDM()
    local_vec = dm.getCoordinatesLocal()
    global_vec = dm.getCoordinates()
    vloc = coord_dm.getLocalVec()
    vglob = coord_dm.getGlobalVec()

    coords = np.asarray(local_vec.array, dtype=np.float64).reshape(-1, cdim).copy()

    # Fixed computational reference = coords at first call, cached on mesh
    # (ghosted: this rank's local array including halo).
    ref = getattr(mesh, "_mmpde_reference_coords", None)
    if ref is None or ref.shape != coords.shape:
        ref = coords.copy()
        mesh._mmpde_reference_coords = ref

    # --- RBF/Shepard bake of the metric (the production-fast path) ------
    # Bake the metric at the CURRENT mesh NODES (its own DOF locations), then
    # interpolate to the moving centroids each step via k-NN inverse-distance
    # (Shepard). Source = nodes, NOT the fixed reference cloud `ref` (`ref` is
    # kept for the _edge_mats reference frame). Two reasons:
    #   * MONOTONE: a P1 density is positive by construction; Shepard is a convex
    #     (positive-weight) average of the sampled node values, so the result is
    #     GUARANTEED positive — no negative/non-SPD garbage, the SPD floor / NaN
    #     bail never has to fire.
    #   * ROBUST + FAST: nodes are always inside the mesh (never out-of-domain),
    #     and Shepard needs no per-step cell location. RBF doesn't need
    #     high-precision eval — speed + monotonicity. (Restores the earlier
    #     "RBF metric eval" design intent: the fixed-`ref` FE bake could
    #     mis-locate / drift outside a deformed interior and return ρ<0.)
    if metric_eval == "rbf":
        from scipy.spatial import cKDTree
        M_src = _eval_M_analytic(coords)                 # nodal values (positive)
        _tree = cKDTree(coords)
        _kk = int(rbf_k) if rbf_k else (cdim + 2)

        def _eval_M(pts):
            dist, idx = _tree.query(pts, k=_kk)
            if _kk == 1:
                return M_src[idx]
            w = 1.0 / np.maximum(dist, 1.0e-12) ** 2
            w /= w.sum(axis=1, keepdims=True)
            return np.einsum('nk,nkab->nab', w, M_src[idx])
    else:
        _eval_M = _eval_M_analytic

    # --- SPD sanitiser on the evaluated metric -------------------------
    # The metric is a guide field FE-evaluated at the FIXED reference
    # cloud; once the interior has deformed, a reference point can fall
    # OUTSIDE the current mesh and the P1 metric field is then evaluated
    # by FE EXTRAPOLATION (out-of-cell basis functions go negative),
    # yielding a non-SPD tensor — e.g. a scalar density ρ·I with ρ<0
    # whose determinant ρ² still passes a detM>0 test. Project every
    # evaluated tensor onto SPD (module-level _spd_sanitise) so the
    # fractional powers in the energy stay finite.
    _eval_M_raw = _eval_M

    def _eval_M(pts):
        return _spd_sanitise(_eval_M_raw(pts))

    # Mesh-owned boundary slip is applied per outer iter via mesh.boundary_slip
    # (below). Pre-touch Gamma_P1 here so the projected-normal MeshVariable
    # exists before any DM snapshot (footgun-safe; redundant with the central
    # pre-touch in smooth_mesh_interior, kept as defence-in-depth).
    from .graph import _resolve_slip
    _slip_pretouch = _resolve_slip(mesh, boundary_slip)  # pre-touch Gamma_P1 before DM build

    # Reference edge matrices (fixed) for the owned cells.
    def _edge_mats(X, cells):
        pc = X[cells]                               # (Nc, d+1, d)
        cols = [pc[:, k + 1] - pc[:, 0] for k in range(cdim)]
        return np.stack(cols, axis=2)               # (Nc, d, d) columns
    Eh = _edge_mats(ref, cells_own)
    detEh = np.linalg.det(Eh)

    if reference in ("ideal", "ideal-metric"):
        # RELAXATION frame (see the ``reference`` docstring): replace the
        # per-cell reference by ONE regular simplex, so "deviation from the
        # reference" means "deviation from equilateral-in-the-metric"
        # instead of "deviation from the mesh I was handed". This is what
        # lets the mover repair the needles/slivers that refinement leaves
        # behind: with reference="mesh" a distorted start is its own
        # optimum (𝕁 = I ⇒ zero shape gradient), which is why a
        # post-refinement mover call moves nothing under M = I.
        # Only the SHAPE target changes — the size/equidistribution term
        # (r = detEh/detE against detM) is untouched, so the metric still
        # sets the grading exactly as before.
        # Unit regular simplex, columns c_k = v_k - v_0: |c_k| = 1 and
        # c_i·c_j = 1/2 (so every edge, including v_i-v_j, has length 1).
        # That Gram matrix factors exactly, so take its Cholesky.
        G = 0.5 * (np.eye(cdim) + np.ones((cdim, cdim)))
        R = np.linalg.cholesky(G).T                 # R^T R = G
        detR = math.sqrt(cdim + 1.0) / 2.0 ** (cdim / 2.0)
        # Scale EACH cell's reference to ITS OWN current volume, and match
        # the cell orientation. Then r = detEh/detE = 1 for every cell at
        # entry, so the size/equidistribution term already sits at its
        # optimum and the existing size distribution is what gets
        # preserved — the mover has nothing to say about size and only
        # the shape term does work. That is what makes this a RELAXATION
        # (keep the resolution the refinement installed, repair the
        # element shapes it had no way to choose) rather than a
        # re-adaptation. Under M = I this is pure shape repair.
        # Per-cell orientation (flip one column where the cell is negative)
        # and per-cell scale, so det(Eh) == detEh exactly, cell by cell.
        # No reduction and no median => correct on a rank that owns no
        # cells, where a global orientation vote would have to be faked.
        _sgn = np.sign(detEh)
        _sgn[_sgn == 0.0] = 1.0
        Rb = np.broadcast_to(R, (detEh.shape[0], cdim, cdim)).copy()
        Rb[_sgn < 0.0, :, -1] *= -1.0
        if reference == "ideal":
            _vol = np.abs(detEh)                    # keep each cell's size
        else:
            # "ideal-metric": ONE reference volume for every cell, so the
            # metric alone sets size. Needed whenever the sizes we were
            # handed are themselves the problem — a sliver cannot be
            # repaired while being told to keep its (near-zero) volume,
            # because its volume is half of what is wrong with it.
            _n = int(detEh.shape[0])
            _loc = float(np.median(np.abs(detEh))) if _n else 0.0
            _tot = _global_sum(float(_n))
            _vol = np.full(_n, (_global_sum(_loc * _n) / _tot) if _tot else 1.0)
        Eh = ((_vol / detR) ** (1.0 / cdim))[:, None, None] * Rb
        detEh = np.linalg.det(Eh)
    elif reference != "mesh":
        raise ValueError(
            f"_mmpde_mover: reference={reference!r}; use 'mesh' (the "
            f"metric-redistribution frame, default), 'ideal' (shape "
            f"relaxation at fixed per-cell size) or 'ideal-metric' "
            f"(shape relaxation with the metric setting size).")

    a0 = signed_vol(coords, cells_all)
    orient = np.sign(np.median(a0)) or 1.0
    # Collapse guard reference: EACH cell's own starting volume. The floor
    # is then "no cell may shrink below area_floor_frac of what it started
    # as" -- scale-free and grading-free.
    #
    # It used to be one absolute floor, area_floor_frac x the median cell
    # volume, which is only meaningful on a near-uniform mesh. On a graded
    # mesh (anything out of mesh.adapt) the finest cells are orders of
    # magnitude below the median and so start *already under* the floor:
    # every trial step is rejected, the line search backtracks to scale=0
    # and the mover silently does nothing. Worse, the median was a
    # _global_mean of per-rank medians, so which cells fell foul of it
    # depended on the partition -- the same mesh moved in serial and stood
    # still on 2 ranks. See #419.
    a0_own = np.abs(signed_vol(coords, cells_own))
    a0_own = np.maximum(a0_own, 1.0e-300)
    # Representative background cell size h0 (mean reference edge length over
    # owned cells), used to make the convergence test SCALE-RELATIVE: a move
    # of dmax < tol·h0 is negligible vs the cell size, so the adapt has
    # converged regardless of absolute coordinate units. (The old absolute
    # outer_tol=1e-7 never fired — dx~1e-6 ≫ 1e-7 yet ≪ h0~0.08 — so every
    # adapt ran to the n_outer cap.)
    _ecols = np.linalg.norm(Eh, axis=1)            # (n_own, cdim) edge lengths
    h0_scale = float(np.mean(_ecols)) if _ecols.size else 1.0
    h0_scale = _global_mean(h0_scale)

    def _halo_sync(X):
        """Make ghost vertices exact copies of their owners."""
        if not parallel:
            return X
        local_vec.array[:] = X.ravel()
        coord_dm.localToGlobal(local_vec, global_vec, addv=False)
        coord_dm.globalToLocal(global_vec, local_vec)
        return np.asarray(local_vec.array).reshape(-1, cdim).copy()

    def _energy(X):
        """I_h = Σ_owned |K| G (collective)."""
        E = _edge_mats(X, cells_own)
        detE = np.linalg.det(E)
        Einv = np.linalg.inv(E)
        J = np.einsum('mij,mjk->mik', Eh, Einv)
        r = detEh / detE
        cen = X[cells_own].mean(axis=1)
        M = _eval_M(cen); Minv = np.linalg.inv(M); detM = np.linalg.det(M)
        JMi = np.einsum('mij,mjk->mik', J, Minv)
        S = np.einsum('mij,mij->m', JMi, J)
        G = (theta * np.sqrt(detM) * S ** q
             + (1.0 - 2.0 * theta) * dq * r ** p * detM ** ((1 - p) / 2))
        K = np.abs(detE) / fact
        return _global_sum(np.sum(K * G))

    def _min_area_frac(X):
        """Smallest cell volume as a FRACTION of that cell's starting
        volume (global). > 0 also certifies no cell has folded."""
        return _global_min(
            (signed_vol(X, cells_own) * orient / a0_own).min(initial=np.inf))

    prevI = _energy(coords)
    _Iwin = [prevI]   # accepted-energy history for the stol stagnation test
    # Acceleration of the first-order steepest-descent direction (``accel``).
    # The energy+min-area line-search below stays the fold guard, so any
    # accelerator overshoot is backtracked — never tangles (verified fold-proof
    # even at step_frac=2). ``accel`` in {"none","heavyball","hb-restart","cg"}:
    #   none      : plain steepest descent
    #   heavyball : step += momentum * previous accepted displacement (Polyak);
    #               ``momentum`` defaults to 0.9 if left at 0 for this mode
    #   hb-restart: heavyball + gradient restart (drop momentum when it opposes
    #               the descent direction — O'Donoghue & Candès robustness)
    #   cg        : nonlinear conjugate gradient (Polak-Ribière+), parameter-free
    #               — the default (≈13× fewer outer iters than plain descent on
    #               the first radial adapt, best mesh quality, no tuning).
    _accel = str(accel).lower() if accel is not None else "none"
    _valid_accel = ("none", "heavyball", "hb-restart", "cg")
    if _accel not in _valid_accel:
        raise ValueError(
            f"_mmpde_mover: unknown accel {accel!r}; "
            f"choose from {_valid_accel}")
    _mmpde_beta = float(momentum)
    if _accel in ("heavyball", "hb-restart") and _mmpde_beta == 0.0:
        _mmpde_beta = 0.9
    _prev_disp = np.zeros_like(coords)
    _prev_v = np.zeros_like(coords)
    _prev_dir = np.zeros_like(coords)

    def _gdot(a, b, mask):
        return _global_sum(np.sum(a[mask] * b[mask]))

    for outer in range(n_outer):
        # Mesh-owned tangent slip (see boundary-slip-strategy.md): the
        # reference is the current coords (refreshed each outer iter), so the
        # tangent slide / surface restore are measured from this iteration's
        # mesh — matching the previous per-iter _build_slip_projector build.
        is_pinned, _project = mesh.boundary_slip(
            boundary_slip, reference_coords=coords,
            boundary_labels=pinned_labels)
        free = ~is_pinned

        # --- per-element terms on owned cells (rank-local d×d algebra) -
        E = _edge_mats(coords, cells_own)
        detE = np.linalg.det(E)
        Einv = np.linalg.inv(E)
        J = np.einsum('mij,mjk->mik', Eh, Einv)
        r = detEh / detE
        cen = coords[cells_own].mean(axis=1)
        M = _eval_M(cen); Minv = np.linalg.inv(M); detM = np.linalg.det(M)
        sdetM = np.sqrt(detM)
        JMi = np.einsum('mij,mjk->mik', J, Minv)
        S = np.einsum('mij,mij->m', JMi, J)
        G = (theta * sdetM * S ** q
             + (1.0 - 2.0 * theta) * dq * r ** p * detM ** ((1 - p) / 2))
        K = np.abs(detE) / fact
        # ∂G/∂𝕁 = 2qθ√detM S^{q-1} M⁻¹ 𝕁ᵀ ; ∂G/∂r = p(1-2θ)dq detM^{(1-p)/2} r^{p-1}
        MinvJT = np.einsum('mij,mkj->mik', Minv, J)
        dGdJ = (2.0 * q * theta * sdetM * S ** (q - 1.0))[:, None, None] * MinvJT
        dGdr = (p * (1.0 - 2.0 * theta) * dq
                * detM ** ((1 - p) / 2) * r ** (p - 1.0))
        # local vertex velocities: V rows = -G E⁻¹ + E⁻¹ dGdJ Eh E⁻¹ + dGdr r E⁻¹
        mid = np.einsum('mij,mjk,mkl,mln->min', Einv, dGdJ, Eh, Einv)
        V = (-G[:, None, None] * Einv + mid
             + (dGdr * r)[:, None, None] * Einv)        # (Nc, d, d): rows v1..vd
        # grad_i (G+Jacobian part) = -Σ |K| v ; v0 = -(Σ_k vk)
        vrows = V                                        # rows index local vert 1..d
        v0 = -vrows.sum(axis=1)                          # (Nc, d)
        grad_loc = np.zeros((n_verts, cdim))
        np.add.at(grad_loc, cells_own[:, 0], -(K[:, None] * v0))
        for k in range(cdim):
            np.add.at(grad_loc, cells_own[:, k + 1],
                      -(K[:, None] * vrows[:, k, :]))

        # --- metric-variation term ∂G/∂M : ∂M/∂x (ESSENTIAL on the feature)
        # ∂G/∂M = θ√detM[½Sq M⁻¹ - q S^{q-1} M⁻¹ 𝕁ᵀ𝕁 M⁻¹]
        #         + (1-2θ)dq rᵖ (1-p)/2 detM^{(1-p)/2} M⁻¹
        JTJ = np.einsum('mji,mjk->mik', J, J)
        MJTJM = np.einsum('mij,mjk,mkl->mil', Minv, JTJ, Minv)
        dGdM = (theta * sdetM)[:, None, None] * (
            0.5 * (S ** q)[:, None, None] * Minv
            - q * (S ** (q - 1.0))[:, None, None] * MJTJM)
        dGdM += ((1.0 - 2.0 * theta) * dq * r ** p
                 * ((1.0 - p) / 2.0) * detM ** ((1 - p) / 2)
                 )[:, None, None] * Minv
        dMdx = _dM_dx(cen)                                # (Nc,d,d,c)
        # grad contribution per centroid component c, shared 1/(d+1) per vert
        gmet = np.einsum('mab,mabc->mc', dGdM, dMdx)      # tr(dGdM·∂_cM)
        gmet = (K / (cdim + 1.0))[:, None] * gmet
        for k in range(cdim + 1):
            np.add.at(grad_loc, cells_own[:, k], gmet)

        # velocity = -grad, assembled cross-rank via coord DM (ADD ghost)
        vel_loc = -grad_loc
        if parallel:
            vloc.array[:] = vel_loc.ravel()
            # localToGlobal(ADD_VALUES) accumulates into vglob; it is fetched
            # once (getGlobalVec, before the loop) and reused every outer iter,
            # so it must be zeroed first — otherwise it carries stale pooled
            # values on the first use and the previous iteration's assembled
            # velocity on every subsequent one.
            vglob.zeroEntries()
            coord_dm.localToGlobal(vloc, vglob, addv=True)
            coord_dm.globalToLocal(vglob, vloc)
            vel = np.asarray(vloc.array).reshape(-1, cdim).copy()
        else:
            vel = vel_loc

        # P_i balancing at vertices (pointwise, complete everywhere)
        Mv = _eval_M(coords); detMv = np.linalg.det(Mv)
        Pi = detMv ** ((p - 1.0) / 2.0)
        v = (Pi / tau)[:, None] * vel

        # nonlinear-CG (Polak-Ribière+): replace the steepest-descent direction
        # v with the conjugate direction d = v + beta_cg * d_prev (β from gradient
        # history — parameter-free; auto-restarts when β<0).
        if _accel == "cg":
            _fo_cg = free & is_owned_v
            _den = _gdot(_prev_v, _prev_v, _fo_cg)
            _beta_cg = (max(0.0, _gdot(v, v - _prev_v, _fo_cg) / _den)
                        if _den > 0.0 else 0.0)
            _prev_v = v.copy()
            v = v + _beta_cg * _prev_dir
            _prev_dir = v.copy()

        # Per-node step cap from the min incident edge over rank-local
        # cells. NOTE (parallel): a partition-boundary owned vertex may not
        # see every incident edge from rank-local cells, so its cap differs
        # slightly from serial → an ~0.006%-level serial/parallel drift in
        # the final mesh. The velocity ASSEMBLY itself is bit-identical
        # serial vs parallel (localToGlobal(ADD_VALUES) is exact); only this
        # cap is rank-dependent. The drift is below the move's own
        # non-determinism, so we accept it rather than force a ghost-complete
        # MIN reduction (PETSc localToGlobal has no portable MIN/MAX mode
        # here — MAX_VALUES errors on this DM). Left as a known small
        # non-reproducibility; revisit only if a bit-exact mesh is required.
        h = _min_incident_edge_nd(cells_all, coords)
        mag = np.linalg.norm(v, axis=1)
        cap = step_frac * h
        sc = np.ones_like(mag)
        m = (mag > cap) & (mag > 0.0)
        sc[m] = cap[m] / mag[m]
        step = v * sc[:, None]
        # Robustness guard (esp. parallel): a degenerate / near-inverted cell can
        # produce a non-finite gradient (inf v -> mag=inf -> sc=cap/inf=0 ->
        # step = inf*0 = NaN here). A NaN/inf displacement then makes a NaN trial
        # whose centroid query blows up `_energy`/`_eval_M` (kd-tree) and, on a
        # subset of ranks, deadlocks the whole job. Zero any non-finite step so
        # that node simply does not move this iteration while the rest of the
        # mesh still adapts.
        step = np.where(np.isfinite(step), step, 0.0)

        if _accel in ("heavyball", "hb-restart") and _mmpde_beta > 0.0:
            _disp = _prev_disp
            if _accel == "hb-restart":
                # gradient restart: drop momentum when it opposes the descent
                # step (overlap < 0) so it never drives uphill.
                if _gdot(step, _prev_disp, free & is_owned_v) < 0.0:
                    _disp = np.zeros_like(_prev_disp)
            step = step + _mmpde_beta * _disp
            step = np.where(np.isfinite(step), step, 0.0)

        # only owned interior vertices move; ghosts halo-synced each trial
        free_owned = free & is_owned_v

        # energy line-search backtrack (monotone, fold-free; collective)
        scale = 1.0
        accepted = coords
        Inew = prevI
        for _bt in range(24):
            trial = coords.copy()
            trial[free_owned] += scale * step[free_owned]
            trial = _project(trial)
            trial = _halo_sync(trial)
            # reject any non-finite trial (defense-in-depth: projection/halo
            # could still introduce inf/NaN) so `_energy` never queries NaN.
            if (np.all(np.isfinite(trial))
                    and _min_area_frac(trial) > float(area_floor_frac)):
                Itr = _energy(trial)
                if Itr < prevI:
                    accepted = trial; Inew = Itr; break
            scale *= 0.5
        else:
            accepted = coords; Inew = prevI; scale = 0.0
        dmax = _global_max(np.linalg.norm(
            (accepted - coords)[is_owned_v], axis=1).max(initial=0.0))
        _prev_disp = accepted - coords   # accepted move, for next-iter momentum
        coords = accepted
        mesh._deform_mesh(coords)
        if verbose:
            uw.pprint(
                f"  mmpde outer {outer+1}/{n_outer}: I={Inew:.6e} "
                f"dI={Inew-prevI:+.2e} scale={scale:.3f} max|Δx|={dmax:.2e}")
        # Converged when (a) the line-search could make no downhill move
        # (scale collapsed to 0 — at a local minimum / stuck), or (b) the
        # accepted node move is negligible relative to the cell size
        # (dmax < tol·h0). tol defaults to 1e-3 (move < 0.1% of a cell).
        # The legacy absolute `outer_tol` is retained as an additional, even
        # tighter floor for callers that set it.
        prevI = Inew
        # Stagnation (residual stol) exit: PETSc-`stol`-style "give up when the
        # meshing functional stops dropping well below the last steps". The
        # node-step `dmax` is capped and never shrinks on this descent mover, so
        # a step-test can't fire; instead test the *energy* (the residual) drop
        # over the last `stol_k` accepted iterations -- a WINDOW (not single
        # step), which is immune to the line-search per-iteration noise and to
        # the occasional big drop after a scale reduction. Opt-in: stol=None/0
        # preserves the previous behaviour bit-for-bit.
        if stol is not None and stol > 0.0:
            _Iwin.append(Inew)
            if len(_Iwin) > stol_k:
                _Iref = _Iwin[-1 - stol_k]
                _rel = (_Iref - Inew) / max(abs(_Iref), 1.0e-30)
                if _rel < stol:
                    if verbose:
                        uw.pprint(
                            f"  mmpde stol-exit at outer {outer+1}/{n_outer}: "
                            f"rel energy drop over last {stol_k} = {_rel:.2e} "
                            f"< stol={stol:.1e}")
                    break
        if scale == 0.0 or dmax < tol * h0_scale or dmax < outer_tol:
            break

    coord_dm.restoreLocalVec(vloc)
    coord_dm.restoreGlobalVec(vglob)
