"""The anisotropic tensor-metric mover — the one genuine Winslow
(M-weighted Laplace) coordinate-map smooth in the package. See the
package docstring for the module map.
"""

import numpy as np

import underworld3 as uw

from .graph import (_tri_cells, _signed_areas, _mean_edge_length,
                    _backtracked_move,
                    _reweight_displacement_radial_tangential,
                    _global_sum, _global_min, _global_max, _global_mean)
from .monge_ampere import _solver_wiring, _warm_start_krylov


# Cached anisotropic-mover state keyed by (mesh-id, pinned-labels,
# topology, solver, φ-order, slip): the ∇ρ projector, the
# eigen-clamped metric-tensor field D, and the cdim displacement
# Poisson solvers (all sharing the tensor operator _c = D). Rebuilt
# on a topology change (a new key).
_ANISO_CACHE: dict = {}


# Per-(mesh,config) running state for the equidistribution
# normaliser's temporal damping: the EMA of ln G carried across
# adaptation events (same key as _ANISO_CACHE). Empty ⇒ first
# event seeds it. Only touched in the resolution_ratio>1 regime.
_GEMA_STATE: dict = {}


def _winslow_anisotropic(mesh, metric, pinned_labels, verbose,
                         n_outer=12, relax=0.2, beta=200.0,
                         resolution_ratio=1.0,
                         geom_mean_smoothing=0.25,
                         aniso_to_base=False,
                         aniso_cap=2.0, coarsen_cap=1.0,
                         boundary_slip=False,
                         linear_solver="direct", phi_degree=2,
                         move_anisotropy=None, metric_role="M",
                         outer_tol=1.0e-4,
                         rest_size_cap_max=None,
                         rest_size_cap_min=None,
                         rest_spring_K=1.0,
                         h0_override=None,
                         rest_coords_override=None,
                         metric_refresh_per_iter=False):
    r"""Anisotropic metric-tensor mesh redistribution — approach (3).

    The settled scalar equidistribution paths (``_spring_equilibrium_mover``,
    ``_monge_ampere_mover``) cannot do coherent *anisotropic* bulk
    transport on a fixed topology — a scalar potential is isotropic,
    so an annulus radial feature over-collapses one pinned-boundary
    sliver layer while the tangential edges sit frozen (see the
    project memory + the design doc's angular-OT section). This is
    the **tensor** mover: it solves the M-weighted Laplace smooth of
    the coordinate map with an *anisotropic* metric tensor, so cells
    are reshaped (short across the feature, long along it) and the
    slivers / wasted isotropic resolution are removed.

    Construction (verified — ``scripts/ma_metric_tensor_viz.py``):
    from a scalar density ``ρ`` (typically Lagrangian
    ``f(r0.sym)``), the *projected* gradient ``∇ρ`` (a first
    derivative only — UW3-clean) builds, per node,

    .. math::

        M \;=\; \tfrac1{h_0^2}\!\left[\,I
              + \beta\,\hat g\hat g^{\mathsf T}
                (|\nabla\rho|/\nabla\rho_{\mathrm{ref}})^2\right],

    eigen-clamped so the spacing ratio ``≤ aniso_cap`` (``≤8:1`` by
    default). The eigenframe **auto-aligns to the feature** from the
    Cartesian ``∇ρ`` alone — no ``(r,θ)`` frame is specified.

    Mover: solve, per physical coordinate component ``c``, the
    displacement form of the M-weighted Laplace (Winslow) map

    .. math::

        \nabla\!\cdot(D\,\nabla u_c) \;=\;
            -\,\nabla\!\cdot(D\,e_c)
          \;=\; -\textstyle\sum_j \partial_j D_{jc},
        \qquad u_c = 0 \text{ on the pinned boundary},

    with ``D = M`` (the eigen-clamped metric). Then
    ``ψ_c = x_c + u_c`` is exactly the M-harmonic coordinate map
    ``∇·(D∇ψ_c)=0``, ``ψ=x`` on the boundary; the direct Winslow
    smoother clusters nodes where ``D`` is large (fine spacing), so
    ``D = M`` grades the mesh toward the metric. The two components
    share the **same** tensor operator (``_c = D``, the
    ``_CofDiff``-style ``DiffusionModel`` pattern) and the
    factor-once-reuse direct solver. **Linear** — one solve per
    component per outer step, no Picard (much cheaper than the BFO
    ``_monge_ampere_mover``). Homogeneous Dirichlet ``u=0`` on the
    pinned boundary makes the per-component operator non-singular —
    no ``constant_nullspace``, side-stepping the GAMG-pure-Neumann
    fragility entirely (``boundary_slip=True`` falls back to the
    pure-Neumann + ring-projection treatment of
    ``_monge_ampere_mover``). ``n_outer`` composes the map (re-project
    ``∇ρ`` / rebuild ``D`` on the moved mesh — the standard MMPDE
    outer iteration). Reuses ``_monge_ampere_mover``'s coherent global
    signed-area backtrack, ``boundary_slip`` and ``move_anisotropy``.

    .. warning::

       (3) improves cell **alignment / quality** and removes the
       slivers + wasted isotropic resolution; it does **not** beat
       the fixed node-count grading cap (≈1.5–1.8× for an 8–20×
       density target — that needs ``mesh.adapt``, a topology
       change). For a *separable* feature the explicit 1-D OT
       (``scripts/ma_analytic_check.py`` /
       ``ma_angular_ot_target.py``) is exact and strictly cheaper;
       (3) earns its keep on the general **non-separable** case.
       Validate with anisotropy-aware diagnostics
       (radial/tangential edge split + minA/meanA, *not* the
       anisotropy-blind d/n).

    Parameters mirror ``_monge_ampere_mover`` where shared.

    The **decoupled direct** Winslow form (each physical coordinate
    M-harmonic, independently) has no Rado–Kneser–Choquet
    non-folding guarantee, so its stable regime is bounded by the
    metric anisotropy/contrast. Empirically (interior radial
    feature, the validation arc) there is a clean Pareto frontier:

    * ``aniso_cap=2``, ``relax≈0.1–0.2`` → minA/meanA ≈ 0.5 (a
      near-pristine, valid, feature-aligned mesh — cleaner than the
      isotropic MA ≈0.18 / spring ≈0.25 which sliver), modest 2:1
      cell alignment. **The robust default.**
    * higher ``aniso_cap`` is only stable with a *gentler* ``relax``
      + more ``n_outer`` (cap 4 needs relax ≈0.05, n_outer ≳25 →
      minA ≈0.35, sharper alignment). ``aniso_cap ≳ 6`` folds the
      decoupled map regardless — it would need the coupled / inverse
      Winslow (the heavy MMPDE, out of this prototype's scope).

    **Single-knob model (`resolution_ratio` R).** The gradient-only
    metric ``M ⪰ base·I`` is *refine-only* (keeps only ``∇ρ``,
    discards ρ's magnitude ⇒ flat cells pinned at ``h0``, cannot
    release nodes, the steepest feature scavenges the budget). The
    fix makes the isotropic density a genuinely **equidistributed**
    field ``s = base·ρ/G`` (``G`` = geometric mean of ρ on the
    near-uniform undeformed D mesh ⇒ ``⟨ln s⟩=ln base``, node budget
    centred). Refine (``s>base``) and coarsen (``s<base``) are then
    **complementary by the conservation law itself** — there is no
    coarsening parameter. ``R`` only sets the safety eigen-clamp
    ``[base/R², base·R²]`` (cells ∈ ``[h0/R, h0·R]``); M-harmonic
    scale-invariance makes the normalisation constant irrelevant, so
    the geometric-mean centring merely places the band symmetrically
    around the bulk. ``R=1`` ⇒ exact refine-only no-op (every prior
    result bit-preserved); ``R≈2`` is the validated production
    point. The legacy two-knob ``aniso_cap``/``coarsen_cap`` clamp
    is retained only as a bit-for-bit expert override when ``R≤1``.
    ``G`` is recomputed from the *instantaneous* field every
    adaptation event; in a violent transient that sloshes the whole
    ``ρ/G`` distribution across the fixed clamp band → mass
    clamp-saturation → a visible mesh "wobble".
    ``geom_mean_smoothing`` (``a``, default 0.25) low-passes ``ln
    G`` across events (``lnG←a·lnG_now+(1−a)·lnG_prev``; ``a=1`` ⇒
    off/instantaneous, ``a≈0.25`` ⇒ strongly damped) so the band
    stays centred — smoothing **only the global intensity scalar**
    (the spatial ρ pattern still tracks the field every event, so
    the user-facing API stays single-knob; one scalar is carried in
    ``_GEMA_STATE`` across events). ``relax`` (default 0.2)
    under-relaxes the per-step displacement;
    ``n_outer`` (default 12) composes the damped steps toward the
    fixed-D M-harmonic map. ``beta`` (default 200) sets how fast the
    metric saturates the ``aniso_cap`` eigen-clamp (the clamp, not
    ``beta``, is the binding anisotropy lever). ``metric_role``
    (``"M"`` default, or ``"Minv"``) is an experimental knob — the
    overall scale of ``D`` is irrelevant to ``∇·(D∇u)=src`` (both
    sides scale together); only the anisotropy + spatial variation
    matter.
    """
    import sympy

    pinned_labels = tuple(pinned_labels)
    cdim = mesh.cdim
    if cdim != 2:
        raise NotImplementedError(
            "_winslow_anisotropic: 2D triangle meshes only "
            "(the eigen-clamp + Annulus diagnostics are 2D)")
    if linear_solver not in ("direct", "gamg"):
        raise ValueError(
            f"linear_solver must be 'direct' or 'gamg', "
            f"got {linear_solver!r}")
    if metric_role not in ("M", "Minv"):
        raise ValueError(
            f"metric_role must be 'M' or 'Minv', got {metric_role!r}")

    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    cStart, cEnd = dm.getHeightStratum(0)
    cone_size = dm.getConeSize(cStart) if cEnd > cStart else 0
    phi_degree = int(phi_degree)
    aux_degree = max(1, phi_degree - 1)
    key = (id(mesh), pinned_labels, pEnd - pStart, cEnd - cStart,
           cone_size, linear_solver, phi_degree, bool(boundary_slip))

    cache = _ANISO_CACHE.get(key)
    if cache is None:
        _wire = _solver_wiring(linear_solver)

        X = mesh.CoordinateSystem.X
        # Projected ∇ρ — first derivative only (UW3-clean), the
        # same construction verified in ma_metric_tensor_viz. ρ may
        # be Lagrangian f(r0.sym): metric.diff(X) then differentiates
        # through the frozen r0 field (FE ∂r0/∂x), so ∇ρ is
        # re-evaluated on the moved mesh each outer step (MMPDE).
        grho = uw.discretisation.MeshVariable(
            f"aniso_grho_{id(mesh)}", mesh,
            vtype=uw.VarType.VECTOR, degree=aux_degree,
            continuous=True)
        gproj = uw.systems.Vector_Projection(mesh, grho)
        gproj.smoothing = 0.0
        gproj.uw_function = sympy.Matrix(
            [metric.diff(X[i]) for i in range(cdim)]).T
        _wire(gproj, elliptic=False)

        # Eigen-clamped metric tensor field D (filled numerically
        # per outer step). Init to the identity so an unsolved D is
        # a harmless isotropic operator.
        Df = uw.discretisation.MeshVariable(
            f"aniso_D_{id(mesh)}", mesh,
            vtype=uw.VarType.TENSOR, degree=aux_degree,
            continuous=True)
        Df.array[:, 0, 0] = 1.0
        Df.array[:, 1, 1] = 1.0
        Df.array[:, 0, 1] = 0.0
        Df.array[:, 1, 0] = 0.0
        Dsym = Df.sym                      # 2×2 sympy Matrix (stable)

        class _TensorDiff(uw.constitutive_models.DiffusionModel):
            def _build_c_tensor(self):
                self._c = Dsym

        # boundary_slip ⇒ pure-Neumann per component (constant
        # nullspace, ring-projected in the move — exactly the
        # _monge_ampere_mover slip treatment). Default (pinned) ⇒
        # homogeneous Dirichlet u=0 → non-singular, no nullspace.
        singular = bool(boundary_slip)
        usolvers, ufields = [], []
        for c in range(cdim):
            uc = uw.discretisation.MeshVariable(
                f"aniso_u{c}_{id(mesh)}", mesh,
                vtype=uw.VarType.SCALAR, degree=phi_degree,
                continuous=True)
            ps = uw.systems.Poisson(mesh, uc)
            ps.constitutive_model = _TensorDiff
            # f_c = div(column c of D) = Σ_j ∂D_{jc}/∂x_j. UW3
            # SNES_Poisson is F0=-f ⇒ strong form ∇·(D∇u)=-ps.f;
            # we want ∇·(D∇u_c) = -div_c ⇒ ps.f = +div_c. (First
            # derivative of the projected D field — UW3-legal.)
            src = sympy.Integer(0)
            for j in range(cdim):
                src = src + Dsym[j, c].diff(X[j])
            ps.f = sympy.Matrix([[src]])
            if singular:
                ps.constant_nullspace = True
            else:
                for lbl in pinned_labels:
                    try:
                        ps.add_dirichlet_bc(0.0, lbl)
                    except Exception:
                        pass
            _wire(ps, singular=singular, elliptic=True)
            usolvers.append(ps)
            ufields.append(uc)

        _ANISO_CACHE[key] = (grho, gproj, Df, usolvers, ufields)
    else:
        grho, gproj, Df, usolvers, ufields = cache

    zero_init_guess = not _warm_start_krylov(linear_solver)

    # ---- build the eigen-clamped metric tensor field D ONCE ------
    # on the *undeformed* mesh (the design metric), then hold it
    # fixed and Lagrangian (the field rides material points through
    # _deform_mesh, exactly as _spring_equilibrium_mover computes its
    # rest-lengths / A0 once). Re-projecting ∇ρ on the progressively
    # distorted mesh inside the outer loop is a positive feedback —
    # D blows up on squashed cells → catastrophic over-collapse
    # (verified failure mode). With D fixed the outer loop is a
    # *stable damped fixed-point iteration* of one linear operator
    # toward the M-harmonic map; no feedback.
    dm = mesh.dm
    # `old0` is the SPRING REST reference — vertices get pulled
    # toward these positions when a cell exceeds the size caps.
    # If the caller passes `rest_coords_override`, use that
    # (typically the truly-undeformed mesh coords captured at
    # the first adapt). Falling back to the entry-state of THIS
    # call makes the spring "preserve" each successive refined
    # state instead of pulling back to undeformed — the third
    # leg of the compounding-refinement bug (2026-05-22).
    if rest_coords_override is not None:
        old0 = np.asarray(rest_coords_override).copy()
    else:
        old0 = np.asarray(mesh.X.coords).copy()
    # h0 = undeformed mean edge length; `h0_override` lets the caller
    # supply the value cached at the FIRST adapt (see the
    # compounding-refinement note on _mean_edge_length).
    if h0_override is not None:
        h0 = float(h0_override)
    else:
        h0 = _mean_edge_length(dm, old0)
    # CRITICAL no-op guard: uniform ρ ⇒ ∇ρ ≡ 0, but the L2
    # projection of the zero function leaves ~1e-18 round-off.
    # Normalising by that noisy max would make (|∇ρ|/gref)² ~ O(1)
    # from pure round-off → a fabricated huge anisotropy and a
    # spurious move. Any *real* feature gradient is O(AMP/WIDTH)
    # ~ O(1–100); g_eps=1e-9 is ~9 orders above projection noise
    # and ~10 below the weakest meaningful feature, so AMP=0 is an
    # exact isotropic no-op while AMP>0 is bit-identical to the
    # verified ma_metric_tensor_viz construction.
    g_eps = 1.0e-9
    base = 1.0 / h0 ** 2
    # Metric-build state shared with the closure below: bound (via
    # ``nonlocal``) by ``_build_M_tensor()``, whose pre-loop call is the
    # single build site (the per-iteration refresh path rebinds the same
    # names when ``metric_refresh_per_iter`` is on).
    Dcoords = gvec = gn = gmax = gref = None

    # --- isotropic density: which redistribution model ------------
    # Three regimes, in precedence order:
    #
    #  (1) ``resolution_ratio > 1`` → SINGLE-KNOB EQUIDISTRIBUTION
    #      (the primary, documented API). The isotropic density is
    #      ``s = base·ρ/G`` with ``G`` the geometric mean of ρ on
    #      the (near-uniform, *undeformed*) D mesh, so
    #      ``⟨ln s⟩ = ln base``: the node budget is centred and
    #      refine ⇄ coarsen are **complementary by the conservation
    #      law itself** — there is no coarsening parameter. The
    #      eigen-clamp ``[base/R², base·R²]`` (cells ∈
    #      ``[h0/R, h0·R]``) is a pure safety rail set by the one
    #      knob ``R``. M-harmonic is scale-invariant, so the
    #      normalisation *constant* is irrelevant to the realised
    #      mesh — only ρ's spatial *ratio* and the clamp matter;
    #      the geometric-mean centring just places the band
    #      symmetrically so the clamp bites tails, not the bulk.
    #
    #  (2) ``coarsen_cap > 1`` (legacy expert override, not the
    #      documented API) → the earlier ad-hoc
    #      ``s = base·cc^(q-1)`` law. Preserved **bit-for-bit** so
    #      every historical ``a16c*`` result still reproduces.
    #
    #  (3) otherwise → refine-only metric (``s ≡ base``),
    #      **bit-identical** to the validated historical default.
    #      ``resolution_ratio = 1`` (the default) lands here ⇒ an
    #      exact no-op vs. all prior results.
    def _build_M_tensor():
        """Compute the metric tensor field Df from the current
        metric and mesh state. Mutates Dout-equivalent into Df.
        Called once before the iteration loop, and (when
        metric_refresh_per_iter=True) also at the start of each
        outer iteration to re-query the metric against the
        deformed mesh."""
        nonlocal Dcoords, gvec, gn, gmax, gref
        Dcoords = np.asarray(Df.coords)  # picks up deformed mesh
        gproj.solve()
        gvec = np.asarray(
            uw.function.evaluate(grho.sym, Dcoords)
        ).reshape(-1, cdim)
        gn = np.linalg.norm(gvec, axis=1)
        # Local max first, THEN the (collective) reduction — every rank
        # must participate even if it owns no D-mesh points.
        gmax = float(gn.max()) if gn.size else 0.0
        gmax = _global_max(gmax)
        gref = gmax if gmax > g_eps else 1.0
        # Density branches (same as legacy code path)
        if resolution_ratio > 1.0:
            R_ = float(resolution_ratio)
            rho_v_ = np.asarray(
                uw.function.evaluate(metric, Dcoords)
            ).reshape(-1)
            s_log_ = np.log(np.clip(rho_v_, 1.0e-12, None))
            if uw.mpi.size > 1:
                tot = _global_sum(s_log_.sum())
                cnt = _global_sum(s_log_.size)
                ln_g_ = tot / max(cnt, 1)
            else:
                ln_g_ = float(s_log_.mean())
            a_ = float(geom_mean_smoothing)
            if 0.0 < a_ < 1.0:
                prev = _GEMA_STATE.get(key)
                if prev is not None:
                    ln_g_ = a_ * ln_g_ + (1.0 - a_) * prev
                _GEMA_STATE[key] = ln_g_
            iso_ = base * np.exp(s_log_ - ln_g_)
            lam_lo_ = base / R_ ** 2
            lam_hi_ = base * R_ ** 2
            aniso_keyed_ = (np.full(Dcoords.shape[0], base)
                            if aniso_to_base else iso_)
        elif coarsen_cap > 1.0:
            rho_v_ = np.asarray(
                uw.function.evaluate(metric, Dcoords)
            ).reshape(-1)
            r_lo_ = float(np.percentile(rho_v_, 10.0))
            r_hi_ = float(np.percentile(rho_v_, 90.0))
            r_lo_ = _global_min(r_lo_)
            r_hi_ = _global_max(r_hi_)
            q_ = np.clip(
                (rho_v_ - r_lo_) / max(r_hi_ - r_lo_, 1e-30),
                0.0, 1.0)
            iso_ = base * float(coarsen_cap) ** (q_ - 1.0)
            lam_lo_ = base / float(coarsen_cap)
            lam_hi_ = 1.0 / (h0 / np.sqrt(aniso_cap)) ** 2
            aniso_keyed_ = np.full(Dcoords.shape[0], base)
        else:
            iso_ = np.full(Dcoords.shape[0], base)
            lam_lo_ = base
            lam_hi_ = 1.0 / (h0 / np.sqrt(aniso_cap)) ** 2
            aniso_keyed_ = np.full(Dcoords.shape[0], base)
        # Assemble M tensor and write to Df
        Dout_ = np.empty((Dcoords.shape[0], 2, 2))
        eye2_ = np.eye(2)
        for ii in range(Dcoords.shape[0]):
            g_ = gvec[ii]
            gni_ = gn[ii]
            bi_ = iso_[ii]
            ai_ = aniso_keyed_[ii]
            if gni_ > g_eps and gmax > g_eps:
                gh_ = g_ / gni_
                M_ = bi_ * eye2_ + ai_ * beta * (gni_ / gref) ** 2 \
                     * np.outer(gh_, gh_)
            else:
                M_ = bi_ * eye2_
            w_, V_ = np.linalg.eigh(M_)
            w_ = np.clip(w_, lam_lo_, lam_hi_)
            if metric_role == "Minv":
                w_ = 1.0 / w_
            Dout_[ii] = (V_ * w_) @ V_.T
        Df.array[:, 0, 0] = Dout_[:, 0, 0]
        Df.array[:, 0, 1] = Dout_[:, 0, 1]
        Df.array[:, 1, 0] = Dout_[:, 1, 0]
        Df.array[:, 1, 1] = Dout_[:, 1, 1]

    # Build D once here, on the undeformed mesh. (This call replaced a
    # ~100-line inline duplicate of the closure body — READ-02.)
    _build_M_tensor()

    # Pre-compute the undeformed-mesh median cell area, used by the
    # backtrack's sliver guard. Captured ONCE before the iteration
    # loop so the floor doesn't shrink as cells refine — the same
    # absolute floor is enforced throughout.
    _tris_for_a0 = _tri_cells(mesh.dm)
    if _tris_for_a0 is not None and _tris_for_a0.size:
        _a0_undeformed_med = float(np.median(np.abs(
            _signed_areas(old0, _tris_for_a0))))
    else:
        _a0_undeformed_med = 0.0

    for outer in range(n_outer):
        dm = mesh.dm
        pStart, pEnd = dm.getDepthStratum(0)
        n_verts = pEnd - pStart
        tris = _tri_cells(dm)
        old_coords = np.asarray(mesh.X.coords).copy()
        _cdim = mesh.cdim

        # If requested, re-query the metric at the deformed
        # mesh state and rebuild M tensor. Default off
        # preserves the legacy behaviour (M frozen at first
        # iteration). Used to isolate whether Eulerian
        # re-querying of the metric changes the outcome.
        if metric_refresh_per_iter and outer > 0:
            _build_M_tensor()

        # Boundary tangential slip via the mesh-owned contract
        # (boundary-slip-strategy.md): slip vertices slide tangentially and
        # snap back onto their bounding surface (radial ring / plane / facet);
        # non-slip, junction, and degenerate-normal vertices pin. Replaces the
        # inline per-ring COM radial snap (one node/ring anchored the rotation
        # gauge; the signed-area backtrack below still guards against tangle).
        is_pinned, _project = mesh.boundary_slip(
            boundary_slip, reference_coords=old_coords,
            boundary_labels=pinned_labels)

        # D is fixed & Lagrangian (built once, above) — no
        # re-projection feedback. The outer loop is a damped
        # fixed-point iteration toward the fixed M-harmonic map.

        # --- solve the cdim displacement components ----------------
        disp = np.zeros_like(old_coords)
        for c in range(cdim):
            usolvers[c].solve(zero_init_guess=zero_init_guess)
            disp[:, c] = np.asarray(
                uw.function.evaluate(ufields[c].sym, old_coords)
            ).reshape(-1)

        # Directional move-weighting (opt-in; default None ⇒ unchanged).
        if move_anisotropy is not None and cdim == 2:
            disp = _reweight_displacement_radial_tangential(
                disp, old_coords, move_anisotropy)

        # --- per-cell Lagrangian rest-size spring -----------------
        # When `rest_size_cap_max` / `rest_size_cap_min` are set,
        # add a restoring force to each vertex that pulls it
        # toward its rest position (`old0`, captured before the
        # mover started) whenever an incident cell's edge would
        # overshoot the cap under the proposed move.
        #
        # We use **max-edge** for the coarsening cap (a cell
        # grew in *any* direction beyond `h0·coarsening`) and
        # **min-edge** for the refinement cap (a cell shrunk
        # in *any* direction below `h0/refinement`). Both
        # measures are sliver-aware — they catch anisotropic
        # cells that mean-edge wouldn't flag.
        #
        # Motivation: the metric-mover is a local graph-Laplacian
        # — nodes cannot transport across high-gradient ridges,
        # so cells *adjacent* to a refinement zone absorb most
        # of the freed area while cells topologically isolated
        # from the refinement stay near rest size. Without a
        # spring, the adjacent cells over-coarsen by ~2× the cap
        # and the BL cells over-refine to thin slivers (aspect
        # ratios > 10). The spring restores both by literally
        # pulling nodes back along the original positions,
        # weighted by how much the local cell exceeds the cap.
        if (rest_size_cap_max is not None
                or rest_size_cap_min is not None):
            proposed = old_coords + float(relax) * disp
            p = proposed[tris]
            e0 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
            e1 = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
            e2 = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
            # Sliver-aware per-cell extremes:
            max_h = np.maximum(np.maximum(e0, e1), e2)
            min_h = np.minimum(np.minimum(e0, e1), e2)
            # Per-cell fractional excess vs cap. Both ≥ 0.
            #   over  = max(any edge)/cap_max - 1       (coarsening
            #     fault: at least one edge too long)
            #   under = cap_min / min(any edge) - 1     (refinement
            #     fault: at least one edge too short, i.e. sliver)
            if rest_size_cap_max is not None:
                over = np.maximum(
                    max_h / float(rest_size_cap_max) - 1.0, 0.0)
            else:
                over = np.zeros_like(max_h)
            if rest_size_cap_min is not None:
                under = np.maximum(
                    float(rest_size_cap_min)
                    / np.maximum(min_h, 1.0e-30) - 1.0, 0.0)
            else:
                under = np.zeros_like(min_h)
            # Per-vertex restoring weight ← Σ over incident cells,
            # CAPPED AT 1. Without the cap, a vertex incident on
            # several violating cells accumulates restore_w > 1
            # and the spring overshoots its rest position
            # (`new = old + restore_w · (rest - old)` lands past
            # `rest`), pulling two vertices together and creating
            # degenerate (near-zero-area) triangles. Capping at 1
            # makes the worst-case per-iteration motion "exactly
            # back to rest", never further.
            restore_w = np.zeros(old_coords.shape[0])
            cell_w = float(rest_spring_K) * (over + under)
            np.add.at(restore_w, tris[:, 0], cell_w)
            np.add.at(restore_w, tris[:, 1], cell_w)
            np.add.at(restore_w, tris[:, 2], cell_w)
            np.minimum(restore_w, 1.0, out=restore_w)
            # Add the restoring contribution to disp. (Divide by
            # relax so the downstream `step = relax · disp` gives
            # the intended fraction restore_w · (rest - current).)
            spring_disp = restore_w[:, None] * (old0 - old_coords)
            disp = disp + spring_disp / max(float(relax), 1.0e-30)

        # Damped MMPDE step. The *direct* Winslow form (physical
        # coords as M-harmonic functions of themselves) has no
        # Rado–Kneser–Choquet non-folding guarantee — applied as a
        # single elliptic jump it overshoots and the signed-area
        # backtrack thrashes into a degenerate sliver. The standard
        # remedy is to integrate the mesh PDE as a damped gradient
        # flow: under-relax the displacement and compose over
        # n_outer steps (the metric is re-projected each step). This
        # is the exact analogue of _monge_ampere_mover's picard_relax
        # (the BFO path needs ω≈0.4 or its Hessian grows unbounded).
        step = float(relax) * disp

        # --- coherent global signed-area backtrack + slip + move --
        # Positive area floor: the flip-only test (`a1min > 0`) misses
        # near-degenerate cells with three near-collinear vertices, so
        # require min area > 1% of the **undeformed-mesh** median cell
        # area (`_a0_undeformed_med`, captured before the iteration
        # loop, so the same absolute floor is enforced throughout). A
        # refinement of 3 in 2D legitimately shrinks cells by 3²=9× in
        # area, so 1% rejects degenerate slivers (1000× smaller)
        # without rejecting legitimate refinement.
        free = ~is_pinned
        new_coords, scale = _backtracked_move(
            old_coords, step, free, tris, _project,
            area_floor=0.01 * _a0_undeformed_med)

        mesh._deform_mesh(new_coords)

        d = float(np.linalg.norm(
            new_coords - old_coords, axis=1).max())
        if uw.mpi.size > 1:
            d = _global_sum(d ** 2) ** 0.5
        if verbose:
            uw.pprint(
                f"  anisotropic mover outer {outer+1}/{n_outer}: "
                f"h0={h0:.3e}  scale={scale:.3f}  "
                f"max|Δx|={d:.3e}")
        if d < outer_tol:
            break
