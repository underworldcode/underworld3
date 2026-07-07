"""Public dispatch: :func:`smooth_mesh_interior` (with the Phase-1
field-transfer wrap), the bare mover dispatch, and the
:func:`follow_metric` two-knob adapter. See the package docstring for
the module map.
"""

import warnings
from typing import Optional, Sequence

import numpy as np

import underworld3 as uw

from .graph import (_auto_pinned_labels, _pinned_mask,
                    _build_adjacency_matrix, _build_local_to_owned_map,
                    _tri_cells, _signed_areas, _mean_edge_length,
                    _global_sum, _global_min, _global_max)
from .metrics import (ADAPT_STRATEGIES, _UNSET, mesh_metric_mismatch,
                      metric_density_from_gradient)
from .spring import _spring_equilibrium_mover
from .monge_ampere import _monge_ampere_mover, _ot_improvement_step
from .anisotropic import _winslow_anisotropic
from .mmpde import _mmpde_mover


# Cached adjacency keyed by (mesh-id, pinned-label-tuple, topology).
# Rebuilt automatically when the mesh topology changes.
_ADJ_CACHE: dict = {}


# Cache of the **original** (undeformed) state per mesh,
# captured the first time follow_metric is called on that mesh:
#   h0           — mean edge length
#   rest_coords  — vertex positions (the spring's pull-back target)
# Subsequent calls reuse these references instead of measuring the
# (already-refined) current mesh, otherwise the spring's reference
# state shrinks at every adapt and the refinement compounds,
# crashing the CFL-bound dt by 2× per adapt step.
# Keyed by id(mesh).
_FOLLOW_METRIC_H0_CACHE: dict = {}
_FOLLOW_METRIC_REST_CACHE: dict = {}


def smooth_mesh_interior(
    mesh,
    pinned_labels: Optional[Sequence[str]] = None,
    n_iters: int = 5,
    alpha: float = 0.5,
    metric=None,
    method: str = "spring",
    boundary_slip: bool = False,
    method_kwargs: Optional[dict] = None,
    verbose: bool = False,
    skip_threshold=_UNSET,
    strategy: Optional[str] = None,
    slip_surfaces=None,
):
    r"""Smooth a mesh's interior vertices, optionally toward a
    spatially-varying target spacing.

    **Default (``metric=None``)** — graph-Laplacian Jacobi: each
    interior vertex is blended toward the plain mean of its edge
    neighbours,

    .. math::

        x_i^{n+1} = (1 - \alpha)\, x_i^n
                    + \alpha \cdot \frac{1}{|N(i)|}
                    \sum_{j \in N(i)} x_j^n ,

    over ``n_iters`` sweeps. Equalises connectivity → equant cells.

    **With a ``metric``** — an elastic-spring network relaxed to
    equilibrium. Every edge is a linear spring with rest length
    ``∝ ρ_tgt^{-1/d}`` (``ρ_tgt = metric``), scaled so the mean rest
    length equals the current mean edge length (overall scale
    preserved — pure redistribution). Damped Jacobi force iteration
    relaxes interior nodes to force balance, with a coherent global
    signed-area backtrack guaranteeing no cell inverts. The rest
    length is an *absolute* target, so the mesh genuinely grades
    toward spacing ``∝ ρ_tgt^{-1/d}`` (a regime the weighted
    Laplacian / Jacobi cannot reach). ``n_iters`` and ``alpha`` are
    ignored on this path (it has its own internal sweep budget). A
    Lagrangian density (``f(r0.sym)`` peaked at the original outer
    radius) keeps the rest lengths fixed per material point, so the
    *design* boundary-layer grading is restored even after
    free-surface deformation.

    Vertices in any of ``pinned_labels`` are held fixed (preserves
    boundary geometry). The mesh's coordinate vector is updated in
    place via ``mesh._deform_mesh`` once at the end.

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
        The mesh to smooth. Modified in place.
    pinned_labels : sequence of str, optional
        Names of boundary labels whose vertices stay fixed. If
        ``None`` (default), all non-sentinel labels on
        ``mesh.boundaries`` are pinned — i.e. every named boundary
        stays put. Pass an explicit list to release some boundaries.
    n_iters : int, default 5
        Number of Jacobi sweeps. 5-10 is typical for surface-
        deformation cleanup. **Ignored when ``metric`` is given**
        (the spring path has its own internal sweep budget).
    alpha : float, default 0.5
        Under-relaxation in ``(0, 1]`` for the Jacobi path. 1.0 is
        pure Jacobi; smaller is more damped. **Ignored when
        ``metric`` is given.**
    metric : sympy / UW expression, optional
        Target *density* :math:`\rho_{\mathrm{tgt}}` (larger ⇒
        finer cells). Typically ``f(r0.sym)`` for a refinement
        function ``f`` of a Lagrangian state variable ``r0`` (a
        degree-1 scalar MeshVariable set once to the original
        coordinate and never reassigned, so its value rides each
        material point through deformation). Should be strictly
        positive and finite. ``None`` (default) ⇒ the
        graph-Laplacian Jacobi path, unchanged behaviour
        bit-for-bit.
    method : {"spring", "ma", "anisotropic", "mmpde"}, default "spring"
        Metric-grading solver (ignored when ``metric is None``).
        ``"mmpde"`` is the recommended production mover for adaptive
        meshing; ``"ot"`` is accepted but deprecated (incomplete —
        prefer ``"mmpde"`` with a scalar metric):

        * ``"spring"`` — *volumetric* elastic-spring equilibrium:
          equal edge springs (shape regulariser, equant cells, no
          slivers) + a per-cell area constraint
          ``A0 ∝ 1/ρ_tgt`` (the size grading), minimised by
          preconditioned nonlinear CG. **Fast** (~0.3 s on a
          res-16 Annulus), robust, scales with the metric
          amplitude; slightly anisotropic at sharp interior
          features.
        * ``"ma"`` — Benamou–Froese–Oberman convex-branch
          **Monge–Ampère** equidistribution. Highest-fidelity
          *isotropic* refinement and robust to the boundary
          treatment, but ~60× costlier than the spring.
        * ``"anisotropic"`` — **tensor** metric mover: an
          M-weighted Laplace (Winslow) smooth of the coordinate
          map with an eigen-clamped, gradient-derived *anisotropic*
          metric tensor. Reshapes cells (short across a feature,
          long along it) and removes the slivers / wasted isotropic
          resolution the scalar paths leave near a boundary-peaked
          feature. Linear (one solve/component/step — cheaper than
          ``"ma"``). It improves cell **alignment / quality**, not
          the grading magnitude (see the cap note below); for a
          *separable* feature the explicit 1-D OT is exact and
          cheaper — ``"anisotropic"`` earns its keep on the general
          non-separable case.
        * ``"mmpde"`` — variational moving-mesh (Huang–Kamenski
          MMPDE) with a full tensor (or scalar) metric; the
          recommended production mover for adaptive meshing.
          **Currently 2D-only** (triangle meshes) — a 3D mesh
          raises ``NotImplementedError``.
        * ``"ot"`` (deprecated) — one linear OT-improvement step,
          composable; boundary slip is gated to radial geometries.
          Kept for the internal ``mesh.OT_adapt`` reset path; new
          code should use ``"mmpde"``.

        With a fixed node count neither can exceed ≈1.3–1.8×
        deep/near grading (the optimal-transport ≈10× needs *more
        nodes* — a topology change, not this smoother). See
        ``docs/developer/subsystems/mesh-metric-redistribution.md``.
    boundary_slip : bool, default False
        Let boundary nodes slide tangentially along their boundary
        (snapped back to the boundary each step — they cannot leave
        it; serial circular/spherical boundaries only). Strongly
        helps the spring (+~10 % grading, faster); near-no-op for
        ``ma`` (its natural Neumann BC already handles the
        boundary). Off by default — for a free surface the boundary
        is the moving surface, so sliding interacts with the
        free-surface coupling; enable per use-context.
    method_kwargs : dict, optional
        Extra tuning forwarded to the chosen metric solver (ignored
        when ``metric is None``). Keeps the shared signature clean
        while exposing the per-method knobs. For
        ``method="anisotropic"`` there is **one primary knob**:

        * ``resolution_ratio`` (``R``, default **1.0 = exact
          no-op**) — *the* tuneable. Cells may refine to ``h0/R``
          and coarsen to ``h0·R``; the refine ⇄ coarsen split is
          **not a parameter** — the isotropic density is
          equidistribution-normalised (``s = base·ρ/G``, ``G`` the
          geometric mean of ρ), so flat regions release exactly the
          budget the fronts consume, *complementary by the
          conservation law itself*. The eigen-clamp
          ``[h0/R, h0·R]`` is just a safety rail. ``R=1`` ⇒
          bit-identical to the refine-only historical default (an
          exact no-op vs. every prior result). ``R≈2`` is the
          validated production point (clean mesh through a full
          convection lifecycle, ``minA/meanA``≈0.2, genuine
          plume-reaching de-resolution, settled physics intact).
          One number; complementary coarsening is automatic.
        * ``geom_mean_smoothing`` (``a``, default 0.25) —
          *internal* temporal damping of the equidistribution
          normaliser ``G`` (not a grading knob; only acts when
          ``R>1``). ``G`` is recomputed from the instantaneous
          field every adaptation event; in a violent transient
          that lurches the whole ``ρ/G`` distribution across the
          fixed clamp band → clamp-saturation → the mesh visibly
          "wobbles". An EMA in log space
          (``lnG ← a·lnG_now+(1−a)·lnG_prev``) keeps the band
          centred: ``a=1`` ⇒ no damping (instantaneous, the
          original wobbly behaviour); ``a≈0.25`` ⇒ strong damping
          of the startup over-reaction + steady-state contrast
          pulse. It smooths **only the one global intensity
          scalar** — the spatial ρ(x) pattern still tracks the
          current field every event, so the API stays single-knob
          (``R``); ``a`` carries one internal scalar across events.
        * ``relax`` (0.2) / ``n_outer`` (12) — damped-MMPDE
          under-relaxation + composed steps (early-exit
          ``outer_tol``). ``linear_solver`` (``"direct"`` | MUMPS |
          ``"gamg"``, bit-parity, parallel-scalable). ``beta``
          (200) — anisotropic-bump saturation. ``move_anisotropy``
          — optional radial/tangential move reweight.
        * **Expert overrides (not the documented API; only honoured
          when ``resolution_ratio≤1``):** ``aniso_cap`` (2.0) and
          ``coarsen_cap`` (1.0) are the legacy two-knob clamp
          (``h_min=h0/√aniso_cap``, ``h_max=h0·√coarsen_cap``,
          ad-hoc ``s=base·cc^(q-1)``). Retained **bit-for-bit** so
          historical scripts reproduce; superseded by
          ``resolution_ratio``.

        Example::

            smooth_mesh_interior(
                mesh, metric=rho, method="anisotropic",
                method_kwargs=dict(resolution_ratio=2.0,
                                   relax=0.05, n_outer=25))
    verbose : bool, default False
        Print per-sweep (Jacobi) or periodic (spring/MA) progress.
    skip_threshold : float, optional
        If set, evaluate the *misalignment* between current mesh
        cell density and the metric (via
        :func:`mesh_metric_mismatch`) and **skip the adapt** when
        misalignment is below this threshold. Misalignment is
        ``√(1 − r²)`` where ``r`` is the Pearson correlation of
        ``log(1/A_cell)`` with ``log(ρ_cell)`` — a magnitude-free
        measure of whether cell density is aligned with the
        metric. 0 ⇒ perfectly aligned; 1 ⇒ orthogonal /
        anti-aligned. Ignored when ``metric is None``. Calibration
        from one of the R=1.5 stagnant-lid tests: a uniform mesh
        gives misalignment ≈ 1.00 (r ≈ 0); a freshly-adapted mesh
        gives misalignment ≈ 0.85 (r ≈ 0.52). So ``0.9`` is a
        sensible "skip if reasonably aligned" default for an
        adaptive convection loop; ``0.5`` is strict (only skip
        when very well aligned); ``0`` ⇒ always adapt
        (equivalent to ``None``). Cost: one ``metric`` evaluate
        at cell centroids + a few NumPy reductions.

    Notes
    -----
    **Parallel implementation (Jacobi path)**: the vertex-vertex
    adjacency is assembled as a parallel PETSc AIJ matrix; each rank
    inserts entries for every locally-visible edge using GLOBAL
    vertex indices and ``mat.assemble()`` routes cross-rank
    contributions so that owned-vertex rows are complete after
    assembly. The per-sweep update is a per-component ``A.mult``
    followed by a pointwise divide by the precomputed degree vector.
    Results are bit-identical (to a single ULP) between serial and
    parallel runs at any rank count.

    **Spring path**: serial-exact. Edge forces are accumulated over
    locally-visible edges only, so rank-partition-boundary nodes
    under-count their incident forces in parallel (a future PR can
    assemble the edge forces cross-rank like the Jacobi adjacency
    Mat). The edge list and per-node degree are cached against the
    topology key and rebuilt only on a topology change.

    **Topology preservation**: vertex IDs, DOF mappings, and the
    rank partition are unchanged. Only coordinates move. Anything
    cached against the topology version stays valid; anything
    cached against coords is invalidated by the final
    ``mesh._deform_mesh`` call.

    Examples
    --------
    Pin all named boundaries (the usual case)::

        import underworld3 as uw
        from underworld3.meshing import smooth_mesh_interior

        mesh = uw.meshing.Annulus(...)
        # ... some deformation that leaves bad cells ...
        smooth_mesh_interior(mesh, n_iters=5, alpha=0.5)

    Pin only the outer boundary, allowing the inner to drift::

        smooth_mesh_interior(mesh, pinned_labels=["Upper"])

    Pin nothing (free-floating; rare — boundary will collapse)::

        smooth_mesh_interior(mesh, pinned_labels=[])

    Restore a design grading via a Lagrangian refinement metric::

        r0 = uw.discretisation.MeshVariable(
            "r0", mesh, uw.VarType.SCALAR, degree=1)
        X0 = np.asarray(mesh.X.coords)
        r0.data[:, 0] = np.sqrt((X0 ** 2).sum(axis=1))   # set once
        # ... deformation that crushes near-surface cells ...
        f = 1 + 8 * sympy.exp(-((r0.sym[0] - 1.0) / 0.12) ** 2)
        smooth_mesh_interior(mesh, metric=f)
    """
    # slip_surfaces is the public alias for the deprecated boundary_slip;
    # resolve to a single spec threaded to the bare mover.
    if slip_surfaces is not None:
        if boundary_slip not in (None, False):
            warnings.warn(
                "smooth_mesh_interior: pass either slip_surfaces or the "
                "deprecated boundary_slip, not both; using slip_surfaces.",
                stacklevel=2)
        boundary_slip = slip_surfaces
    if boundary_slip is None:
        boundary_slip = False
    # Pre-create the projected-normal field (mesh.Gamma_P1) ONCE here,
    # before the mover snapshots the DM. Creating this MeshVariable mid-
    # mover restructures the DM and hard-aborts (project_uw3_smoother_footguns).
    if boundary_slip not in (None, False, (), []):
        try:
            _ = mesh.Gamma_P1
        except Exception:
            pass
    # Phase-1 remesh redesign: the adapt op now owns field transfer.
    # Wrap the mover body so every REMAP-policy variable on the mesh is
    # snapshotted, the mover runs, and a single deform-back /
    # global_evaluate / deform-forward pair carries every variable onto
    # the adapted node positions. Re-entrancy guard
    # ``mesh._in_remesh_transfer`` lets composite adapts (OT_adapt) wrap
    # the whole reset+build+smooth dance once at the outer level and
    # have this inner call skip its own wrap.
    if not getattr(mesh, "_in_remesh_transfer", False):
        from underworld3.discretisation.remesh import (
            remesh_with_field_transfer)
        def _do_move():
            _smooth_mesh_interior_bare(
                mesh,
                pinned_labels=pinned_labels,
                n_iters=n_iters,
                alpha=alpha,
                metric=metric,
                method=method,
                boundary_slip=boundary_slip,
                method_kwargs=method_kwargs,
                verbose=verbose,
                skip_threshold=skip_threshold,
                strategy=strategy,
            )
        remesh_with_field_transfer(mesh, _do_move, verbose=verbose)
        return
    # Re-entrant call from inside a composite adapt op: fall through to
    # the bare mover.
    _smooth_mesh_interior_bare(
        mesh,
        pinned_labels=pinned_labels,
        n_iters=n_iters,
        alpha=alpha,
        metric=metric,
        method=method,
        boundary_slip=boundary_slip,
        method_kwargs=method_kwargs,
        verbose=verbose,
        skip_threshold=skip_threshold,
        strategy=strategy,
    )


def _smooth_mesh_interior_bare(
    mesh,
    pinned_labels: Optional[Sequence[str]] = None,
    n_iters: int = 5,
    alpha: float = 0.5,
    metric=None,
    method: str = "spring",
    boundary_slip: bool = False,
    method_kwargs: Optional[dict] = None,
    verbose: bool = False,
    skip_threshold=_UNSET,
    strategy: Optional[str] = None,
):
    """Internal mover dispatch — no transfer, no helper wrap.

    Identical to the body of :func:`smooth_mesh_interior` minus the
    Phase-1 transfer wrap. Composite adapt ops (``_ot_adapt_step``,
    ``follow_metric``) own the wrap at their level and call this bare
    form to avoid nesting the snapshot/restore dance. End-users should
    keep using :func:`smooth_mesh_interior`.
    """
    if pinned_labels is None:
        pinned_labels = _auto_pinned_labels(mesh)
    pinned_labels = tuple(pinned_labels)

    # Resolve strategy defaults — individual kwargs override.
    # "off" → early-exit, mesh stays uniform.
    if strategy is not None:
        if strategy not in ADAPT_STRATEGIES:
            raise ValueError(
                f"unknown strategy {strategy!r}; choose from "
                f"{list(ADAPT_STRATEGIES.keys())}")
        if strategy == "off":
            if verbose:
                print("  smooth_mesh_interior: strategy='off' "
                      "→ skipping", flush=True)
            return
        _s = ADAPT_STRATEGIES[strategy]
        if skip_threshold is _UNSET:
            skip_threshold = _s["skip_threshold"]
        # method_kwargs: fill in resolution_ratio from strategy
        # if caller didn't already set it.
        if method_kwargs is None:
            method_kwargs = {}
        else:
            method_kwargs = dict(method_kwargs)
        # TODO(BUG): this injection is unconditional, but only the
        # 'anisotropic' and 'mmpde' movers accept resolution_ratio —
        # strategy= combined with the default method='spring' (or
        # 'ma'/'ot') raises TypeError at the mover call. Pre-existing
        # at the Wave D base (verified by signature-binding probe);
        # spring/MA/OT are retired in favour of 'mmpde' (see #346
        # context), so the fix is to inject only for movers that
        # accept it (or route strategy= to a surviving mover).
        method_kwargs.setdefault(
            "resolution_ratio", _s["resolution_ratio"])
    if skip_threshold is _UNSET:
        skip_threshold = None

    if metric is not None:
        mk = dict(method_kwargs or {})
        # Skip-if-good-enough: compare current cell sizes to what
        # the metric would prescribe via equidistribution and bail
        # out early when the mesh is already aligned. Cheap (one
        # evaluate + a few NumPy reductions) — avoids a redundant
        # mover call when the mesh hasn't drifted from its target.
        # Mismatch is measured against the R-clamped achievable
        # target (when the anisotropic mover's resolution_ratio is
        # given), so a perfectly-adapted mesh measures ~0.
        if skip_threshold is not None:
            _R = mk.get("resolution_ratio", None)
            mm = mesh_metric_mismatch(
                mesh, metric, resolution_ratio=_R)
            # `misalignment` = √(1 - r²) where r is Pearson of
            # log(1/A_c) vs log(ρ_c). 0 ⇒ mesh density is
            # perfectly aligned with the metric; 1 ⇒ uncorrelated.
            # Skip when misalignment is below threshold.
            #
            # COLLECTIVE remesh decision: the mover is a collective operation,
            # so the skip/adapt choice MUST be unanimous or the ranks deadlock.
            # `misalignment` is reduced globally (mesh_metric_mismatch) so it is
            # already identical on every rank; the OR-reduction below is the
            # belt-and-suspenders guarantee that **if any rank needs to remesh,
            # all ranks remesh** (and all skip together otherwise).
            _need_adapt = bool(_global_max(
                mm["misalignment"] >= float(skip_threshold)))
            if not _need_adapt:
                if verbose and uw.mpi.rank == 0:
                    print(f"  smooth_mesh_interior: skipping "
                          f"(misalignment {mm['misalignment']:.3f} "
                          f"< threshold {float(skip_threshold):.3f}; "
                          f"alignment r={mm['alignment']:.3f})",
                          flush=True)
                return
            if verbose and uw.mpi.rank == 0:
                print(f"  smooth_mesh_interior: adapting "
                      f"(misalignment {mm['misalignment']:.3f} ≥ "
                      f"threshold {float(skip_threshold):.3f}; "
                      f"alignment r={mm['alignment']:.3f})",
                      flush=True)
        if method == "spring":
            _spring_equilibrium_mover(mesh, metric, pinned_labels, verbose,
                            boundary_slip=boundary_slip, **mk)
        elif method in ("ma", "monge-ampere", "monge_ampere"):
            _monge_ampere_mover(mesh, metric, pinned_labels, verbose,
                              boundary_slip=boundary_slip, **mk)
        elif method in ("ot", "equidistribute", "improve"):
            # The OT / equidistribution mover is incomplete — e.g. its boundary
            # slip is gated to radial geometries (box boundaries are pinned, not
            # slid; see boundary-slip-strategy.md) — and is expected to be
            # superseded by ``method='mmpde'`` with a scalar metric. This fires
            # for every OT use, including the internal ``mesh.OT_adapt`` reset
            # path. (Python shows a given DeprecationWarning once per location.)
            warnings.warn(
                "smooth_mesh_interior(method='ot'/'equidistribute'/'improve') "
                "is an incomplete mesh mover (boundary slip is gated to radial "
                "geometries) and is expected to be superseded by "
                "method='mmpde' with a scalar metric. Prefer 'mmpde' for "
                "production adaptive meshing.",
                DeprecationWarning, stacklevel=2)
            _ot_improvement_step(mesh, metric, pinned_labels,
                                     verbose,
                                     boundary_slip=boundary_slip,
                                     **mk)
        elif method in ("anisotropic", "aniso", "tensor"):
            _winslow_anisotropic(mesh, metric, pinned_labels,
                                 verbose,
                                 boundary_slip=boundary_slip, **mk)
        elif method in ("mmpde", "variational"):
            _mmpde_mover(mesh, metric, pinned_labels, verbose,
                           boundary_slip=boundary_slip, **mk)
        else:
            raise ValueError(
                f"smooth_mesh_interior: unknown method {method!r}; "
                f"use 'spring' (default, fast volumetric), "
                f"'ma' (Monge–Ampère, isotropic, ~60× costlier), "
                f"'anisotropic' (tensor metric — reshapes cells / "
                f"removes slivers; does not beat the node-count "
                f"cap), 'mmpde' (variational moving mesh — the "
                f"recommended production mover) or "
                f"'ot' / 'equidistribute' (deprecated linear "
                f"OT-improvement step).")
        return

    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    cStart, cEnd = dm.getHeightStratum(0)
    cone_size = dm.getConeSize(cStart) if cEnd > cStart else 0
    cache_key = (id(mesh), pinned_labels,
                 pEnd - pStart, cEnd - cStart, cone_size)

    cache = _ADJ_CACHE.get(cache_key)
    if cache is None:
        A, dm_scalar, gsection = _build_adjacency_matrix(mesh)
        # A scratch global Vec of the right shape — also used to read
        # the ownership range when packing/unpacking coord components.
        x_vec = A.createVecRight()
        y_vec = A.createVecLeft()
        ones = A.createVecLeft()
        ones.set(1.0)
        degrees = A.createVecLeft()
        A.mult(ones, degrees)
        owned_local, owned_vec_pos, is_owned = (
            _build_local_to_owned_map(dm, gsection, x_vec))
        is_pinned = _pinned_mask(dm, pinned_labels)
        _ADJ_CACHE[cache_key] = (
            A, dm_scalar, gsection, x_vec, y_vec, degrees,
            owned_local, owned_vec_pos, is_owned, is_pinned)
    else:
        (A, dm_scalar, gsection, x_vec, y_vec, degrees,
         owned_local, owned_vec_pos, is_owned, is_pinned) = cache

    # is_int_owned over the LOCAL chart — selects interior owned
    # vertices for displacement reporting.
    is_int_owned = is_owned & ~is_pinned
    # Subset of owned_local that's also interior (i.e. not pinned)
    # — used to write the per-sweep updates into the numpy buffer.
    int_mask_on_owned = ~is_pinned[owned_local]
    int_owned_local = owned_local[int_mask_on_owned]
    int_owned_vec_pos = owned_vec_pos[int_mask_on_owned]

    coord_dm = dm.getCoordinateDM()
    local_vec = dm.getCoordinatesLocal()
    global_vec = dm.getCoordinates()
    cdim = mesh.cdim
    parallel = uw.mpi.size > 1

    coords = np.asarray(
        local_vec.array, dtype=np.float64).reshape(-1, cdim).copy()

    for sweep in range(n_iters):
        new_int = np.empty((int_owned_local.shape[0], cdim),
                           dtype=np.float64)
        # For each coordinate component, do A @ coord_comp (PETSc
        # handles cross-rank communication), then divide by degree
        # to get the per-vertex neighbour average.
        for d in range(cdim):
            x_vec.array[owned_vec_pos] = coords[owned_local, d]
            A.mult(x_vec, y_vec)
            y_vec.pointwiseDivide(y_vec, degrees)
            avg_owned = np.asarray(y_vec.array)
            new_int[:, d] = (
                (1.0 - alpha) * coords[int_owned_local, d]
                + alpha * avg_owned[int_owned_vec_pos])

        if verbose:
            disp = float(np.linalg.norm(
                new_int - coords[int_owned_local]))
            if parallel:
                disp = _global_sum(disp ** 2) ** 0.5
            uw.pprint(
                f"  smooth_mesh_interior sweep "
                f"{sweep+1}/{n_iters}: "
                f"||Δx||_interior = {disp:.3e}")

        coords[int_owned_local] = new_int

        if parallel:
            # Halo exchange so the next sweep sees updated owned
            # values on every rank's ghost copies. (PETSc's mat.mult
            # handles cross-rank READS internally via the matrix's
            # column communication, so this halo exchange is only
            # needed to keep the LOCAL coord array consistent for
            # the final ``mesh._deform_mesh`` call.)
            local_vec.array[:] = coords.ravel()
            coord_dm.localToGlobal(
                local_vec, global_vec, addv=False)
            coord_dm.globalToLocal(global_vec, local_vec)
            coords[:] = np.asarray(
                local_vec.array).reshape(-1, cdim)

    mesh._deform_mesh(coords)


# =============================================================================
# Public node-moving adapter
# =============================================================================
def follow_metric(
    mesh,
    field,
    *,
    refinement: float,
    coarsening="auto",
    metric: str = "front-following",
    skip_threshold: float = 0.9,
    gradient_smoothing_length=None,
    polish_max_iters: int = 5,
    polish_quality_target: float = 0.3,
    polish_alpha: float = 0.2,
    method_kwargs: Optional[dict] = None,
    name: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    r"""Move the mesh's interior nodes so cell sizes follow a target
    derived from ``|∇field|``.

    Two-knob, cell-size-envelope API for the anisotropic node mover.
    The user specifies how *fine* the densest cells can get and
    (optionally) how *coarse* the sparsest can get; the function
    derives the metric density and invokes the mover.

    Cell-size envelope (approximate)
    --------------------------------

    The mover's eigenvalue → cell-size map is
    :math:`h = h_0/\sqrt{\hat\rho}` (after the mover's
    geometric-mean normalisation :math:`\hat\rho = \rho/G`), so
    asking for the envelope

    .. math::

        h \;\in\; \bigl[\, h_0/\text{refinement},\;
                          h_0\cdot\text{coarsening} \,\bigr]

    corresponds to :math:`\hat\rho \in [1/\text{coarsening}^2,
    \text{refinement}^2]` — note this is **dimension-
    independent** (the eigenvalue λ has units of 1/length²).

    Validation on a sharp-tanh annulus test problem shows:

    * **Refinement side:** achieved :math:`h_\min` within ~5-10%
      of :math:`h_0/\text{refinement}` for refinement ∈ [1.5, 3].
    * **Coarsening side:** achieved :math:`h_\max` typically
      ~2× the requested :math:`h_0\cdot\text{coarsening}`. The
      mover's anisotropic cells and iterative deformation map
      together don't honour the eigenvalue clamp on a per-cell
      basis as tightly as the refinement side. This is a known
      feature of the underlying mover, not of the new API.

    The :func:`mesh_metric_mismatch` diagnostic is the right tool
    for measuring how close the achieved mesh is to the requested
    metric in practice.

    Metric ansatz
    -------------

    Each cell's percentile rank :math:`p \in [0,1]` in the global
    :math:`|\nabla\text{field}|` distribution maps to the
    log-density via a piecewise-linear function with the break
    :math:`\rho = 1` at

    .. math::

        p^{\ast} \;=\; \frac{\log\text{refinement}}
                            {\log(\text{refinement}\cdot
                                  \text{coarsening})} .

    This break point makes :math:`\mathrm{geomean}(\rho) = 1`
    by construction, so the mover's :math:`G`-normalisation
    leaves :math:`\rho` unshifted and the eigenvalue clamps land
    on the desired envelope. Concretely:

    * "front-following" (default) — log-:math:`\rho` is linear
      in percentile rank on each side of :math:`p^{\ast}`. Every
      1% of cells contributes the same log(h) increment. Mild
      grading; the budget is spread continuously across the
      gradient distribution.
    * "gradient-uniform" — :math:`\rho \propto |\nabla\text{field}|^2`,
      clipped to the envelope. Targets uniform per-cell
      :math:`\Delta\text{field}` (the natural goal for advection-
      diffusion accuracy). The clipping makes the achieved
      grading regress to the front-following profile when the
      gradient distribution is concentrated.
    * "arc-length" — smooth arc-length monitor
      :math:`\rho = \sqrt{1 + (A\,|\nabla\text{field}|/g_{hi})^2}`,
      clipped to the envelope. Grades continuously from
      :math:`\rho = 1` in flat regions (no clip kink), giving
      cleaner OT / Monge–Ampère meshes.

    Auto coarsening (the budget-conserving default)
    -----------------------------------------------

    With a fixed node count (no remeshing), refining one cell to
    :math:`h_0/\text{refinement}` requires growing others by at
    least

    .. math::

        \text{coarsening} \;=\; \text{refinement}^{\,1/d}

    to absorb the freed cell area. ``coarsening="auto"`` (default)
    picks exactly this minimum — anything less would mean the
    mover can't actually deliver the requested refinement.
    Pass an explicit ``coarsening>auto`` to free up more budget
    for a smoother transition zone.

    Adapt-on-demand
    ---------------

    Before invoking the mover, the current mesh is checked against
    the requested target via
    :func:`mesh_metric_mismatch`. If the alignment is already good
    (misalignment below ``skip_threshold``), the mesh isn't
    re-adapted — the function returns ``False`` and the caller can
    keep stepping. This lets a per-step adapt cadence become
    "adapt only when needed".

    Parameters
    ----------
    mesh : underworld3 mesh
        Modified in place if adaptation runs.
    field : MeshVariable or sympy scalar expression
        The field whose gradient drives refinement.
    refinement : float, must be >= 1.0
        Maximum local refinement, expressed as a multiplicative
        factor on the background cell size:
        :math:`h_\min = h_0 / \text{refinement}`. ``refinement=1``
        is a no-op (uniform metric ⇒ background spacing).
    coarsening : float or "auto", default "auto"
        Maximum local coarsening,
        :math:`h_\max = h_0 \cdot \text{coarsening}`. ``"auto"``
        uses the budget-conserving minimum
        :math:`\text{refinement}^{1/d}`. Larger values free more
        budget for smoother grading at the cost of a wider
        cell-size spread.
    metric : {"front-following", "gradient-uniform", "arc-length"}, default "front-following"
        Strategic equidistribution rule. ``"front-following"``
        concentrates cells where the gradient is steepest (mild
        grading). ``"gradient-uniform"`` aims for the same
        per-cell field change everywhere (best for advection-
        diffusion accuracy). ``"arc-length"`` is a smooth
        arc-length monitor — grades continuously from flat
        regions with no clip kink (cleaner OT / Monge–Ampère
        meshes).
    skip_threshold : float, default 0.9
        Misalignment threshold for the adapt-on-demand skip. If the
        existing mesh's :func:`mesh_metric_mismatch` ``misalignment``
        (``= sqrt(1 - max(0, r)**2)``, 0 = perfectly aligned) is
        *below* this threshold, no adaptation happens and the
        function returns ``False``. The default 0.9 corresponds to
        skipping once alignment ``r`` exceeds ~0.44.
    gradient_smoothing_length : float or Pint Quantity, optional
        Length scale for screened-Poisson smoothing of the
        projected ``|∇field|`` before building the metric.
        Suppresses sub-cell metric-mesh feedback noise without
        destroying boundary-layer features. A useful default is
        ``≈ 2 * h_0`` (background cell size).
    polish_max_iters : int, default 5
        Maximum Jacobi (graph-Laplacian) polish iterations
        applied AFTER the anisotropic mover. The polish runs
        adaptively: each iteration averages every interior
        vertex toward the mean of its edge neighbours
        (cell-quality cleanup), and the loop stops as soon as
        the worst cell-shape quality exceeds
        ``polish_quality_target``. ``polish_max_iters=0``
        disables the polish entirely.
    polish_quality_target : float, default 0.3
        Adaptive-polish stopping criterion: target minimum
        cell shape quality
        :math:`q = 4\sqrt{3}\,A/(e_0^2+e_1^2+e_2^2)`. ``q=1``
        is equilateral; ``q<0.3`` is the threshold below which
        cells look like visible slivers. Lower values allow
        more sliver-y cells through; higher values demand
        more polish iterations.
    polish_alpha : float, default 0.2
        Under-relaxation in ``(0, 1]`` for each Jacobi
        sweep. Lower = gentler.
    name : str, optional
        Cache disambiguator. Pass distinct names if you build
        several independent metrics on the same mesh.
    verbose : bool, default False
        Verbose mover diagnostics.

    Returns
    -------
    bool
        ``True`` if the mesh was moved; ``False`` if the
        skip-on-mismatch check short-circuited adaptation.

    Examples
    --------
    Default usage on a stagnant-lid convection T field, with
    coarsening picked automatically::

        moved = uw.meshing.follow_metric(
            mesh, T,
            refinement=3.0,                  # h_min = h0/3
        )                                    # coarsening = √3 ≈ 1.73 (2D auto)

    Wider grading transition with explicit coarsening, gradient-
    side smoothing, and the gradient-uniform rule for advection
    accuracy::

        uw.meshing.follow_metric(
            mesh, T,
            refinement=2.0, coarsening=2.0,
            metric="gradient-uniform",
            gradient_smoothing_length=2.0 * mesh._radii.mean(),
        )

    See Also
    --------
    metric_density_from_gradient : The underlying metric builder
        (expert tool — exposes percentile / amp / power dials).
    smooth_mesh_interior : The underlying mover (expert tool —
        unaware of refinement/coarsening, takes a pre-built
        metric expression).
    mesh_metric_mismatch : The alignment / misalignment metric
        used by the skip threshold.
    """
    rho = metric_density_from_gradient(
        mesh,
        field,
        refinement=float(refinement),
        coarsening=coarsening,
        metric_choice=metric,
        gradient_smoothing_length=gradient_smoothing_length,
        name=name,
    )
    # Resolve auto coarsening
    if coarsening is None or coarsening == "auto":
        coar_val = float(refinement) ** (1.0 / mesh.cdim)
    else:
        coar_val = float(coarsening)
    # Mover's `resolution_ratio` is a SYMMETRIC eigenvalue clamp
    # (h ∈ [h0/R, h0·R]) — too loose for either side on its own.
    # We pass R = max(refinement, coarsening) so the clamp doesn't
    # bind tightly, then rely on the per-cell *rest-size spring*
    # (below) to enforce the literal cell-size envelope.
    R = max(float(refinement), coar_val)

    # The spring caps refer to h0 — the **undeformed** mean edge
    # length, captured ONCE per mesh and reused (the dt-crash /
    # compounding-refinement bug, 2026-05-22 — full story on the
    # _FOLLOW_METRIC_H0_CACHE declaration and _mean_edge_length).
    _key = id(mesh)
    h0 = _FOLLOW_METRIC_H0_CACHE.get(_key)
    rest_coords = _FOLLOW_METRIC_REST_CACHE.get(_key)
    if h0 is None:
        coords = np.asarray(mesh.X.coords)
        h0 = _mean_edge_length(mesh.dm, coords)
        _FOLLOW_METRIC_H0_CACHE[_key] = h0
        rest_coords = coords.copy()
        _FOLLOW_METRIC_REST_CACHE[_key] = rest_coords
        if verbose:
            uw.pprint(f"  follow_metric: captured h0={h0:.4e}, "
                      f"rest_coords (first call on this mesh)")

    mover_kwargs = dict(
        relax=0.2,
        n_outer=12,
        # Per-cell Lagrangian rest-size spring: literal cell-size
        # cap enforced by pulling vertices back toward their
        # rest positions when an incident cell exceeds the cap.
        # h0 is the undeformed mean edge length.
        rest_size_cap_max=h0 * coar_val,
        rest_size_cap_min=h0 / float(refinement),
        rest_spring_K=1.0,
        # Override the mover's internal h0 measurement (which
        # would otherwise re-measure on the already-deformed
        # mesh and shrink each adapt — the second leg of the
        # dt-crash bug surfaced 2026-05-22).
        h0_override=h0,
        # Override the spring's rest-coords (and the area-floor
        # baseline) so they refer to the **truly-undeformed**
        # mesh. Otherwise each adapt's "rest" is the previous
        # adapt's output, the spring "preserves" each successive
        # refinement, and refinement compounds — third leg of
        # the dt-crash bug.
        rest_coords_override=rest_coords,
    )
    if method_kwargs:
        mover_kwargs.update(method_kwargs)

    # Phase-1 remesh redesign: wrap the whole anisotropic-move + polish
    # pipeline in a single field-transfer pass at this composite level.
    # The inner smooth_mesh_interior calls see ``mesh._in_remesh_transfer``
    # set by the helper and skip their own wrap, so REMAP variables
    # (including hidden solver history) are transferred exactly once,
    # after the polish.
    from underworld3.discretisation.remesh import (
        remesh_with_field_transfer)
    _state = {"moved": False}

    def _do_move():
        _old_X = np.asarray(mesh.X.coords).copy()
        smooth_mesh_interior(
            mesh,
            metric=rho,
            method="anisotropic",
            method_kwargs={**mover_kwargs, "resolution_ratio": R},
            skip_threshold=skip_threshold,
            verbose=verbose,
        )
        _new_X = np.asarray(mesh.X.coords)
        _state["moved"] = not np.allclose(_new_X, _old_X)
        _polish(_state["moved"])

    def _polish(moved):
        # ADAPTIVE Jacobi polish: gentle graph-Laplacian smoothing
        # of interior nodes toward neighbour-centroid average,
        # repeated until the worst cell-shape quality
        #
        #     q = 4√3 · A / (e₀² + e₁² + e₂²)
        #
        # exceeds ``polish_quality_target`` (default 0.3 — the
        # threshold below which cells look like visible slivers; an
        # equilateral has q=1, a degenerate sliver q→0). Capped at
        # ``polish_max_iters`` so pathological cases can't run away.
        #
        # The polish doesn't significantly undo the metric
        # distribution (each step is averaging toward neighbours,
        # not enforcing any spatial target), so the BL refinement
        # stays intact while sliver cells get rounded out.
        # `polish_max_iters=0` disables entirely.
        if moved and polish_max_iters > 0:
            tris_polish = _tri_cells(mesh.dm)
            for _polish_iter in range(int(polish_max_iters)):
                # Check current shape quality
                p = np.asarray(mesh.X.coords)[tris_polish]
                e0 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
                e1 = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
                e2 = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
                A = np.abs(_signed_areas(np.asarray(mesh.X.coords),
                                           tris_polish))
                q = (4.0 * np.sqrt(3.0) * A
                     / (e0 * e0 + e1 * e1 + e2 * e2 + 1.0e-30))
                q_min = _global_min(q.min())
                if verbose:
                    uw.pprint(
                        f"  follow_metric polish iter {_polish_iter}: "
                        f"q_min={q_min:.3f} (target {polish_quality_target:.2f})")
                if q_min >= float(polish_quality_target):
                    break
                smooth_mesh_interior(
                    mesh, n_iters=1, alpha=float(polish_alpha))

    remesh_with_field_transfer(mesh, _do_move, verbose=verbose)
    return _state["moved"]
