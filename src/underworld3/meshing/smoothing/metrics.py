"""Named adaptation strategies, the |∇field|-driven metric-density
builder, and the mesh/metric alignment diagnostic. See the package
docstring for the module map.
"""

from typing import Optional

import numpy as np

import underworld3 as uw

from .graph import _tri_cells, _signed_areas, _global_sum, _global_max


# Named adaptation strategies (off / vlow / low / med / high /
# extreme). Each maps to a coherent set of (amp, percentile
# window, power, R, skip_threshold) values. Use the
# ``strategy=`` kwarg on :func:`metric_density_from_gradient`
# and :func:`smooth_mesh_interior` to dial intensity; individual
# kwargs still work and override the strategy choice where given.
ADAPT_STRATEGIES = {
    "off":     dict(amp=0.0, lo_percentile=0.0,
                    hi_percentile=100.0, power=1.0,
                    resolution_ratio=1.0,
                    skip_threshold=None,
                    description="no adaptation (no-op)"),
    "vlow":    dict(amp=4.0, lo_percentile=80.0,
                    hi_percentile=99.0, power=1.0,
                    resolution_ratio=1.2,
                    skip_threshold=0.9,
                    description="hardly any refinement; "
                                "top 20% gradient cells only"),
    "low":     dict(amp=6.0, lo_percentile=70.0,
                    hi_percentile=97.0, power=1.0,
                    resolution_ratio=1.3,
                    skip_threshold=0.9,
                    description="gentle front bunching"),
    "med":     dict(amp=7.0, lo_percentile=60.0,
                    hi_percentile=97.0, power=1.0,
                    resolution_ratio=1.4,
                    skip_threshold=0.9,
                    description="moderate front bunching "
                                "(default)"),
    "high":    dict(amp=8.0, lo_percentile=50.0,
                    hi_percentile=97.0, power=1.0,
                    resolution_ratio=1.5,
                    skip_threshold=0.9,
                    description="front-following — historical "
                                "production point"),
    "extreme": dict(amp=8.0, lo_percentile=50.0,
                    hi_percentile=97.0, power=1.5,
                    resolution_ratio=2.0,
                    skip_threshold=0.9,
                    description="midway to gradient-uniform; "
                                "near the danger zone for the "
                                "mover — use deliberately"),
}


# Sentinel used to detect whether a kwarg was explicitly set by
# the caller versus left at the function default. Lets us layer
# strategy defaults beneath explicit user overrides cleanly.
_UNSET = object()


def mesh_metric_mismatch(mesh, metric, resolution_ratio=None):
    r"""Geometric mismatch between the current mesh and what the
    equidistribution rule would prescribe from ``metric``.

    Per cell compute the equidistribution-prescribed area
    ``A_target = A_total · (1/ρ_c) / Σ(1/ρ)`` (the conservation
    law of §1 in ``mesh-adaptation-formulation.md``). When the
    mover's eigen-clamp ``[h0/R, h0·R]`` is in play, clip the
    target so it represents what the mover can *actually*
    achieve, not the unbounded ideal. Then

    .. math::

        \delta_c = \tfrac12\,\log\!\Big(
            \frac{A_{\mathrm{actual},c}}{A_{\mathrm{target},c}}\Big)

    (signed, log-space symmetric: a 2× refine needed = +0.35;
    a 2× coarsen needed = -0.35). Scale-invariant under
    ``ρ → αρ``.

    Returns a 5-key dict: ``rms`` / ``max`` / ``median_abs``
    summarise ``|δ|`` over cells (a mesh already at the mover's
    achievable equidistribution gives ~0; a pre-adapted mesh
    against a strongly-peaked metric gives O(1) or larger), plus
    ``alignment`` / ``misalignment`` — the magnitude-free signal
    the ``skip_threshold`` machinery consumes (see Returns).

    Cheap: one ``metric`` evaluate at cell centroids + a few
    NumPy reductions. Used by
    :func:`smooth_mesh_interior(skip_threshold=...)` to skip
    adapting when the mesh is already aligned with the target.

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
        Triangle (2D) or tetrahedral (3D) mesh; "area" throughout
        this docstring reads as the cell measure (volume in 3D).
    metric : sympy / UW expression
        The target *density* ρ (larger ⇒ finer cells) — same
        object you would pass to ``smooth_mesh_interior``.
    resolution_ratio : float, optional
        The mover's eigen-clamp ``R``. When given, the
        equidistribution target areas are clipped to
        ``[A_mean / R², A_mean · R²]`` — the achievable band the
        mover honours — so a perfectly-adapted mesh measures
        ``δ ≈ 0``. Without it, mismatch is measured against the
        unbounded equidistribution target (so even a
        perfectly-adapted mesh has ``δ ≠ 0`` against the
        unreachable ideal).

    Returns
    -------
    dict
        ``{"rms", "max", "median_abs", "alignment", "misalignment"}``
        (all float):

        * ``rms`` / ``max`` / ``median_abs`` — moments of ``|δ|``
          over cells.
        * ``alignment`` — Pearson r of ``log(1/A_c)`` vs
          ``log(ρ_c)``, computed from GLOBALLY reduced moment sums
          so every rank sees the same value (a rank-local corrcoef
          would diverge across ranks and deadlock the collective
          mover).
        * ``misalignment`` — ``sqrt(1 − max(0, r)²)``: 0 = cell
          density perfectly aligned with the metric, 1 = orthogonal
          (a negative correlation clamps to r=0 ⇒ misalignment 1.0).
          This is the skip criterion consumed by
          ``smooth_mesh_interior(skip_threshold=...)``.
    """
    import underworld3 as _uw

    # Rank-SYMMETRIC refusal for non-simplex meshes: mesh.isSimplex is a
    # rank-uniform property, so every rank raises together. (The per-rank
    # tris-is-None check below cannot serve — a starved rank has no cells
    # to inspect and would sail into the collective reductions while the
    # populated ranks raise; issue #351 review.)
    if not mesh.isSimplex:
        raise NotImplementedError(
            "mesh_metric_mismatch: simplex (triangle/tetrahedral) mesh "
            "required")

    coords = np.asarray(mesh.X.coords)
    cStart, cEnd = mesh.dm.getHeightStratum(0)
    if cEnd > cStart:
        if mesh.cdim == 2:
            tris = _tri_cells(mesh.dm)
            A_actual = None if tris is None else np.abs(
                _signed_areas(coords, tris))
        else:
            from .graph import _tet_cells, _signed_volumes
            tris = _tet_cells(mesh.dm)
            A_actual = None if tris is None else np.abs(
                _signed_volumes(coords, tris))
        if tris is None:
            raise NotImplementedError(
                "mesh_metric_mismatch: simplex (triangle/tetrahedral) mesh "
                "required")
        centroids = coords[tris].mean(axis=1)
        rho = np.asarray(_uw.function.evaluate(
            metric, centroids)).reshape(-1)
        rho = np.maximum(rho, 1.0e-12)   # guard
    else:
        # A rank owning zero cells participates in the global reductions
        # below with empty contributions — raising or returning early here
        # would desynchronise the collectives.
        #
        # KNOWN LIMIT: this skips uw.function.evaluate, which is itself
        # collective for metrics containing MESH-VARIABLE data — a starved
        # rank then deadlocks the populated ranks inside evaluate. Analytic
        # (pure-sympy) metrics are fine: their evaluation is rank-local.
        #
        # Issue #405 made the reduction layer under evaluate empty-rank safe
        # (radii, points_in_domain), which was expected to lift this limit on
        # its own. Measured at np=4 on a starved mesh: it does not. The
        # remaining blocker is below UW3 — the DMPlex sub-DM clone that the
        # mesh-variable path builds fails with MPI_ERR_BUFFER when a rank has
        # no cells. That is issue #314's territory; revisit this branch when
        # #314 closes.
        A_actual = np.empty(0)
        rho = np.empty(0)
    inv_rho = 1.0 / rho if rho.size else np.empty(0)
    # Global equidistribution target (issue #351): the sums and the cell
    # count are reduced across ranks — rank-local sums measured each
    # rank's cells against a per-rank target, so the returned moments
    # were partition-dependent and inconsistent with the docstring.
    sum_A = _global_sum(A_actual.sum())
    sum_inv_rho = _global_sum(inv_rho.sum())
    A_target = sum_A * inv_rho / sum_inv_rho
    if resolution_ratio is not None:
        R = float(resolution_ratio)
        A_mean = sum_A / _global_sum(A_actual.size)
        # Clip target areas to the mover's achievable band
        # [A_mean/R², A_mean·R²] (h in [h0/R, h0·R] ⇒
        # A in [h0²/R², h0²·R²] = [A_mean/R², A_mean·R²]).
        A_target = np.clip(A_target, A_mean / R ** 2,
                           A_mean * R ** 2)
    delta = 0.5 * np.log(A_actual / A_target) if A_actual.size else np.empty(0)
    abs_delta = np.abs(delta)

    # Alignment — Pearson r of log(1/A_c) with log(ρ_c).
    # Equidistribution gives log(1/A) ∝ (1/d)·log(ρ) ⇒ r → 1.
    # Uniform mesh has nearly-zero sd(log A) ⇒ r ≈ 0.
    # An over-aggressive mover that overshoots in proportional
    # fashion still has r ≈ 1 (just with the wrong slope), so r
    # measures whether cell density is *aligned with* the metric,
    # independent of grading magnitude. This is the right signal
    # for "is this mesh built around this metric?" — and the
    # appropriate skip-or-adapt criterion in a dynamic loop.
    log_density = -np.log(A_actual)
    log_rho = np.log(rho)
    # Pearson r of log(1/A_c) vs log(rho_c), from GLOBAL moment sums. Cells are
    # partitioned across ranks, so a rank-local np.corrcoef yields a DIFFERENT
    # alignment on each rank — the skip/adapt decision in smooth_mesh_interior
    # then diverges across ranks and the (collective) mover deadlocks. Reducing
    # the moments makes every rank agree. Serial: identical to np.corrcoef (the
    # 1/n normalisation cancels in the ratio).
    n_c = _global_sum(log_density.size)
    sx = _global_sum(log_density.sum())
    sy = _global_sum(log_rho.sum())
    sxx = _global_sum((log_density * log_density).sum())
    syy = _global_sum((log_rho * log_rho).sum())
    sxy = _global_sum((log_density * log_rho).sum())
    var_x = sxx / n_c - (sx / n_c) ** 2
    var_y = syy / n_c - (sy / n_c) ** 2
    if var_x > 1.0e-24 and var_y > 1.0e-24:
        alignment = float((sxy / n_c - (sx / n_c) * (sy / n_c))
                          / np.sqrt(var_x * var_y))
        alignment = max(-1.0, min(1.0, alignment))
    else:
        alignment = 0.0
    # Misalignment: 0 = perfectly aligned, 1 = orthogonal.
    misalignment = float(
        np.sqrt(max(0.0, 1.0 - max(0.0, alignment) ** 2)))

    # Global moments of |δ| (issue #351): every rank reports the same
    # diagnostics. The exact median needs the values in one place — this
    # is a diagnostic at adapt cadence, so gather |δ| to rank 0 and
    # broadcast the scalar.
    n_cells = _global_sum(delta.size)
    rms = float(np.sqrt(_global_sum((delta ** 2).sum()) / n_cells))
    delta_max = _global_max(abs_delta.max() if abs_delta.size else 0.0)
    if _uw.mpi.size > 1:
        gathered = _uw.mpi.comm.gather(abs_delta, root=0)
        median_abs = (float(np.median(np.concatenate(gathered)))
                      if _uw.mpi.rank == 0 else None)
        median_abs = _uw.mpi.comm.bcast(median_abs, root=0)
    else:
        median_abs = float(np.median(abs_delta))

    return dict(rms=rms,
                max=float(delta_max),
                median_abs=median_abs,
                alignment=alignment,
                misalignment=misalignment)


# Cached (∇field projector, |∇field| density) per (mesh, degree,
# name, topology) so metric_density_from_gradient is cheap and
# leak-free when called every step in an adaptive loop.
_MDG_CACHE: dict = {}


def metric_density_from_gradient(
    mesh,
    field,
    *,
    refinement=None,
    coarsening="auto",
    metric_choice: str = "front-following",
    strategy: str = "med",
    amp=_UNSET,
    lo_percentile=_UNSET,
    hi_percentile=_UNSET,
    power=_UNSET,
    mode: str = "percentile",
    smoothing_length=None,
    gradient_smoothing_length=None,
    degree: int = 1,
    name: Optional[str] = None,
):
    r"""Build a target-**density** metric ``ρ ∝ normalised |∇field|``
    for the metric movers — the relative, fixed-node-budget
    analogue of :func:`underworld3.adaptivity.metric_from_gradient`
    (which maps ``|∇field|`` to an *absolute* target edge length
    for the MMG re-mesher; the mover has a fixed node budget so it
    redistributes *relatively* instead).

    .. math::

        \rho = (1 + \mathrm{amp}\cdot t)^{\mathrm{power}},\qquad
        t = \mathrm{clip}\!\Big(
            \frac{|\nabla\mathrm{field}| - g_{lo}}
                 {g_{hi} - g_{lo}}, 0, 1\Big),

    with ``g_lo, g_hi`` the lo/hi percentiles of ``|∇field|`` (the
    same percentile-window idea as the adaptation metric).

    **What the power knob does (strategic choice).** The mover
    equidistributes ``ρ`` (cell area × ρ ≈ const). Combined with
    ``A_c = h_c^d`` in ``d`` dimensions that gives
    ``h_c ∝ ρ_c^{-1/d}``. For the linear ramp ``ρ ∝ |∇T|`` (i.e.
    ``power=1``, the historical default) this means
    ``h_c ∝ |∇T|^{-1/d}`` and the per-cell temperature change
    ``ΔT_c ≈ |∇T|·h_c ∝ |∇T|^{1-1/d}`` — strong-gradient cells
    still carry MORE temperature change than weak-gradient cells.
    Choosing ``power = d`` (so ``ρ ∝ |∇T|^d``) gives
    ``h_c ∝ 1/|∇T|`` and ``ΔT_c ≈ const`` — a **gradient-uniform
    target**: every cell carries the same temperature change.
    ``power = 1`` (default) targets "refinement of fronts /
    boundaries" (mild grading concentrated where gradients are
    strongest); ``power = d`` targets "uniform per-cell error in
    a piecewise-linear T interpolant" (the natural goal for
    advection-diffusion accuracy). Values in between blend the
    two; ``power < 1`` softens grading further.
    ``|∇field|`` is L2-projected (a *first* derivative — UW3-clean)
    and the normalised ``t`` is stored in a **frozen Lagrangian
    scalar field**, so the returned metric rides material points —
    required by the movers, which build the metric once on the
    undeformed mesh. Pass the result straight to
    :func:`smooth_mesh_interior`::

        rho = metric_density_from_gradient(mesh, T, amp=8.0)
        smooth_mesh_interior(mesh, metric=rho, method="mmpde")

    The projector/fields are cached per ``(mesh, degree, name,
    topology)``, so calling this **every step** in an adaptive loop
    is cheap and does not leak MeshVariables. Each call re-projects
    and re-freezes ``t`` at the *current* field state.

    Parameters
    ----------
    mesh : underworld3 mesh
    field : scalar MeshVariable or sympy scalar expression
        The field whose gradient drives refinement (e.g. ``T``).
    refinement : float, optional
        Maximum local refinement factor on the background cell size:
        the metric targets the cell-size envelope
        ``h ∈ [h0/refinement, h0·coarsening]``. When given, the
        **envelope branch** is taken: ρ is built directly from the
        percentile rank of ``|∇field|`` with ``geomean(ρ) = 1`` by
        construction (see the inline commentary at the envelope
        branch), and ``amp`` / ``lo_percentile`` / ``hi_percentile``
        / ``mode`` / ``power`` are **ignored**. The realised ratio
        tracks the request until the fixed node budget saturates it
        (going further needs h-refinement — more nodes, not
        redistribution). ``None`` (default) ⇒ the legacy
        ``amp``-based percentile-ramp path below. This is the knob
        :func:`follow_metric` exposes.
    coarsening : float or "auto", default "auto"
        Envelope-branch partner of ``refinement``: maximum local
        coarsening factor (``h_max = h0·coarsening``). ``"auto"``
        picks the budget-conserving minimum ``refinement**(1/d)``.
        Ignored when ``refinement`` is None.
    metric_choice : {"front-following", "gradient-uniform", "arc-length"}
        Envelope-branch spatial distribution rule (see the inline
        commentary at the envelope branch). Ignored when
        ``refinement`` is None.
    amp : float, default 8.0
        Bunching intensity: ``ρ_max = (1 + amp)^power`` where
        ``|∇field|`` is strongest. Larger ⇒ stronger
        redistribution.
    power : float, default 1.0
        Exponent applied to the metric. ``1`` (default) =
        front-following (``ρ ∝ |∇T|``, mild grading).
        ``d`` (mesh dimension) = gradient-uniform
        (``ρ ∝ |∇T|^d``, uniform per-cell ΔT). Values in
        between blend; ``<1`` softens. The strategic choice is
        between "refine the fronts" and "uniform per-cell
        error", not a free dial — see the docstring math.
    mode : {"percentile", "raw"}, default "percentile"
        How the gradient drives the metric. ``"percentile"``
        (default): ρ = (1 + amp·t)^power with t the
        percentile-clipped normalised |∇field| — concentrates
        budget into the steepest fronts, ignores values below
        ``lo_percentile``. ``"raw"``: ρ = |∇field|^power
        directly (no offset, no clipping, no amp). The mover's
        equidistribution geometric-mean normalisation handles
        the absolute scale; ``amp`` and ``lo/hi_percentile``
        are ignored. Use ``"raw"`` to target gradient-uniform
        per-cell ΔT cleanly; ``"percentile"`` to refine only
        the top X% of gradient values.
    lo_percentile, hi_percentile : float, default 50 / 97
        ``|∇field|`` normalisation window (cf. the 5th/95th of
        ``adaptivity.metric_from_gradient``). Raise ``lo`` to push
        refinement only into the steepest fronts.
    degree : int, default 1
        Polynomial degree of the projected-gradient / density
        fields (P1 is what the mover's per-vertex metric
        evaluation samples).
    name : str, optional
        Cache disambiguator. Pass distinct names if you build
        several independent gradient metrics on the *same* mesh
        simultaneously (otherwise they share the cache slot).
    smoothing_length : float or Pint Quantity, optional
        Length-scale ``L`` for **field-side** screened-Poisson
        smoothing applied to ``field`` BEFORE the gradient is
        taken. Useful to suppress sub-grid noise in the source.
        WARNING: at ``L ≳`` BL width this *erases* the
        boundary-layer gradient — T's transition is spread over
        ~L and the gradient peak ``T_active/h`` collapses to
        ``T_active/L``. Prefer
        ``gradient_smoothing_length`` when targeting features
        with BL-like sub-h structure.
    gradient_smoothing_length : float or Pint Quantity, optional
        Length-scale ``L`` for **gradient-side** screened-Poisson
        smoothing applied to the projected ``|∇field|`` field
        (via the L2-projection's ``smoothing_length``). Peak
        *location* of ``|∇T|`` is preserved (a BL still
        concentrates near where T transitions); only the
        spatial distribution / mesh-noise in the projection is
        smoothed. This is the principled way to break the
        metric/mesh feedback on adapted meshes without
        destroying BL features. Set ``L ≈ h0`` (background
        mean cell size) for mild de-noising;
        ``L ≈ 2·h0`` for stronger.

    Returns
    -------
    sympy expression
        ``(1 + amp * t.sym[0])**power`` — Lagrangian, frozen at
        call time.
    """
    import sympy

    # Resolve strategy defaults — individual kwargs override.
    if strategy not in ADAPT_STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; choose from "
            f"{list(ADAPT_STRATEGIES.keys())}")
    s = ADAPT_STRATEGIES[strategy]
    if amp is _UNSET:
        amp = s["amp"]
    if lo_percentile is _UNSET:
        lo_percentile = s["lo_percentile"]
    if hi_percentile is _UNSET:
        hi_percentile = s["hi_percentile"]
    if power is _UNSET:
        power = s["power"]

    cdim = mesh.cdim

    X = mesh.CoordinateSystem.X
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    cStart, cEnd = dm.getHeightStratum(0)
    tag = name or "mdg"
    key = (id(mesh), int(degree), tag,
           pEnd - pStart, cEnd - cStart)

    cache = _MDG_CACHE.get(key)
    if cache is None:
        g = uw.discretisation.MeshVariable(
            f"mdg_g_{id(mesh):x}_{tag}{degree}", mesh,
            vtype=uw.VarType.VECTOR, degree=int(degree),
            continuous=True)
        gp = uw.systems.Vector_Projection(mesh, g)
        gp.smoothing = 0.0
        rho0 = uw.discretisation.MeshVariable(
            f"mdg_rho_{id(mesh):x}_{tag}{degree}", mesh,
            vtype=uw.VarType.SCALAR, degree=int(degree),
            continuous=True)
        # Optional pre-smoothing of the input field: a scalar
        # screened-Poisson projection (u − L²∇²u = field) at
        # smoothing_length L. Decouples the gradient computation
        # from sub-L mesh structure, breaking the metric/mesh
        # feedback loop.
        f_smooth = uw.discretisation.MeshVariable(
            f"mdg_fs_{id(mesh):x}_{tag}{degree}", mesh,
            vtype=uw.VarType.SCALAR, degree=int(degree),
            continuous=True)
        fp = uw.systems.Projection(mesh, f_smooth)
        _MDG_CACHE[key] = (g, gp, rho0, f_smooth, fp)
    else:
        g, gp, rho0, f_smooth, fp = cache

    f_sym = (field.sym[0] if hasattr(field, "sym")
             else sympy.sympify(field))
    if smoothing_length is not None:
        # Smooth the input field T at length L before computing
        # ∇T. WARNING: at L ≳ BL width this *erases* the BL
        # gradient — the screened-Poisson spreads T's transition
        # layer over ~L and the gradient peak (T_active/h)
        # collapses to T_active/L. For metric construction
        # against a boundary-layer feature, prefer
        # `gradient_smoothing_length` instead (smooths the
        # projected gradient field rather than T).
        fp.uw_function = f_sym
        fp.smoothing_length = smoothing_length
        fp.solve()
        f_for_grad = f_smooth.sym[0]
    else:
        f_for_grad = f_sym
    gp.uw_function = sympy.Matrix(
        [f_for_grad.diff(X[i]) for i in range(cdim)]).T
    # Apply screened-Poisson smoothing on the *gradient
    # projection* — keeps peak location intact (where T
    # transitions, ∇T peaks), just smooths the spatial
    # distribution. This is the principled way to suppress
    # mesh-induced noise in |∇T| without erasing BL features.
    if gradient_smoothing_length is not None:
        gp.smoothing_length = gradient_smoothing_length
    else:
        gp.smoothing = 0.0
    gp.solve()
    gmag = np.linalg.norm(np.asarray(uw.function.evaluate(
        g.sym, rho0.coords)).reshape(-1, cdim), axis=1)
    # Parallel-correct percentile window. np.percentile on the
    # rank-LOCAL gmag gives each rank its *own subdomain*
    # distribution, so the same physical |∇field| maps to a
    # different density on different ranks — a partition-dependent
    # metric ("refine the top X%" silently becomes "each rank's own
    # top X%"). Gather the global gmag so g_lo/g_hi are computed
    # once from the whole-domain distribution and are identical on
    # every rank. Serial (size==1) takes the local array unchanged
    # ⇒ bit-for-bit identical to the previous behaviour. (Partition-
    # boundary DOFs are shared across ranks, so the gathered array
    # slightly over-weights them in the percentile value — a
    # second-order effect vs. the rank-local bug this fixes; exact
    # owned-only de-duplication is a follow-up if ever needed.)
    if uw.mpi.size > 1:
        gmag_global = uw.utilities.gather_data(
            gmag, bcast=True, dtype="float64")
    else:
        gmag_global = gmag
    g_lo = float(np.percentile(gmag_global, lo_percentile))
    g_hi = float(np.percentile(gmag_global, hi_percentile))
    # No-op guard: a uniform field has |∇field| ≡ 0, but the L2
    # projection leaves ~1e-18 round-off. Percentile-normalising
    # that noise would fabricate a spurious [0,1] metric (the same
    # failure the mover's own g_eps floor fixes). Any real field
    # gradient is many orders above 1e-9 ⇒ a (near-)constant field
    # yields ρ ≡ 1 (no refinement) exactly.

    # NEW PATH: cell-size-envelope ansatz keyed by
    # ``refinement`` (+ optional ``coarsening``).
    #
    # The mover's eigenvalue → cell-size map is ``h = h₀/√(ρ̂)``
    # (after the mover's geometric-mean normalisation ρ̂ = ρ/G).
    # So a literal envelope ``h ∈ [h₀/refinement, h₀·coarsening]``
    # corresponds to ``ρ̂ ∈ [1/coarsening², refinement²]`` — note
    # this is **dimension-independent** (the eigenvalue λ has
    # units of 1/length², not 1/area).
    #
    # To make the mover's G normalisation land where we want, we
    # build ρ with ``geomean(ρ) ≡ 1`` by construction. The cleanest
    # form is piecewise-log-linear in the percentile rank ``pct``
    # of |∇field|, with the break ρ=1 placed at
    #
    #     p* = log(refinement) / log(refinement · coarsening)
    #
    # which is exactly the fraction of cells that need to coarsen
    # to ``free up`` the requested refinement at fixed node count.
    #
    # ``metric_choice`` selects the spatial *distribution*:
    #
    # * "front-following" — log(ρ) piecewise linear in pct rank.
    #   Every 1% of cells contributes the same log(h) increment.
    #   Mild, monotone grading concentrated on the high-gradient
    #   tail.
    # * "gradient-uniform" — ρ ∝ |∇field|², clipped to the
    #   envelope. Targets uniform per-cell Δfield (the natural
    #   goal for advection-diffusion accuracy).
    # * "arc-length" — smooth arc-length monitor
    #   ρ = √(1 + (A·|∇field|/g_hi)²), clipped to the envelope.
    #   Grades continuously from ρ=1 in flat regions (no clip
    #   kink) → the smoothest equidistributed meshes.
    #
    # ``coarsening="auto"`` uses the budget-conserving minimum
    # ``refinement^(1/d)`` — the smallest coarsening that
    # geometrically "makes room" for the requested refinement at
    # fixed node count.
    #
    # When the caller passes ``refinement=...``, this branch is
    # taken and amp/lo_percentile/hi_percentile/mode/power are
    # ignored — the envelope is determined directly.
    if refinement is not None:
        ref_val = float(refinement)
        if ref_val < 1.0:
            raise ValueError(
                f"refinement must be >= 1.0, got {ref_val}")
        # 'auto' coarsening = the budget-conserving minimum
        if coarsening is None or coarsening == "auto":
            coar_val = ref_val ** (1.0 / cdim)
        else:
            coar_val = float(coarsening)
            if coar_val < 1.0:
                raise ValueError(
                    f"coarsening must be >= 1.0, got {coar_val}")
        # Trivial-case shortcut: no refinement asked ⇒ ρ ≡ 1
        if ref_val == 1.0 and coar_val == 1.0:
            rho0.data[:, 0] = 1.0
            return rho0.sym[0]
        # Dimension-independent envelope (eigenvalue space)
        log_rho_max = 2.0 * np.log(ref_val)   # ρ at the densest cells
        log_rho_min = -2.0 * np.log(coar_val)  # ρ at the sparsest cells
        N = max(int(gmag_global.size), 1)
        if g_hi <= 1.0e-9:
            # Uniform (or near-uniform) field ⇒ no refinement
            rho0.data[:, 0] = 1.0
            return rho0.sym[0]
        g_sorted = np.sort(gmag_global)
        ranks = np.linspace(0.0, 1.0, N)
        pct = np.interp(gmag, g_sorted, ranks)
        if metric_choice == "front-following":
            # Piecewise log-linear in pct, with the break (log ρ=0)
            # at p* = log(ref) / log(ref·coar). This makes
            # geomean(ρ) = 1 by construction, so the mover's G
            # normalisation passes ρ through unchanged and the
            # eigenvalue clamps land on the literal envelope.
            # Special-case: ref=1 ⇒ no refined half (pure coarsen);
            #               coar=1 ⇒ no coarsened half (pure refine).
            if ref_val == 1.0:
                # Only coarsen
                log_rho = log_rho_min * (1.0 - pct)
            elif coar_val == 1.0:
                # Only refine
                log_rho = log_rho_max * pct
            else:
                p_star = (np.log(ref_val)
                          / np.log(ref_val * coar_val))
                log_rho = np.where(
                    pct < p_star,
                    log_rho_min * (1.0 - pct / p_star),
                    log_rho_max * (pct - p_star)
                    / max(1.0 - p_star, 1.0e-12),
                )
            rho0.data[:, 0] = np.exp(log_rho)
        elif metric_choice == "gradient-uniform":
            # ρ ∝ |∇field|² (dimension-independent), clipped to
            # the envelope. The mover's G normalisation then
            # centres this on whatever cell happens to have the
            # geomean |∇field|, which is field-dependent (in
            # contrast to front-following where ρ̄=1 by construction).
            rho_raw = np.maximum(gmag, 1.0e-30) ** 2
            rho0.data[:, 0] = np.clip(
                rho_raw, np.exp(log_rho_min), np.exp(log_rho_max))
        elif metric_choice == "arc-length":
            # Smooth arc-length monitor rho = sqrt(1 + (A*ghat)^2),
            # ghat = |grad field|/g_hi, A = sqrt(ref^4 - 1) so rho = ref^2 at
            # the hi-percentile gradient. Grades continuously from rho=1 in
            # flat regions (no clip kink) -> the smoothest equidistributed meshes.
            A = np.sqrt(max(ref_val ** 4 - 1.0, 0.0))
            ghat = gmag / max(g_hi, 1.0e-30)
            rho_al = np.sqrt(1.0 + (A * ghat) ** 2)
            rho0.data[:, 0] = np.clip(
                rho_al, np.exp(log_rho_min), np.exp(log_rho_max))
        else:
            raise ValueError(
                f"metric_choice must be 'front-following', "
                f"'gradient-uniform', or 'arc-length', got "
                f"{metric_choice!r}")
        return rho0.sym[0]

    if mode == "raw":
        # Raw mode: ρ = |∇field|^power. Skip the percentile
        # clip + (1+amp·t) wrap. Floor to a small positive so
        # zero-gradient regions still get ρ > 0 (mover's geom-
        # mean normaliser doesn't blow up).
        floor = max(1.0e-12,
                    float(np.max(gmag_global)) * 1.0e-6)
        rho0.data[:, 0] = np.maximum(gmag, floor)
        return rho0.sym[0] ** float(power)
    if g_hi <= 1.0e-9:
        rho0.data[:, 0] = 0.0
    else:
        rho0.data[:, 0] = np.clip(
            (gmag - g_lo) / max(g_hi - g_lo, 1.0e-30), 0.0, 1.0)
    return (1.0 + float(amp) * rho0.sym[0]) ** float(power)
