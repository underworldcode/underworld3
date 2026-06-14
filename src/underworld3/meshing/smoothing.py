"""Mesh smoothing utilities.

Currently provides a Winslow-style Jacobi smoother for interior
vertex positions: each interior vertex is moved toward the average
position of its edge neighbours, with boundary vertices held fixed.

Use after a mesh deformation has left some cells highly distorted
(e.g. free-surface evolution that has crushed cells near the
surface). Topology is unchanged — vertex indices, DOFs, and the
parallel partition are all preserved; only coordinates move.

Parallel: a PETSc parallel AIJ matrix represents the vertex-vertex
adjacency. Each rank inserts entries for every edge it sees locally
using GLOBAL vertex indices; ``mat.assemble()`` combines cross-rank
contributions so that owned-vertex rows are complete after assembly.
Without this, UW3's default cell-overlap-0 distribution under-counts
neighbours for vertices on the rank partition boundary, producing
visibly wrong updates along the rank cut.

Two operators:
  - ``metric=None`` (default): the graph-Laplacian Jacobi sweeps
    described above — equalises connectivity, makes cells equant.
  - ``metric=<expr>``: an **elastic-spring network** relaxed
    toward equilibrium (the default metric path). Every edge is a
    spring whose *rest length* is ``∝ ρ_tgt^(-1/dim)`` (finer where
    ``ρ_tgt = metric`` is large), normalised so the mean rest
    length equals the current mean edge length (scale preserved —
    pure redistribution). A position-based Jacobi relaxation moves
    interior nodes toward rest-length-consistent positions; a
    coherent global signed-area backtrack prevents inversion. A
    Lagrangian density (``r0`` set once to the original radius,
    then ``f(r0.sym)``) keeps the rest lengths fixed per material
    point. **Status: under development** — the fixed-topology
    Jacobi relaxation currently reaches only weak grading
    (deep/near ≈ 1.03 for an 8× target) and can stall against the
    tangle guard; a proper equilibrium solve / preconditioning is
    being investigated.

The optimal-transport / Monge–Ampère mesh-potential approach
(``_winslow_elliptic``, preserved, not the default) was
exhaustively investigated 2026-05-16 and found to cap at the same
~1.07 for every variant (linear / recovered-Hessian / convex-branch
BFO / outer composition). That *every* dissimilar method
(graph-Laplacian, weighted-Laplacian, MA-all-variants, elastic
spring) converges to deep/near ≈ 1.03–1.07 while the *exact*
equidistribution at the same fixed topology is ~10× points to a
common missing ingredient (large coherent long-range node
transport is throttled by pinned-boundary + tangle-guard local
relaxation). Open investigation: elastic-spring redistribution as
a *preconditioner* for the MA solve. See ``scripts/ma_*.py`` and
the project memory.

Future extensions (separate PRs):
  - PR B: nicer pinning API (per-boundary explicit lists, callable
    masks)
  - parallel-exact spring forces (cross-rank edge-force assembly,
    mirroring the Jacobi-path adjacency Mat); currently the spring
    path is serial-exact (rank-boundary nodes under-count forces)
"""

import warnings
from typing import Optional, Sequence

import numpy as np

import underworld3 as uw


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


def _auto_pinned_labels(mesh) -> tuple:
    """All non-sentinel geometric boundary labels on the mesh.

    Skips ``All_Boundaries`` / ``Null_Boundary`` (sentinels) and
    known non-geometric pressure-pin markers such as ``Centre`` on
    the Annulus (a single-point marker whose underlying ``DMLabel``
    has an invalid communicator and hard-crashes any
    ``getNumValues`` / ``getValueIS`` / ``view`` call).
    """
    skip = {"All_Boundaries", "Null_Boundary", "Centre"}
    names = []
    for member in mesh.boundaries:
        name = getattr(member, "name", None)
        if name and name not in skip:
            names.append(name)
    return tuple(names)


def _owned_vertex_mask(dm):
    """Local-chart boolean mask: True for owned vertices, False for
    ghosts (leaves of the point StarForest). Used by the parallel
    tests; the smoother itself derives ownership from the global
    section attached to its scalar DM clone.
    """
    pStart, pEnd = dm.getDepthStratum(0)
    n_verts = pEnd - pStart
    is_owned = np.ones(n_verts, dtype=bool)
    sf = dm.getPointSF()
    if sf is None:
        return is_owned
    try:
        _n_roots, leaves, _remote = sf.getGraph()
    except Exception:
        return is_owned
    if leaves is None or len(leaves) == 0:
        return is_owned
    for leaf in leaves:
        if pStart <= leaf < pEnd:
            is_owned[leaf - pStart] = False
    return is_owned


def _pinned_mask(dm, pinned_labels):
    """Local-chart boolean mask: True where the vertex belongs to (or
    is the endpoint of an edge in) any of ``pinned_labels``.

    UW3 mesh generators tag boundaries by EDGE rather than by
    vertex; the vertex stratum sometimes misses 1-2 endpoint
    vertices at the gmsh seam (e.g. θ=0°/180° on the Annulus outer
    rim). Pinning by vertex-stratum-only would leave those
    "seam" vertices free, and the smoother would pull them
    inward. Taking the closure of the tagged edges recovers them.

    Tolerates labels that are present but empty (e.g. the
    ``Centre`` pressure-pin marker on an Annulus, whose underlying
    ``DMLabel`` has no strata and hard-crashes any query)."""
    pStart, pEnd = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)
    n_verts = pEnd - pStart
    is_pinned = np.zeros(n_verts, dtype=bool)
    for lname in pinned_labels:
        label = dm.getLabel(lname)
        if label is None:
            continue
        try:
            if label.getNumValues() == 0:
                continue
            vIS = label.getValueIS()
        except Exception:
            continue
        if vIS is None:
            continue
        for val in vIS.getIndices():
            try:
                iset = label.getStratumIS(int(val))
            except Exception:
                continue
            if iset is None:
                continue
            for idx in iset.getIndices():
                if pStart <= idx < pEnd:
                    # Tagged vertex — pin directly.
                    is_pinned[idx - pStart] = True
                elif eStart <= idx < eEnd:
                    # Tagged edge — pin both endpoint vertices.
                    cone = dm.getCone(idx)
                    for c in cone:
                        if pStart <= c < pEnd:
                            is_pinned[c - pStart] = True
    return is_pinned


def _build_scalar_dm(dm):
    """A clone of the topological DM with a 1-dof-per-vertex local
    section. Used to size the adjacency Mat and to produce the global
    vertex numbering."""
    from petsc4py import PETSc
    chart_start, chart_end = dm.getChart()
    pStart, pEnd = dm.getDepthStratum(0)
    section = PETSc.Section().create(comm=dm.getComm())
    section.setChart(chart_start, chart_end)
    for p in range(chart_start, chart_end):
        section.setDof(p, 1 if pStart <= p < pEnd else 0)
    section.setUp()
    dm_scalar = dm.clone()
    dm_scalar.setLocalSection(section)
    return dm_scalar


def _build_adjacency_matrix(mesh):
    """Build the parallel vertex-vertex adjacency as a PETSc AIJ Mat.

    Each rank inserts entries for every locally-visible edge using
    GLOBAL vertex indices; ``mat.assemble()`` combines cross-rank
    contributions, so that after assembly an owned-vertex row has
    every neighbour it would in a serial run — even when the
    incident edge lives in a cell owned by another rank that is not
    in this rank's overlap.

    Returns
    -------
    A : PETSc.Mat
        Unweighted vertex-vertex adjacency, entries are 1.0 where an
        edge exists. Divide the result of ``A @ x`` by the degree
        vector to get the neighbour average.
    dm_scalar : PETSc.DMPlex
        Clone of ``mesh.dm`` with a 1-dof-per-vertex section. Owns
        the parallel layout for the Mat and any vectors of the same
        shape.
    gsection : PETSc.Section
        Global section of ``dm_scalar`` — the owned-vertex numbering.
    """
    from petsc4py import PETSc
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)

    dm_scalar = _build_scalar_dm(dm)
    gsection = dm_scalar.getGlobalSection()

    def gidx(p):
        off = gsection.getOffset(p)
        return off if off >= 0 else -(off + 1)

    A = dm_scalar.createMatrix()
    A.setOption(A.Option.NEW_NONZERO_LOCATION_ERR, False)
    A.setOption(A.Option.IGNORE_OFF_PROC_ENTRIES, False)

    for e in range(eStart, eEnd):
        cone = dm.getCone(e)
        if len(cone) != 2:
            continue
        v0, v1 = cone[0], cone[1]
        if not (pStart <= v0 < pEnd and pStart <= v1 < pEnd):
            continue
        g0, g1 = gidx(v0), gidx(v1)
        A.setValues([g0], [g1], [1.0], PETSc.InsertMode.INSERT)
        A.setValues([g1], [g0], [1.0], PETSc.InsertMode.INSERT)
    A.assemble()
    return A, dm_scalar, gsection


# Cached spring-smoother topology state keyed by (mesh-id,
# pinned-labels, topology): the edge vertex-index pairs and per-node
# incident-edge degree. Rebuilt automatically on a topology change
# (remesh / adapt / repartition), which produces a new cache key.
_SPRING_CACHE: dict = {}


def _min_incident_edge(dm, coords):
    """Per-vertex minimum incident edge length (local-chart
    v-pStart order). Used as an optional secondary per-node cap on
    the spring step (the primary tangle guard is the coherent global
    signed-area backtrack in ``_winslow_spring``)."""
    pStart, pEnd = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)
    h = np.full(pEnd - pStart, np.inf)
    for e in range(eStart, eEnd):
        cone = dm.getCone(e)
        if len(cone) != 2:
            continue
        v0, v1 = cone[0], cone[1]
        if not (pStart <= v0 < pEnd and pStart <= v1 < pEnd):
            continue
        i0, i1 = v0 - pStart, v1 - pStart
        L = float(np.linalg.norm(coords[i0] - coords[i1]))
        if L < h[i0]:
            h[i0] = L
        if L < h[i1]:
            h[i1] = L
    return h


def _tri_cells(dm):
    """Triangle vertex-index triples (local-chart, v-pStart order).

    Returns an ``(n_tri, 3)`` int array, or ``None`` if the mesh is
    not all-triangle (then the global signed-area backtrack is
    skipped and only the optional per-node edge cap guards against
    tangling).
    """
    cStart, cEnd = dm.getHeightStratum(0)
    pStart, pEnd = dm.getDepthStratum(0)
    tris = []
    for c in range(cStart, cEnd):
        closure = dm.getTransitiveClosure(c)[0]
        vs = [p - pStart for p in closure if pStart <= p < pEnd]
        if len(vs) != 3:
            return None
        tris.append(vs)
    if not tris:
        return None
    return np.asarray(tris, dtype=np.int64)


def _signed_areas(coords, tris):
    """Signed area of each triangle (sign = orientation)."""
    a = coords[tris[:, 0]]
    b = coords[tris[:, 1]]
    c = coords[tris[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


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

    Returns ``{"rms": ..., "max": ..., "median_abs": ...}``
    summarising ``|δ|`` over cells. A mesh already at the
    mover's achievable equidistribution gives ~0; the
    pre-adapted mesh against a strongly-peaked metric gives
    O(1) or larger.

    Cheap: one ``metric`` evaluate at cell centroids + a few
    NumPy reductions. Used by
    :func:`smooth_mesh_interior(skip_threshold=...)` to skip
    adapting when the mesh is already aligned with the target.

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
        Triangle mesh (only 2-D for now).
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
        ``{"rms": float, "max": float, "median_abs": float}``.
    """
    import underworld3 as _uw

    coords = np.asarray(mesh.X.coords)
    tris = _tri_cells(mesh.dm)
    if tris is None:
        raise NotImplementedError(
            "mesh_metric_mismatch: triangle mesh required")
    A_actual = np.abs(_signed_areas(coords, tris))
    centroids = coords[tris].mean(axis=1)
    rho = np.asarray(_uw.function.evaluate(
        metric, centroids)).reshape(-1)
    rho = np.maximum(rho, 1.0e-12)   # guard
    inv_rho = 1.0 / rho
    A_target = A_actual.sum() * inv_rho / inv_rho.sum()
    if resolution_ratio is not None:
        R = float(resolution_ratio)
        A_mean = A_actual.sum() / len(A_actual)
        # Clip target areas to the mover's achievable band
        # [A_mean/R², A_mean·R²] (h in [h0/R, h0·R] ⇒
        # A in [h0²/R², h0²·R²] = [A_mean/R², A_mean·R²]).
        A_target = np.clip(A_target, A_mean / R ** 2,
                           A_mean * R ** 2)
    delta = 0.5 * np.log(A_actual / A_target)
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
    if _uw.mpi.size > 1:
        _ar = lambda v: _uw.mpi.comm.allreduce(float(v))
    else:
        _ar = float
    n_c = _ar(log_density.size)
    sx = _ar(log_density.sum()); sy = _ar(log_rho.sum())
    sxx = _ar((log_density * log_density).sum())
    syy = _ar((log_rho * log_rho).sum())
    sxy = _ar((log_density * log_rho).sum())
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

    return dict(rms=float(np.sqrt(np.mean(delta ** 2))),
                max=float(abs_delta.max()),
                median_abs=float(np.median(abs_delta)),
                alignment=alignment,
                misalignment=misalignment)


def _edge_pairs(dm):
    """``(n_edge, 2)`` int array of edge endpoint vertex indices in
    local-chart (v - pStart) order — the spring network's bars.

    Skips edges whose endpoints are not both in the local vertex
    stratum (rank-ghost incomplete edges); the spring path is
    serial-exact (see module docstring)."""
    pStart, pEnd = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)
    pairs = []
    for e in range(eStart, eEnd):
        cone = dm.getCone(e)
        if len(cone) != 2:
            continue
        v0, v1 = cone[0], cone[1]
        if not (pStart <= v0 < pEnd and pStart <= v1 < pEnd):
            continue
        pairs.append((v0 - pStart, v1 - pStart))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def _winslow_spring(mesh, metric, pinned_labels, verbose,
                    n_sweeps=300, relax=None, step_frac=None,
                    boundary_slip=False, shape_w=1.0, size_w=8.0):
    r"""Metric-driven mesh grading by elastic-spring equilibrium.

    Every mesh edge is a linear spring whose *rest length* is set
    from the target density,

    .. math::

        L^0_{ij} \;\propto\; \rho_{\mathrm{tgt}}^{-1/d},

    scaled once so the total rest length equals the total current
    edge length (overall scale preserved — pure redistribution).
    The interior nodes are moved to the **mechanical equilibrium**
    by *minimising the truss energy*

    .. math::

        E(\mathbf{x}) \;=\; \tfrac12 \sum_{e}
        \big(\,|\mathbf{x}_i-\mathbf{x}_j| - L^0_e\,\big)^2

    over the free (non-pinned) nodes with **nonlinear conjugate
    gradients** (Polak–Ribière⁺) and an Armijo line search whose
    trial step is rejected if any cell would invert. Solving the
    equilibrium — rather than creeping with damped Jacobi sweeps,
    which stall against a per-sweep global tangle freeze — is what
    lets the absolute rest-length target actually grade the mesh
    toward spacing ``∝ ρ_tgt^{-1/d}``.

    ``ρ_tgt`` is Lagrangian (``metric = f(r0)`` with ``r0`` a frozen
    mesh variable), so the rest lengths are fixed per material node
    (computed once) and the *design* grading is restored even after
    the mesh deformed. Uniform ``ρ_tgt`` ⇒ all rest lengths equal
    the mean edge length ⇒ only a benign mild regularisation toward
    uniform spacing (no grading change).

    ``n_sweeps`` caps the CG iterations (CG converges far faster
    than the old Jacobi sweep budget). ``relax`` / ``step_frac`` are
    unused on the equilibrium path (the CG line search controls the
    step and the inversion guard) and are kept only for signature
    stability. ``n_iters`` / ``alpha`` do not apply.
    """
    pinned_labels = tuple(pinned_labels)
    dm = mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    cStart, cEnd = dm.getHeightStratum(0)
    cone_size = dm.getConeSize(cStart) if cEnd > cStart else 0
    n_verts = pEnd - pStart
    key = (id(mesh), pinned_labels,
           n_verts, cEnd - cStart, cone_size)

    cache = _SPRING_CACHE.get(key)
    if cache is None:
        edges = _edge_pairs(dm)
        if edges.shape[0] == 0:
            return
        deg = np.bincount(
            edges.ravel(), minlength=n_verts).astype(np.double)
        deg[deg == 0.0] = 1.0
        _SPRING_CACHE[key] = (edges, deg)
    else:
        edges, deg = cache

    tris = _tri_cells(dm)
    cdim = mesh.cdim
    v0 = edges[:, 0]
    v1 = edges[:, 1]

    coords = np.asarray(mesh.X.coords, dtype=np.double).copy()

    # Boundary tangential slip via the mesh-owned contract
    # (boundary-slip-strategy.md): each slip vertex slides tangentially and
    # snaps back onto its bounding surface (radial ring / plane / facet);
    # non-slip, junction, and degenerate-normal vertices pin. Replaces the
    # per-ring COM radial snap (one node/ring anchored the rotation gauge);
    # the global inversion guard below still blocks a slip node overtaking a
    # neighbour, and tangential θ-drift is a harmless re-parameterisation.
    is_pinned, _project = mesh.boundary_slip(
        boundary_slip, reference_coords=coords, boundary_labels=pinned_labels)

    free = ~is_pinned

    # ===== Volumetric spring network (shape ⟂ size, decoupled) ====
    # EQUAL edge springs (uniform rest length L̄ = current mean
    # edge) are a pure SHAPE regulariser → equant cells, resists
    # the slivers/degeneracy the graded-edge form produced. The
    # SIZE grading lives entirely in a per-CELL area ("volumetric")
    # constraint: each triangle's area is driven to a target
    # A0 ∝ 1/ρ_tgt (scaled so ΣA0 = Σ(initial area) ⇒ total area
    # conserved, pure redistribution). Both energy terms are
    # written as *relative* squared errors so the shape/size
    # weights (shape_w, size_w) are pure dimensionless knobs.
    e_vec = coords[v1] - coords[v0]
    L_cur = np.linalg.norm(e_vec, axis=1)
    sum_L = float(L_cur.sum())
    n_e = float(L_cur.size)
    if uw.mpi.size > 1:
        sum_L = uw.mpi.comm.allreduce(sum_L)
        n_e = uw.mpi.comm.allreduce(n_e)
    Lbar = sum_L / max(n_e, 1.0)          # uniform edge rest length
    L0 = np.full_like(L_cur, Lbar)
    L0_mean = Lbar

    # Per-cell target area from ρ_tgt at the (initial) centroid.
    # Lagrangian metric ⇒ computed ONCE (rides material points).
    if tris is not None:
        ca = coords[tris[:, 0]]
        cb = coords[tris[:, 1]]
        cc = coords[tris[:, 2]]
        cent = (ca + cb + cc) / 3.0
        rho_c = np.asarray(
            uw.function.evaluate(metric, cent)).reshape(-1)
        rho_c = np.maximum(rho_c, 1.0e-30)
        a_init = np.abs(_signed_areas(coords, tris))
        inv = 1.0 / rho_c
        sA = float(a_init.sum())
        sI = float(inv.sum())
        if uw.mpi.size > 1:
            sA = uw.mpi.comm.allreduce(sA)
            sI = uw.mpi.comm.allreduce(sI)
        A0 = (sA / max(sI, 1.0e-30)) * inv     # ΣA0 = Σa_init
        A0 = np.maximum(A0, 1.0e-30)
        ti0, ti1, ti2 = tris[:, 0], tris[:, 1], tris[:, 2]
    else:
        A0 = None

    # ---- Solve the truss EQUILIBRIUM, not Jacobi creep ----------
    # Minimise the spring energy  E(x) = ½ Σ_e (|x_i−x_j| − L0_e)²
    # over the interior nodes (boundary pinned) by nonlinear
    # conjugate gradients (Polak–Ribière⁺) with an Armijo line
    # search whose trial step is REJECTED if any cell would invert
    # — the tangle guard is inside the optimiser, so it converges to
    # the true equilibrium instead of stalling against a per-sweep
    # global freeze (the Jacobi relaxation's failure mode).
    free_idx = np.nonzero(free)[0]
    n_free = int(free_idx.size)
    if n_free == 0:
        mesh._deform_mesh(coords)
        return

    if tris is not None:
        orient = np.sign(np.median(_signed_areas(coords, tris)))
        orient = orient if orient != 0.0 else 1.0

    def _allsum(s):
        if uw.mpi.size > 1:
            return uw.mpi.comm.allreduce(float(s))
        return float(s)

    def _feasible(X):
        if tris is None:
            return True
        amin = float((_signed_areas(X, tris) * orient).min())
        if uw.mpi.size > 1:
            from mpi4py import MPI as _MPI
            amin = uw.mpi.comm.allreduce(amin, op=_MPI.MIN)
        return amin > 0.0

    have_area = (A0 is not None) and (cdim == 2)

    def _tri_signed(X):
        a, b, c = X[ti0], X[ti1], X[ti2]
        return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                      - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def _energy(X):
        ev = X[v1] - X[v0]
        L = np.sqrt((ev * ev).sum(axis=1))
        re = (L - Lbar) / Lbar               # relative edge error
        E = shape_w * _allsum((re * re).sum())
        if have_area:
            area = orient * _tri_signed(X)
            ra = (area - A0) / A0            # relative area error
            E += size_w * _allsum((ra * ra).sum())
        return E

    def _energy_grad(X):
        ev = X[v1] - X[v0]
        L = np.sqrt((ev * ev).sum(axis=1))
        Ls = np.maximum(L, 1.0e-30)
        re = (L - Lbar) / Lbar
        E = shape_w * _allsum((re * re).sum())
        G = np.zeros_like(X)
        # equal-spring shape term: 2·shape_w·re/(Lbar·L)·ev
        ce = (2.0 * shape_w * re / (Lbar * Ls))[:, None]
        np.add.at(G, v1, ce * ev)
        np.add.at(G, v0, -ce * ev)
        if have_area:
            a, b, c = X[ti0], X[ti1], X[ti2]
            S = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                       - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
            area = orient * S
            ra = (area - A0) / A0
            E += size_w * _allsum((ra * ra).sum())
            # ∂(area)/∂· = orient · ∂S/∂· (signed-area vertex grads)
            fac = (2.0 * size_w * ra / A0 * orient)[:, None]
            gA = np.empty_like(a)
            gB = np.empty_like(a)
            gC = np.empty_like(a)
            gA[:, 0] = 0.5 * (b[:, 1] - c[:, 1])
            gA[:, 1] = 0.5 * (c[:, 0] - b[:, 0])
            gB[:, 0] = 0.5 * (c[:, 1] - a[:, 1])
            gB[:, 1] = 0.5 * (a[:, 0] - c[:, 0])
            gC[:, 0] = 0.5 * (a[:, 1] - b[:, 1])
            gC[:, 1] = 0.5 * (b[:, 0] - a[:, 0])
            np.add.at(G, ti0, fac * gA)
            np.add.at(G, ti1, fac * gB)
            np.add.at(G, ti2, fac * gC)
        G[~free] = 0.0
        return E, G

    # Jacobi (diagonal) preconditioner: the truss Hessian is
    # graph-Laplacian-structured (cond ~ (1/h)²), so plain CG crawls
    # for fine meshes. M⁻¹ = diag(1/deg) — the Laplacian diagonal
    # scale, free here since `deg` is already cached — clusters the
    # spectrum and gives the order-of-magnitude convergence speed-up
    # that turns "stuck at ~1.04" into the true graded minimum.
    invdeg = (1.0 / deg)[:, None]

    X = _project(coords.copy())
    E, G = _energy_grad(X)
    g0 = max(_allsum((G * G).sum()) ** 0.5, 1.0e-30)
    r = -G
    s = r * invdeg
    s[~free] = 0.0
    d = s.copy()
    delta_new = _allsum((r * s).sum())
    dmax = max(float(np.linalg.norm(d[free_idx], axis=1).max()),
               1.0e-30)
    if uw.mpi.size > 1:
        from mpi4py import MPI as _MPI
        dmax = uw.mpi.comm.allreduce(dmax, op=_MPI.MAX)
    t0 = 0.5 * L0_mean / dmax
    c_arm = 1.0e-4
    max_iter = int(n_sweeps)
    for it in range(max_iter):
        gnorm = _allsum((G * G).sum()) ** 0.5
        if gnorm <= 1.0e-8 * g0:
            break
        slope = _allsum((G * d).sum())       # = −(r·d)
        if slope >= 0.0:                     # not descent → restart
            d = s.copy()
            slope = _allsum((G * d).sum())
            if slope >= 0.0:
                break
        t = t0
        accepted = False
        for _ls in range(50):
            Xt = X.copy()
            Xt[free_idx] += t * d[free_idx]
            Xt = _project(Xt)                # slip nodes → boundary
            if _feasible(Xt):
                Et = _energy(Xt)
                if Et <= E + c_arm * t * slope:
                    accepted = True
                    break
            t *= 0.5
        if not accepted:
            break                            # at equilibrium / stuck
        Et, Gt = _energy_grad(Xt)
        r_new = -Gt
        s_new = r_new * invdeg
        s_new[~free] = 0.0
        delta_old = delta_new
        delta_mid = _allsum((r_new * s).sum())
        delta_new = _allsum((r_new * s_new).sum())
        beta = max(0.0, (delta_new - delta_mid)
                   / max(delta_old, 1.0e-30))   # preconditioned PR⁺
        X, E, G = Xt, Et, Gt
        d = s_new + beta * d
        s = s_new
        t0 = min(2.0 * t, 100.0 * t0)        # grow but stay sane

        if verbose and (it % 25 == 0 or it == max_iter - 1):
            ev = X[v1] - X[v0]
            L = np.sqrt((ev * ev).sum(axis=1))
            rms = (_allsum(((L - L0) ** 2).sum())
                   / max(_allsum(L0.size), 1.0)) ** 0.5
            uw.pprint(
                f"  spring PCG iter {it+1}/{max_iter}: "
                f"E={E:.4e}  rms(L-L0)/L0="
                f"{rms / max(L0_mean, 1e-30):.3e}  |g|={gnorm:.2e}")

    coords = X
    mesh._deform_mesh(coords)


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
#  an open investigation. Call _winslow_elliptic() directly to use.
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

# Cached anisotropic-mover state keyed by (mesh-id, pinned-labels,
# topology, solver, φ-order, slip): the ∇ρ projector, the
# eigen-clamped metric-tensor field D, and the cdim displacement
# Poisson solvers (all sharing the tensor operator _c = D). Rebuilt
# on a topology change (a new key).
_ANISO_CACHE: dict = {}

# Cached state for the OT-improvement-step path (one weighted-
# Poisson per call). Keyed like the other movers; same lifetime.
_OT_CACHE: dict = {}

# Per-(mesh,config) running state for the equidistribution
# normaliser's temporal damping: the EMA of ln G carried across
# adaptation events (same key as _ANISO_CACHE). Empty ⇒ first
# event seeds it. Only touched in the resolution_ratio>1 regime.
_GEMA_STATE: dict = {}


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
    ``_winslow_elliptic`` call** and reuse it for every inner solve
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
    patch = np.zeros(n_verts, dtype=np.double)
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


def _winslow_elliptic(mesh, metric, pinned_labels, verbose,
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
        if linear_solver == "gamg":
            def _wire(s, singular=False, elliptic=True):
                _use_iterative_solver(s, singular, elliptic)
        else:
            def _wire(s, singular=False, elliptic=True):
                _use_direct_solver(s, singular, elliptic)
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
            patch = np.ones(n_verts, dtype=np.double)
        _va = vol_field.array
        _va[...] = patch.reshape(_va.shape)

        rho_t = np.asarray(
            uw.function.evaluate(metric, old_coords)).reshape(-1)
        b = rho_t * patch
        inv_sqrt_b_mean = float(np.mean(1.0 / np.sqrt(b)))
        if uw.mpi.size > 1:
            inv_sqrt_b_mean = uw.mpi.comm.allreduce(
                inv_sqrt_b_mean) / uw.mpi.size
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

        # The GAMG path warm-starts the Krylov solve from the previous
        # Picard φ (it changes slowly under ω-relaxation) → a handful
        # of CG iters on the once-built hierarchy. The exact direct
        # path is indifferent to the initial guess.
        # Under MPI, ``linear_solver="direct"`` silently falls back to
        # the iterative path inside ``_use_direct_solver`` (the
        # MUMPS-heap-corruption guard at smoothing.py:914); honour the
        # warm-start in that case too — otherwise the parallel mover
        # pays extra Krylov iterations per Picard step for nothing.
        _zig = not (linear_solver == "gamg"
                    or (linear_solver == "direct" and uw.mpi.size > 1))
        prev_change = None
        # If target-side ρ is on, gradphi needs to be tracking the
        # current φ inside the Picard loop (it's used by ps.f via
        # the X→X+gradphi substitution). Initialise to zero so the
        # first ps.solve sees ρ at source (= identity map estimate).
        if target_side_rho:
            gradphi.array[...] = 0.0
        for it in range(n_picard):
            phi_prev = np.asarray(phi.array).copy()
            ps.solve(zero_init_guess=_zig)
            phi.array[...] = ((1.0 - omega) * phi_prev
                              + omega * np.asarray(phi.array))
            hsolver.solve()
            if target_side_rho:
                gproj.solve()   # update target-side ρ for next iter
            change = float(np.abs(
                np.asarray(phi.array) - phi_prev).max())
            if uw.mpi.size > 1:
                from mpi4py import MPI as _MPI
                change = uw.mpi.comm.allreduce(
                    change, op=_MPI.MAX)
            if prev_change is not None and change < 1.0e-6:
                break
            prev_change = change

        if not target_side_rho:
            gproj.solve()
        disp = np.asarray(
            uw.function.evaluate(gradphi.sym, old_coords)
        ).reshape(old_coords.shape)

        # Directional move-weighting (approach (2), opt-in): the
        # annulus node budget is anisotropic — radial is scarce and
        # pinned, tangential is abundant and free ("spare" angular
        # nodes). A scalar equidistribution is isotropic and cannot
        # express "prefer tangential"; here we rescale the realised
        # displacement in the local radial/tangential frame
        # (move_anisotropy=(w_r, w_θ)) so the same metric is met
        # mostly by sliding nodes around rather than crushing
        # radially. This is the BFO-consistent lightweight version
        # (the φ-Poisson operator / BFO branch algebra is untouched
        # — only the move is reweighted). Centre = mesh centroid
        # (origin for a centred annulus). Default None ⇒ unchanged.
        if move_anisotropy is not None and cdim == 2:
            w_r, w_t = (float(move_anisotropy[0]),
                        float(move_anisotropy[1]))
            ctr = old_coords.mean(axis=0)
            rv = old_coords - ctr
            rn = np.linalg.norm(rv, axis=1)
            ok = rn > 1.0e-30
            rhat = np.zeros_like(rv)
            rhat[ok] = rv[ok] / rn[ok, None]
            that = np.stack([-rhat[:, 1], rhat[:, 0]], axis=1)
            d_r = (disp * rhat).sum(axis=1)
            d_t = (disp * that).sum(axis=1)
            disp = (w_r * d_r[:, None] * rhat
                    + w_t * d_t[:, None] * that)

        step = relax * disp
        if step_frac is not None and np.isfinite(step_frac):
            h = _min_incident_edge(dm, old_coords)
            mag = np.linalg.norm(step, axis=1)
            cap = step_frac * h
            clip = np.isfinite(cap) & (mag > cap) & (mag > 0.0)
            sc = np.ones_like(mag)
            sc[clip] = cap[clip] / mag[clip]
            step = step * sc[:, None]

        free = ~is_pinned
        scale = 1.0
        new_coords = old_coords.copy()
        if tris is not None:
            a0 = _signed_areas(old_coords, tris)
            orient = np.sign(np.median(a0)) or 1.0
            for _bt in range(10):
                trial = old_coords.copy()
                trial[free] += scale * step[free]
                trial = _project(trial)      # slip → ring (∥ only)
                a1min = float(
                    (_signed_areas(trial, tris) * orient).min())
                if uw.mpi.size > 1:
                    from mpi4py import MPI as _MPI
                    a1min = uw.mpi.comm.allreduce(
                        a1min, op=_MPI.MIN)
                if a1min > 0.0:
                    new_coords = trial
                    break
                scale *= 0.5
            else:
                scale = 0.0
                new_coords = old_coords.copy()
        else:
            new_coords[free] += step[free]
            new_coords = _project(new_coords)

        mesh._deform_mesh(new_coords)

        d = float(np.linalg.norm(
            new_coords - old_coords, axis=1).max())
        if uw.mpi.size > 1:
            d = uw.mpi.comm.allreduce(d ** 2) ** 0.5
        if verbose:
            uw.pprint(
                f"  equidistribute MA outer {outer+1}/{n_outer}: "
                f"c={c:.4f}  scale={scale:.3f}  max|Δx|={d:.3e}")
        if d < outer_tol:
            break


def _winslow_equidistribute(mesh, metric, pinned_labels, verbose,
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

    Differences from ``_winslow_elliptic`` (the convex-branch
    BFO Picard):

    * Linear: one weighted-Poisson per outer iter, no inner
      Picard, no Hessian recovery, no convex-branch radical.
    * The source uses the *current* mesh's patch volumes; the
      formulation is identically zero at equidistribution, so
      iterations are self-stabilising (no over-correction).
    * ρ at the current node positions (no source-vs-target
      asymmetry; the iteration is on the current mesh, ρ is at
      its physical position).

    Parameters mirror ``_winslow_elliptic`` where they apply.
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
            "_winslow_equidistribute: 2D meshes only for now.")

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
        if linear_solver == "gamg":
            def _wire(s, singular=False, elliptic=True):
                _use_iterative_solver(s, singular, elliptic)
        else:
            def _wire(s, singular=False, elliptic=True):
                _use_direct_solver(s, singular, elliptic)
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

    # See _winslow_elliptic for the rationale on this combined check —
    # ``linear_solver="direct"`` silently routes to the iterative path
    # under MPI, so honour the warm-start there too.
    _zig = not (linear_solver == "gamg"
                or (linear_solver == "direct" and uw.mpi.size > 1))

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
            patch = np.ones(n_verts, dtype=np.double)
        else:
            patch = _patch_volumes(tris, old_coords, n_verts, vol_field)
        # Normalise so the mean over the domain is the cell mean.
        patch_mean = float(np.mean(patch))
        if uw.mpi.size > 1:
            patch_mean = uw.mpi.comm.allreduce(patch_mean) / uw.mpi.size
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
        if uw.mpi.size > 1:
            from mpi4py import MPI as _MPI
            wnum = uw.mpi.comm.allreduce(wnum, op=_MPI.SUM)
            wden = uw.mpi.comm.allreduce(wden, op=_MPI.SUM)
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
        ps.solve(zero_init_guess=_zig)
        gproj.solve()
        disp = np.asarray(uw.function.evaluate(
            gradphi.sym, old_coords)
        ).reshape(old_coords.shape)

        step = float(relax) * disp

        # Per-vertex displacement cap: |step_i| ≤ step_frac · h_i,
        # where h_i is the shortest edge incident on vertex i.
        # This prevents the OT step from creating LOCAL cell folds
        # near features (where the source is sharp) without killing
        # the global motion (the way the global signed-area
        # backtrack does).
        if step_frac is not None and np.isfinite(step_frac):
            h = _min_incident_edge(dm, old_coords)
            mag = np.linalg.norm(step, axis=1)
            cap = float(step_frac) * h
            clip = np.isfinite(cap) & (mag > cap) & (mag > 0.0)
            sc = np.ones_like(mag)
            sc[clip] = cap[clip] / mag[clip]
            step = step * sc[:, None]

        # --- coherent global signed-area backtrack -------------
        free = ~is_pinned
        scale = 1.0
        new_coords = old_coords.copy()
        if tris is not None:
            a0 = _signed_areas(old_coords, tris)
            orient = np.sign(np.median(a0)) or 1.0
            for _bt in range(10):
                trial = old_coords.copy()
                trial[free] += scale * step[free]
                trial = _project(trial)
                a1min = float(
                    (_signed_areas(trial, tris) * orient).min())
                if uw.mpi.size > 1:
                    from mpi4py import MPI as _MPI
                    a1min = uw.mpi.comm.allreduce(
                        a1min, op=_MPI.MIN)
                if a1min > 0.0:
                    new_coords = trial
                    break
                scale *= 0.5
            else:
                scale = 0.0
                new_coords = old_coords.copy()
        else:
            new_coords[free] += step[free]
            new_coords = _project(new_coords)

        mesh._deform_mesh(new_coords)

        d = float(np.linalg.norm(
            new_coords - old_coords, axis=1).max())
        if uw.mpi.size > 1:
            d = uw.mpi.comm.allreduce(d ** 2) ** 0.5

        # Per-iter "imbalance" diagnostic — std of log(V·ρ/K).
        imb = float(np.std(np.log(Vrho_pos) - ln_K))
        if uw.mpi.size > 1:
            from mpi4py import MPI as _MPI
            imb_sq = uw.mpi.comm.allreduce(imb * imb, op=_MPI.SUM)
            cnt = uw.mpi.comm.allreduce(int(Vrho_pos.size),
                                         op=_MPI.SUM)
            imb = (imb_sq / max(cnt, 1)) ** 0.5

        if verbose:
            uw.pprint(
                f"  OT-improve outer {outer+1}/{n_outer}: "
                f"K={K_val:.4f}  imb={imb:.3e}  "
                f"scale={scale:.3f}  max|Δx|={d:.3e}")
        if d < outer_tol:
            break


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

    The settled scalar equidistribution paths (``_winslow_spring``,
    ``_winslow_elliptic``) cannot do coherent *anisotropic* bulk
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
    ``_winslow_elliptic``). Homogeneous Dirichlet ``u=0`` on the
    pinned boundary makes the per-component operator non-singular —
    no ``constant_nullspace``, side-stepping the GAMG-pure-Neumann
    fragility entirely (``boundary_slip=True`` falls back to the
    pure-Neumann + ring-projection treatment of
    ``_winslow_elliptic``). ``n_outer`` composes the map (re-project
    ``∇ρ`` / rebuild ``D`` on the moved mesh — the standard MMPDE
    outer iteration). Reuses ``_winslow_elliptic``'s coherent global
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

    Parameters mirror ``_winslow_elliptic`` where shared.

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
        if linear_solver == "gamg":
            def _wire(s, singular=False, elliptic=True):
                _use_iterative_solver(s, singular, elliptic)
        else:
            def _wire(s, singular=False, elliptic=True):
                _use_direct_solver(s, singular, elliptic)

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
        # _winslow_elliptic slip treatment). Default (pinned) ⇒
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

    # See _winslow_elliptic for rationale — ``linear_solver="direct"``
    # silently falls back to the iterative path under MPI, so honour
    # the warm-start there too.
    _zig = not (linear_solver == "gamg"
                or (linear_solver == "direct" and uw.mpi.size > 1))

    # ---- build the eigen-clamped metric tensor field D ONCE ------
    # on the *undeformed* mesh (the design metric), then hold it
    # fixed and Lagrangian (the field rides material points through
    # _deform_mesh, exactly as _winslow_spring computes its
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
    gproj.solve()
    Dcoords = np.asarray(Df.coords)
    gvec = np.asarray(
        uw.function.evaluate(grho.sym, Dcoords)).reshape(-1, cdim)
    # h0 = undeformed mean edge length. If the caller passes
    # `h0_override` (e.g. a value cached at the FIRST adapt on
    # this mesh), use that — re-measuring from a deformed mesh
    # makes h0 shrink as the mesh refines, which then shifts
    # the eigenvalue clamps tighter and tighter and compounds
    # refinement across repeated adapt cycles.
    if h0_override is not None:
        h0 = float(h0_override)
    else:
        ep = _edge_pairs(dm)
        if ep.shape[0]:
            h0 = float(np.linalg.norm(
                old0[ep[:, 1]] - old0[ep[:, 0]], axis=1).mean())
        else:
            h0 = 1.0
        if uw.mpi.size > 1:
            h0 = uw.mpi.comm.allreduce(h0) / uw.mpi.size
    gn = np.linalg.norm(gvec, axis=1)
    gmax = float(gn.max()) if gn.size else 0.0
    if uw.mpi.size > 1:
        from mpi4py import MPI as _MPI
        gmax = uw.mpi.comm.allreduce(gmax, op=_MPI.MAX)
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
    gref = gmax if gmax > g_eps else 1.0
    base = 1.0 / h0 ** 2

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
        gmax = float(gn.max()) if gn.size else 0.0
        if uw.mpi.size > 1:
            from mpi4py import MPI as _MPI
            gmax = uw.mpi.comm.allreduce(gmax, op=_MPI.MAX)
        gref = gmax if gmax > g_eps else 1.0
        # Density branches (same as legacy code path)
        if resolution_ratio > 1.0:
            R_ = float(resolution_ratio)
            rho_v_ = np.asarray(
                uw.function.evaluate(metric, Dcoords)
            ).reshape(-1)
            s_log_ = np.log(np.clip(rho_v_, 1.0e-12, None))
            if uw.mpi.size > 1:
                from mpi4py import MPI as _MPI
                tot = uw.mpi.comm.allreduce(
                    float(s_log_.sum()), op=_MPI.SUM)
                cnt = uw.mpi.comm.allreduce(
                    int(s_log_.size), op=_MPI.SUM)
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
            if uw.mpi.size > 1:
                from mpi4py import MPI as _MPI
                r_lo_ = uw.mpi.comm.allreduce(r_lo_, op=_MPI.MIN)
                r_hi_ = uw.mpi.comm.allreduce(r_hi_, op=_MPI.MAX)
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

    if resolution_ratio > 1.0:
        R = float(resolution_ratio)
        rho_v = np.asarray(
            uw.function.evaluate(metric, Dcoords)).reshape(-1)
        s_log = np.log(np.clip(rho_v, 1.0e-12, None))
        if uw.mpi.size > 1:
            from mpi4py import MPI as _MPI
            tot = uw.mpi.comm.allreduce(float(s_log.sum()),
                                        op=_MPI.SUM)
            cnt = uw.mpi.comm.allreduce(int(s_log.size),
                                        op=_MPI.SUM)
            ln_g = tot / max(cnt, 1)
        else:
            ln_g = float(s_log.mean())
        # --- temporal damping of the normaliser G (EMA in log
        # space) -------------------------------------------------
        # G is recomputed from the *instantaneous* field every
        # adaptation event; during a violent transient that lurches
        # the whole ρ/G distribution sideways across the *fixed*
        # eigen-clamp band → mass clamp-saturation → the mesh
        # visibly "wobbles". Low-pass ln G across events (G is a
        # geometric quantity ⇒ average in log space) so the band
        # stays centred. This smooths **only the one global
        # intensity scalar** — the spatial ρ(x) pattern still
        # tracks the current field every event, so *where* it
        # refines stays fully responsive. a=geom_mean_smoothing:
        # a≥1 ⇒ no damping (instantaneous, the original behaviour);
        # 0<a<1 ⇒ EMA, lnG_eff = a·lnG_now + (1−a)·lnG_prev (a≈0.25
        # strong); the first event seeds the state (no history yet).
        # Carried in _GEMA_STATE under the _ANISO_CACHE key so it
        # persists across adaptation events but is per-run/per-mesh.
        a = float(geom_mean_smoothing)
        if 0.0 < a < 1.0:
            prev = _GEMA_STATE.get(key)
            if prev is not None:
                ln_g = a * ln_g + (1.0 - a) * prev
            _GEMA_STATE[key] = ln_g
        # ρ̂ = ρ/G (geometric mean 1 ⇒ ⟨ln ρ̂⟩=0, budget-centred);
        # iso = base·ρ̂ → refine where ρ̂>1, coarsen where ρ̂<1,
        # the two complementary by construction (no coarsen knob).
        iso = base * np.exp(s_log - ln_g)
        lam_lo = base / R ** 2
        lam_hi = base * R ** 2
        # Anisotropic-bump magnitude. Default: ride the local
        # density (M = iso·(I+β·bump) — the clean scale-invariant
        # form). aniso_to_base=True keys it to constant `base`
        # instead (M = iso·I + base·β·bump), matching the legacy
        # cc=2 regime that produced a markedly solver-friendlier
        # mesh: it stops a coarsened-near-front cell from being
        # large AND strongly stretched (the clustered poor cells
        # the equidist form makes during a violent transient).
        aniso_keyed = (np.full(Dcoords.shape[0], base)
                       if aniso_to_base else iso)
    elif coarsen_cap > 1.0:
        rho_v = np.asarray(
            uw.function.evaluate(metric, Dcoords)).reshape(-1)
        r_lo = float(np.percentile(rho_v, 10.0))
        r_hi = float(np.percentile(rho_v, 90.0))
        if uw.mpi.size > 1:
            from mpi4py import MPI as _MPI
            r_lo = uw.mpi.comm.allreduce(r_lo, op=_MPI.MIN)
            r_hi = uw.mpi.comm.allreduce(r_hi, op=_MPI.MAX)
        q = np.clip((rho_v - r_lo) / max(r_hi - r_lo, 1.0e-30),
                    0.0, 1.0)
        iso = base * float(coarsen_cap) ** (q - 1.0)   # q=1 → base
        lam_lo = base / float(coarsen_cap)             # coarsest
        lam_hi = 1.0 / (h0 / np.sqrt(aniso_cap)) ** 2  # finest
        aniso_keyed = np.full(Dcoords.shape[0], base)
    else:
        iso = np.full(Dcoords.shape[0], base)
        lam_lo = base                                  # coarsest
        lam_hi = 1.0 / (h0 / np.sqrt(aniso_cap)) ** 2  # finest
        aniso_keyed = np.full(Dcoords.shape[0], base)

    Dout = np.empty((Dcoords.shape[0], 2, 2))
    eye2 = np.eye(2)
    for i in range(Dcoords.shape[0]):
        g = gvec[i]
        gni = gn[i]
        bi = iso[i]
        ai = aniso_keyed[i]
        if gni > g_eps and gmax > g_eps:
            gh = g / gni
            # iso·I (equidistribution density) + anisotropic bump
            # (regime 1: keyed to local iso ⇒ the whole metric is
            # one scale-invariant density·shape field, clamp = rail;
            # regimes 2/3: keyed to base ⇒ aniso_cap/beta retain
            # their exact validated meaning).
            M = bi * eye2 + ai * beta * (gni / gref) ** 2 \
                * np.outer(gh, gh)
        else:
            M = bi * eye2
        w, V = np.linalg.eigh(M)
        w = np.clip(w, lam_lo, lam_hi)
        if metric_role == "Minv":
            w = 1.0 / w
        Dout[i] = (V * w) @ V.T
    Df.array[:, 0, 0] = Dout[:, 0, 0]
    Df.array[:, 0, 1] = Dout[:, 0, 1]
    Df.array[:, 1, 0] = Dout[:, 1, 0]
    Df.array[:, 1, 1] = Dout[:, 1, 1]

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
            usolvers[c].solve(zero_init_guess=_zig)
            disp[:, c] = np.asarray(
                uw.function.evaluate(ufields[c].sym, old_coords)
            ).reshape(-1)

        # Directional move-weighting (opt-in; same frame + default
        # None ⇒ unchanged as _winslow_elliptic).
        if move_anisotropy is not None and cdim == 2:
            w_r, w_t = (float(move_anisotropy[0]),
                        float(move_anisotropy[1]))
            ctr = old_coords.mean(axis=0)
            rv = old_coords - ctr
            rn = np.linalg.norm(rv, axis=1)
            ok = rn > 1.0e-30
            rhat = np.zeros_like(rv)
            rhat[ok] = rv[ok] / rn[ok, None]
            that = np.stack([-rhat[:, 1], rhat[:, 0]], axis=1)
            d_r = (disp * rhat).sum(axis=1)
            d_t = (disp * that).sum(axis=1)
            disp = (w_r * d_r[:, None] * rhat
                    + w_t * d_t[:, None] * that)

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
        # is the exact analogue of _winslow_elliptic's picard_relax
        # (the BFO path needs ω≈0.4 or its Hessian grows unbounded).
        step = float(relax) * disp

        # --- coherent global signed-area backtrack + slip + move --
        free = ~is_pinned
        scale = 1.0
        new_coords = old_coords.copy()
        if tris is not None:
            a0 = _signed_areas(old_coords, tris)
            orient = np.sign(np.median(a0)) or 1.0
            # Minimum acceptable cell area for the backtrack. The
            # original test (`a1min > 0`) only catches *flipped*
            # cells; near-degenerate cells with three near-collinear
            # vertices pass it but produce invisible sliver
            # triangles. Require min area > a fixed fraction of
            # the **undeformed-mesh** median cell area
            # (`_a0_undeformed_med`, captured before the iteration
            # loop). A refinement of 3 in 2D legitimately shrinks
            # cells by 3²=9× in area, so a floor at 1% of the
            # undeformed median rejects degenerate slivers (which
            # are 1000× smaller) without rejecting legitimate
            # refinement.
            a_min_floor = 0.01 * _a0_undeformed_med
            for _bt in range(10):
                trial = old_coords.copy()
                trial[free] += scale * step[free]
                trial = _project(trial)
                a_signed = _signed_areas(trial, tris) * orient
                a1min = float(a_signed.min())
                if uw.mpi.size > 1:
                    from mpi4py import MPI as _MPI
                    a1min = uw.mpi.comm.allreduce(
                        a1min, op=_MPI.MIN)
                # Accept only if no cell flipped AND no cell
                # collapsed below the area floor.
                if a1min > a_min_floor:
                    new_coords = trial
                    break
                scale *= 0.5
            else:
                scale = 0.0
                new_coords = old_coords.copy()
        else:
            new_coords[free] += step[free]
            new_coords = _project(new_coords)

        mesh._deform_mesh(new_coords)

        d = float(np.linalg.norm(
            new_coords - old_coords, axis=1).max())
        if uw.mpi.size > 1:
            d = uw.mpi.comm.allreduce(d ** 2) ** 0.5
        if verbose:
            uw.pprint(
                f"  anisotropic mover outer {outer+1}/{n_outer}: "
                f"h0={h0:.3e}  scale={scale:.3f}  "
                f"max|Δx|={d:.3e}")
        if d < outer_tol:
            break


def _build_local_to_owned_map(dm, gsection, vec):
    """Compute, for each local owned vertex, its position in the
    rank's slice of the global Vec.

    Returns (owned_local_indices, owned_vec_positions, is_owned_local)
    where:
      * owned_local_indices : local-chart indices of owned vertices
        (shape n_owned, dtype int64)
      * owned_vec_positions : positions in vec.array (same shape)
      * is_owned_local : bool mask over the local chart
    """
    pStart, pEnd = dm.getDepthStratum(0)
    n_local = pEnd - pStart
    rstart, rend = vec.getOwnershipRange()
    is_owned = np.zeros(n_local, dtype=bool)
    owned_local = []
    owned_vec_pos = []
    for v in range(pStart, pEnd):
        off = gsection.getOffset(v)
        if off < 0:
            continue  # ghost
        is_owned[v - pStart] = True
        owned_local.append(v - pStart)
        owned_vec_pos.append(off - rstart)
    return (np.asarray(owned_local, dtype=np.int64),
            np.asarray(owned_vec_pos, dtype=np.int64),
            is_owned)


def smooth_surface_field(
    field,
    n_iters: int = 10,
    alpha: float = 0.5,
    taubin: bool = True,
    mu: Optional[float] = None,
    passband: float = 0.1,
    pinned_labels: Optional[Sequence[str]] = None,
):
    r"""Low-pass a scalar field over a (surface) mesh's vertex-edge graph.

    The *field* analogue of the coordinate graph-Laplacian Jacobi path in
    :func:`smooth_mesh_interior`. Each sweep blends every vertex value toward
    the mean of its edge-neighbours,

    .. math::

        h_i \leftarrow h_i + f\,\Big( \tfrac{1}{|N(i)|}\sum_{j\in N(i)} h_j
                                       - h_i \Big),

    attenuating high-wavenumber (facet-scale / sawtooth) content while leaving
    the smooth, long-wavelength field. Plain Laplacian smoothing
    (``taubin=False``) also damps the smooth part and shrinks amplitudes; the
    **Taubin** :math:`\lambda\,|\,\mu` scheme — a positive blend ``alpha``
    followed by a negative back-step ``mu`` each iteration — is a near-flat
    passband low-pass that leaves the mean and the long-wavelength amplitude
    essentially unchanged (the discrete analogue of a volume-preserving
    filter).

    Designed for the free-surface height field carried on a codim-1 surface
    submesh (:meth:`Mesh.extract_surface`): the graph operator needs only the
    edge connectivity, so it works on a 1-manifold loop where an FE solve is
    not yet available. No global gather, no FFT — unlike a spectral
    (Fourier ``h(θ)``) low-pass it is purely local on the surface graph.

    Parameters
    ----------
    field : degree-1 scalar ``MeshVariable``
        Smoothed **in place** (its nodal values are overwritten).
    n_iters : int, default 10
        Number of Taubin (or plain-Laplacian) iterations.
    alpha : float, default 0.5
        Positive blend factor :math:`\lambda \in (0, 1]`.
    taubin : bool, default True
        Apply the volume-preserving :math:`\lambda\,|\,\mu` pair each
        iteration. ``False`` ⇒ plain Laplacian smoothing (shrinks).
    mu : float, optional
        Negative back-step factor. Default derived from ``passband`` as
        :math:`\mu = 1/(k_{pb} - 1/\lambda)` (the standard Taubin choice;
        gives :math:`|\mu| \gtrsim \lambda`). Ignored when ``taubin`` is False.
    passband : float, default 0.1
        Taubin passband wavenumber :math:`k_{pb}` used to derive ``mu``.
    pinned_labels : sequence of str, optional
        Boundary labels whose vertices are held fixed (e.g. the endpoints of
        an open surface arc). A closed loop (annulus / shell ``Upper``) has
        none, so the default ``None`` is correct there.

    Notes
    -----
    Serial-correct. The parallel path is a drop-in: assemble the
    vertex-vertex adjacency with :func:`_build_adjacency_matrix` (already
    parallel-correct, bit-identical serial/parallel) and the owned-vertex map
    with :func:`_build_local_to_owned_map`, then run the same blend via
    ``A.mult`` on the global Vec and ``globalToLocal`` the result back. The
    serial path here exercises the algorithm on the (serial) free-surface
    testbed; the parallel wiring is a contained follow-up.
    """
    dm = field.mesh.dm
    pStart, pEnd = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)
    nv = pEnd - pStart

    # Vertex-edge graph from the edge cones (depth-1 points). On a
    # 1-manifold the edges ARE the top cells; on a 2-manifold they are the
    # facet edges — getDepthStratum(1) is the edge stratum either way.
    e0 = []
    e1 = []
    for e in range(eStart, eEnd):
        cone = dm.getCone(e)
        if len(cone) != 2:
            continue
        v0, v1 = cone[0] - pStart, cone[1] - pStart
        if 0 <= v0 < nv and 0 <= v1 < nv:
            e0.append(v0)
            e1.append(v1)
    e0 = np.asarray(e0, dtype=np.int64)
    e1 = np.asarray(e1, dtype=np.int64)

    deg = np.zeros(nv)
    np.add.at(deg, e0, 1.0)
    np.add.at(deg, e1, 1.0)
    deg_safe = np.maximum(deg, 1.0)

    pinned = None
    if pinned_labels:
        pinned = _pinned_mask(dm, tuple(pinned_labels))

    if taubin and mu is None:
        # Standard Taubin: 1/λ + 1/μ = k_pb  ⇒  μ = 1/(k_pb − 1/λ) < 0.
        mu = 1.0 / (passband - 1.0 / alpha)

    # field.data is the local degree-1 array, ordered by vertex
    # (data[i] ↔ vertex pStart+i ↔ mesh.X.coords[i]).
    x = np.asarray(field.data[:, 0], dtype=float).copy()

    def _blend(x, f):
        s = np.zeros(nv)
        np.add.at(s, e0, x[e1])
        np.add.at(s, e1, x[e0])
        avg = np.where(deg > 0, s / deg_safe, x)
        xn = x + f * (avg - x)
        if pinned is not None:
            xn[pinned] = x[pinned]
        return xn

    for _ in range(n_iters):
        x = _blend(x, alpha)
        if taubin:
            x = _blend(x, mu)

    field.data[:, 0] = x
    return field


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
    method : {"spring", "ma"}, default "spring"
        Metric-grading solver (ignored when ``metric is None``):

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



# ===== grafted from feature/elliptic-ma: mmpde mover + helpers =====
def _tet_cells(dm):
    """Tetrahedron vertex-index quadruples (local-chart), or ``None`` if the
    mesh is not all-tet. The 3D analogue of :func:`_tri_cells` — used for the
    signed-volume backtrack of the equidistribution mover in 3D."""
    cStart, cEnd = dm.getHeightStratum(0)
    pStart, pEnd = dm.getDepthStratum(0)
    tets = []
    for c in range(cStart, cEnd):
        closure = dm.getTransitiveClosure(c)[0]
        vs = [p - pStart for p in closure if pStart <= p < pEnd]
        if len(vs) != 4:
            return None
        tets.append(vs)
    if not tets:
        return None
    return np.asarray(tets, dtype=np.int64)


def _owned_cell_mask(dm):
    """Local-chart boolean mask over cells (height stratum 0): True for
    owned cells, False for ghost/overlap cells (leaves of the point SF).
    Indexed like ``_tri_cells`` / ``_signed_areas`` (cell i ↔ point
    cStart+i). Assembly must sum over OWNED cells only so that a
    ``localToGlobal(ADD_VALUES)`` ghost reduction does not double-count
    overlap cells.
    """
    cStart, cEnd = dm.getHeightStratum(0)
    is_owned = np.ones(cEnd - cStart, dtype=bool)
    sf = dm.getPointSF()
    if sf is None:
        return is_owned
    try:
        _n_roots, leaves, _remote = sf.getGraph()
    except Exception:
        return is_owned
    if leaves is None or len(leaves) == 0:
        return is_owned
    for leaf in leaves:
        if cStart <= leaf < cEnd:
            is_owned[leaf - cStart] = False
    return is_owned


def _min_incident_edge_nd(cells, coords):
    """Dimension-general shortest-incident-edge per vertex. ``cells`` is
    (n_cells, d+1); returns (n_verts,). Used by the MMPDE per-node step
    cap. (The 2D-only ``_min_incident_edge`` reads the DM directly; this
    works for tets too and takes an explicit cell array so the caller can
    restrict the stencil.)"""
    n_verts = coords.shape[0]
    ncorner = cells.shape[1]
    v = np.full(n_verts, np.inf)
    for a in range(ncorner):
        for b in range(a + 1, ncorner):
            e = np.linalg.norm(coords[cells[:, a]] - coords[cells[:, b]],
                               axis=1)
            np.minimum.at(v, cells[:, a], e)
            np.minimum.at(v, cells[:, b], e)
    return v


def _winslow_mmpde(mesh, metric, pinned_labels, verbose,
                   n_outer=150, p=1.5, theta=1.0 / 3.0, tau=1.0,
                   step_frac=0.2, area_floor_frac=0.01,
                   boundary_slip=False, outer_tol=1.0e-7, tol=1.0e-3,
                   stol=None, stol_k=3,
                   fd_eps=1.0e-6, metric_eval="rbf", rbf_k=None,
                   accel="cg", momentum=0.0,
                   **_ignored):
    r"""Anisotropic variational moving-mesh adaptation (Huang–Kamenski
    MMPDE; the direct simplex discretization of JCP 301 (2015) 322,
    arXiv:1410.7872). Dimension-general (`d = 2, 3`) and parallel-safe.

    Generates the physical mesh as the image of a **fixed computational
    (reference) mesh** under the inverse coordinate map, minimizing
    Huang's functional ``G = theta*sqrt(detM)*S**q + (1-2theta)*d**q *
    r**p * detM**((1-p)/2)`` with ``q = d*p/2``, ``S = tr(J Minv J^T)``,
    ``J = Ehat @ inv(E)``, ``r = det J``.
    Because `G → ∞` as `det𝕁 → 0` the map is non-folding (Math. Comp. 87
    (2018) 1887); because it is the inverse map of a convex computational
    domain it genuinely *clusters and aligns* to `M` — a thin strip on a
    fault, not the isotropic centre-of-gravity blob the scalar MA mover
    produces, and not the non-clustering smooth of the decoupled
    `_winslow_anisotropic`. See
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
    """
    import sympy
    from petsc4py import PETSc
    pinned_labels = tuple(pinned_labels)
    cdim = mesh.cdim
    if cdim not in (2, 3):
        raise NotImplementedError(
            "_winslow_mmpde supports 2D/3D simplex meshes only")
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
            f"_winslow_mmpde metric must be {cdim}x{cdim} (or a scalar "
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
    if cdim == 2:
        cells_all = _tri_cells(dm)
        signed_vol = _signed_areas
    else:
        cells_all = _tet_cells(dm)
        signed_vol = _signed_volumes
    if cells_all is None:
        return
    fact = 2.0 if cdim == 2 else 6.0           # d! → |K| = |detE|/d!
    owned_cell = _owned_cell_mask(dm)
    cells_own = cells_all[owned_cell]
    is_owned_v = _owned_vertex_mask(dm)

    coord_dm = dm.getCoordinateDM()
    local_vec = dm.getCoordinatesLocal()
    global_vec = dm.getCoordinates()
    vloc = coord_dm.getLocalVec()
    vglob = coord_dm.getGlobalVec()

    coords = np.asarray(local_vec.array, dtype=np.double).reshape(-1, cdim).copy()

    # Fixed computational reference = coords at first call, cached on mesh
    # (ghosted: this rank's local array including halo).
    ref = getattr(mesh, "_mmpde_reference_coords", None)
    if ref is None or ref.shape != coords.shape:
        ref = coords.copy()
        mesh._mmpde_reference_coords = ref

    # --- RBF/Shepard bake of the metric (the production-fast path) ------
    # Evaluate the analytic metric ONCE on the fixed reference cloud, then
    # interpolate to the moving centroids each step via k-NN inverse-
    # distance (Shepard). The reference cloud is fixed in space ⇒ Eulerian.
    if metric_eval == "rbf":
        from scipy.spatial import cKDTree
        M_ref = _eval_M_analytic(ref)                    # one analytic pass
        _tree = cKDTree(ref)
        _kk = int(rbf_k) if rbf_k else (cdim + 2)

        def _eval_M(pts):
            dist, idx = _tree.query(pts, k=_kk)
            if _kk == 1:
                return M_ref[idx]
            w = 1.0 / np.maximum(dist, 1.0e-12) ** 2
            w /= w.sum(axis=1, keepdims=True)
            return np.einsum('nk,nkab->nab', w, M_ref[idx])
    else:
        _eval_M = _eval_M_analytic

    # Mesh-owned boundary slip is applied per outer iter via mesh.boundary_slip
    # (below). Pre-touch Gamma_P1 here so the projected-normal MeshVariable
    # exists before any DM snapshot (footgun-safe; redundant with the central
    # pre-touch in smooth_mesh_interior, kept as defence-in-depth).
    from underworld3.meshing._ot_adapt import _resolve_slip
    _slip_pretouch = _resolve_slip(mesh, boundary_slip)  # pre-touch Gamma_P1 before DM build

    # Reference edge matrices (fixed) for the owned cells.
    def _edge_mats(X, cells):
        pc = X[cells]                               # (Nc, d+1, d)
        cols = [pc[:, k + 1] - pc[:, 0] for k in range(cdim)]
        return np.stack(cols, axis=2)               # (Nc, d, d) columns
    Eh = _edge_mats(ref, cells_own)
    detEh = np.linalg.det(Eh)

    a0 = signed_vol(coords, cells_all)
    orient = np.sign(np.median(a0)) or 1.0
    a0_own_med = float(np.median(np.abs(signed_vol(coords, cells_own))))
    if parallel:
        a0_own_med = uw.mpi.comm.allreduce(a0_own_med) / uw.mpi.size
    a_min_floor = float(area_floor_frac) * a0_own_med
    # Representative background cell size h0 (mean reference edge length over
    # owned cells), used to make the convergence test SCALE-RELATIVE: a move
    # of dmax < tol·h0 is negligible vs the cell size, so the adapt has
    # converged regardless of absolute coordinate units. (The old absolute
    # outer_tol=1e-7 never fired — dx~1e-6 ≫ 1e-7 yet ≪ h0~0.08 — so every
    # adapt ran to the n_outer cap.)
    _ecols = np.linalg.norm(Eh, axis=1)            # (n_own, cdim) edge lengths
    h0_scale = float(np.mean(_ecols)) if _ecols.size else 1.0
    if parallel:
        h0_scale = uw.mpi.comm.allreduce(h0_scale) / uw.mpi.size

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
        loc = float(np.sum(K * G))
        if parallel:
            loc = uw.mpi.comm.allreduce(loc)
        return loc

    def _min_area(X):
        amin = float((signed_vol(X, cells_own) * orient).min())
        if parallel:
            from mpi4py import MPI as _MPI
            amin = uw.mpi.comm.allreduce(amin, op=_MPI.MIN)
        return amin

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
            f"_winslow_mmpde: unknown accel {accel!r}; "
            f"choose from {_valid_accel}")
    _mmpde_beta = float(momentum)
    if _accel in ("heavyball", "hb-restart") and _mmpde_beta == 0.0:
        _mmpde_beta = 0.9
    _prev_disp = np.zeros_like(coords)
    _prev_v = np.zeros_like(coords)
    _prev_dir = np.zeros_like(coords)

    def _gdot(a, b, mask):
        s = float(np.sum(a[mask] * b[mask]))
        if parallel:
            from mpi4py import MPI as _MPI
            s = uw.mpi.comm.allreduce(s, op=_MPI.SUM)
        return s

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
            if np.all(np.isfinite(trial)) and _min_area(trial) > a_min_floor:
                Itr = _energy(trial)
                if Itr < prevI:
                    accepted = trial; Inew = Itr; break
            scale *= 0.5
        else:
            accepted = coords; Inew = prevI; scale = 0.0
        dmax = float(np.linalg.norm(
            (accepted - coords)[is_owned_v], axis=1).max(initial=0.0))
        if parallel:
            from mpi4py import MPI as _MPI
            dmax = uw.mpi.comm.allreduce(dmax, op=_MPI.MAX)
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
            _need_adapt = mm["misalignment"] >= float(skip_threshold)
            if uw.mpi.size > 1:
                from mpi4py import MPI as _MPI
                _need_adapt = bool(uw.mpi.comm.allreduce(
                    int(_need_adapt), op=_MPI.MAX))
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
            _winslow_spring(mesh, metric, pinned_labels, verbose,
                            boundary_slip=boundary_slip, **mk)
        elif method in ("ma", "monge-ampere", "monge_ampere"):
            _winslow_elliptic(mesh, metric, pinned_labels, verbose,
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
            _winslow_equidistribute(mesh, metric, pinned_labels,
                                     verbose,
                                     boundary_slip=boundary_slip,
                                     **mk)
        elif method in ("anisotropic", "aniso", "tensor"):
            _winslow_anisotropic(mesh, metric, pinned_labels,
                                 verbose,
                                 boundary_slip=boundary_slip, **mk)
        elif method in ("mmpde", "variational"):
            _winslow_mmpde(mesh, metric, pinned_labels, verbose,
                           boundary_slip=boundary_slip, **mk)
        else:
            raise ValueError(
                f"smooth_mesh_interior: unknown method {method!r}; "
                f"use 'spring' (default, fast volumetric), "
                f"'ma' (Monge–Ampère, isotropic, ~60× costlier), "
                f"'ot' / 'equidistribute' (linear OT-improvement "
                f"step, composable) or "
                f"'anisotropic' (tensor metric — reshapes cells / "
                f"removes slivers; does not beat the node-count "
                f"cap).")
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
        local_vec.array, dtype=np.double).reshape(-1, cdim).copy()

    for sweep in range(n_iters):
        new_int = np.empty((int_owned_local.shape[0], cdim),
                           dtype=np.double)
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
                disp = uw.mpi.comm.allreduce(
                    disp ** 2) ** 0.5
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
        smooth_mesh_interior(mesh, metric=rho,
                             method="anisotropic")

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
        Target COARSEST:FINEST edge-length ratio ``R = h_max/h_min``
        the metric grades to. When ``> 1`` this **overrides** ``amp``
        (and the strategy default). Because the mover equidistributes
        ``ρ`` (so cell edge ``h ∝ ρ^{-1/d}``), the full ``t∈[0,1]``
        range gives ``h_max/h_min = (1 + amp)^{power/d}``; this is
        inverted to set ``amp = R^{d/power} - 1``. So ``R`` is a
        predictable, **mesh-independent** resolution knob (``R=2`` ⇒
        finest cells ~2× smaller than the coarsest). The *realised*
        ratio tracks ``R`` until the fixed node budget saturates it
        (large ``R`` is capped — going further needs h-refinement to
        add nodes, not just redistribution). ``None`` (default) ⇒ use
        ``amp`` / the ``strategy`` preset. Exposed in the convection
        harness as ``--resolution-ratio``.
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
        fields (1 matches the anisotropic mover's default
        ``aux_degree``).
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

    # `refinement` R = target COARSEST:FINEST edge-length ratio (h_max/h_min).
    # The mover equidistributes ρ=(1+amp·t)^power, so cell edge h ∝ ρ^(-1/d) and
    # the full t∈[0,1] range gives h_max/h_min = (1+amp)^(power/d). Invert that to
    # set amp, so R is a predictable, mesh-independent edge-length ratio. This
    # OVERRIDES the strategy's amp (R is the explicit resolution knob).
    # (Previously `refinement` was accepted but unused — a silent no-op.)
    if refinement is not None and float(refinement) > 1.0:
        amp = float(refinement) ** (cdim / float(power)) - 1.0
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
            # flat regions (no clip kink) -> cleaner OT / Monge-Ampere meshes.
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
    metric : {"front-following", "gradient-uniform"}, default "front-following"
        Strategic equidistribution rule. ``"front-following"``
        concentrates cells where the gradient is steepest (mild
        grading). ``"gradient-uniform"`` aims for the same
        per-cell field change everywhere (best for advection-
        diffusion accuracy).
    skip_threshold : float, default 0.9
        Alignment threshold for the adapt-on-demand skip. If the
        existing mesh's :func:`mesh_metric_mismatch` alignment is
        ≥ this threshold, no adaptation happens and the function
        returns ``False``.
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
    # length of the mesh. Critical: this must be captured ONCE
    # (the first time follow_metric sees this mesh) and reused
    # thereafter. Re-measuring it from a deformed (already-
    # refined) mesh causes h0 to shrink each call, the spring
    # caps to shrink with it, and refinement to compound at
    # every adapt — the dt-crash bug surfaced 2026-05-22.
    _key = id(mesh)
    h0 = _FOLLOW_METRIC_H0_CACHE.get(_key)
    rest_coords = _FOLLOW_METRIC_REST_CACHE.get(_key)
    if h0 is None:
        ep = _edge_pairs(mesh.dm)
        coords = np.asarray(mesh.X.coords)
        if ep.shape[0]:
            h0 = float(np.linalg.norm(
                coords[ep[:, 1]] - coords[ep[:, 0]],
                axis=1).mean())
        else:
            h0 = 1.0
        if uw.mpi.size > 1:
            h0 = uw.mpi.comm.allreduce(h0) / uw.mpi.size
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
                q_min = float(q.min())
                if uw.mpi.size > 1:
                    from mpi4py import MPI as _MPI
                    q_min = uw.mpi.comm.allreduce(
                        q_min, op=_MPI.MIN)
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

