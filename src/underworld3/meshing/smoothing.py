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

from typing import Optional, Sequence

import numpy as np

import underworld3 as uw


# Cached adjacency keyed by (mesh-id, pinned-label-tuple, topology).
# Rebuilt automatically when the mesh topology changes.
_ADJ_CACHE: dict = {}


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

    is_bnd = _pinned_mask(dm, pinned_labels)
    tris = _tri_cells(dm)
    cdim = mesh.cdim
    v0 = edges[:, 0]
    v1 = edges[:, 1]

    coords = np.asarray(mesh.X.coords, dtype=np.double).copy()

    # Boundary tangential slip. Fully locking every boundary node
    # freezes the rim's angular distribution, so near a feature the
    # interior must distort (the "touchy"/anisotropic refinement).
    # Instead let boundary nodes SLIDE ALONG the boundary while
    # staying EXACTLY ON it: each ring gets its OWN centre (robust
    # if rings are not perfectly concentric) and every slip node is
    # snapped back to its original distance from that centre after
    # each step — so a slip node can change θ but can NEVER move
    # off / away from the surface (the radial DOF is removed, not
    # just penalised). One node per ring is a hard anchor (kills
    # the ring's rigid-rotation gauge). The global inversion guard
    # also blocks a slip node overtaking a neighbour (boundary
    # self-tangle). TODO: a general deformed / free-surface
    # boundary needs projection onto the boundary polyline, not a
    # per-ring radius — circular form is exact for the Annulus.
    if boundary_slip and is_bnd.any():
        bc = np.nonzero(is_bnd)[0]
        c0 = coords[bc].mean(axis=0)
        rg = np.round(np.linalg.norm(coords[bc] - c0, axis=1), 6)
        is_anchor = np.zeros(n_verts, dtype=bool)
        slip_center = np.zeros((n_verts, cdim))
        slip_rtarget = np.zeros(n_verts)
        for rv in np.unique(rg):
            grp = bc[rg == rv]
            rc = coords[grp].mean(axis=0)        # this ring's centre
            is_anchor[grp[np.argmax(
                (coords[grp] - rc)[:, 0])]] = True
            slip_center[grp] = rc
            slip_rtarget[grp] = np.linalg.norm(
                coords[grp] - rc, axis=1)
        is_slip = is_bnd & ~is_anchor
        is_pinned = is_anchor
        sidx = np.nonzero(is_slip)[0]
        s_ctr = slip_center[sidx]
        s_rad = slip_rtarget[sidx]

        def _project(Y):
            v = Y[sidx] - s_ctr
            nrm = np.linalg.norm(v, axis=1)
            nrm = np.where(nrm > 1.0e-30, nrm, 1.0)
            Y[sidx] = s_ctr + v * (s_rad / nrm)[:, None]
            return Y
    else:
        is_pinned = is_bnd
        is_slip = np.zeros(n_verts, dtype=bool)

        def _project(Y):
            return Y

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


def _use_direct_solver(solver, singular=False):
    r"""Force a cached MA sub-solver onto a sparse **direct** factorisation
    (MUMPS LU) instead of the UW3 default GMRES + GAMG.

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
    o = solver.petsc_options
    # These three sub-problems are *linear* (φ Poisson with the Hessian
    # source frozen; the SPD Hessian-recovery mass system; the ∇φ
    # projection) → one KSP solve, no Newton line-search / 2nd iterate
    # (which was doubling work and emitting spurious
    # ``DIVERGED_LINEAR_SOLVE`` after 2 iters).
    o["snes_type"] = "ksponly"
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
        o["mg_coarse_pc_type"] = "lu"
        o["mg_coarse_pc_factor_mat_solver_type"] = "mumps"
    else:
        o["ksp_type"] = "cg"              # consistent mass = SPD
        o["pc_type"] = "jacobi"           # mass matrix → trivial
        for k in ("ksp_gmres_restart", "pc_gamg_type",
                  "pc_gamg_agg_nsmooths", "pc_gamg_threshold",
                  "mg_levels_ksp_type", "mg_levels_pc_type",
                  "mg_levels_ksp_max_it", "mg_coarse_pc_type",
                  "mg_coarse_pc_factor_mat_solver_type"):
            try:
                o.delValue(k)
            except Exception:
                pass


def _patch_volumes(tris, coords, n_verts):
    """Per-vertex dual-patch area: a node's share (1/3) of every
    incident triangle's |area|. ρ_cur ∝ 1/patch for the (opt-in,
    n_outer>1) outer MA composition; at equidistribution
    ``patch · ρ_tgt`` is uniform. Serial-exact (parallel under-counts
    at rank-partition boundaries — acceptable for serial validation).
    """
    area = np.abs(_signed_areas(coords, tris)) / 3.0
    patch = np.zeros(n_verts, dtype=np.double)
    for k in range(3):
        np.add.at(patch, tris[:, k], area)
    patch[patch <= 0.0] = patch[patch > 0.0].mean()
    return patch


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
                      move_anisotropy=None):
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
                _use_direct_solver(s, singular)
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
        is_bnd = _pinned_mask(dm, pinned_labels)
        tris = _tri_cells(dm)
        pStart, pEnd = dm.getDepthStratum(0)
        n_verts = pEnd - pStart
        old_coords = np.asarray(mesh.X.coords).copy()
        _cdim = mesh.cdim

        # Boundary tangential slip (same per-ring radius projection
        # as the spring). MA's natural Neumann BC (∇φ·n̂=0) already
        # makes ∇φ tangential at the boundary, so letting boundary
        # nodes move by ∇φ then snapping back to their ring radius
        # is the redistribution the formulation naturally wants —
        # fully pinning them discards it. Nodes provably stay on
        # the surface (radial DOF removed; drift ~machine ε). One
        # node/ring anchors the rotation gauge.
        if boundary_slip and is_bnd.any():
            bc = np.nonzero(is_bnd)[0]
            c0 = old_coords[bc].mean(axis=0)
            rg = np.round(
                np.linalg.norm(old_coords[bc] - c0, axis=1), 6)
            is_anchor = np.zeros(n_verts, dtype=bool)
            slip_center = np.zeros((n_verts, _cdim))
            slip_rtarget = np.zeros(n_verts)
            for rv in np.unique(rg):
                grp = bc[rg == rv]
                rc = old_coords[grp].mean(axis=0)
                is_anchor[grp[np.argmax(
                    (old_coords[grp] - rc)[:, 0])]] = True
                slip_center[grp] = rc
                slip_rtarget[grp] = np.linalg.norm(
                    old_coords[grp] - rc, axis=1)
            is_slip = is_bnd & ~is_anchor
            is_pinned = is_anchor
            _sidx = np.nonzero(is_slip)[0]
            _sctr = slip_center[_sidx]
            _srad = slip_rtarget[_sidx]

            def _project(Y):
                v = Y[_sidx] - _sctr
                nrm = np.linalg.norm(v, axis=1)
                nrm = np.where(nrm > 1.0e-30, nrm, 1.0)
                Y[_sidx] = _sctr + v * (_srad / nrm)[:, None]
                return Y
        else:
            is_pinned = is_bnd

            def _project(Y):
                return Y

        if tris is not None and n_outer > 1:
            patch = _patch_volumes(tris, old_coords, n_verts)
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

        g = c / (metric * vol_field.sym[0])
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
        _zig = (linear_solver != "gamg")
        prev_change = None
        for it in range(n_picard):
            phi_prev = np.asarray(phi.array).copy()
            ps.solve(zero_init_guess=_zig)
            phi.array[...] = ((1.0 - omega) * phi_prev
                              + omega * np.asarray(phi.array))
            hsolver.solve()
            change = float(np.abs(
                np.asarray(phi.array) - phi_prev).max())
            if uw.mpi.size > 1:
                from mpi4py import MPI as _MPI
                change = uw.mpi.comm.allreduce(
                    change, op=_MPI.MAX)
            if prev_change is not None and change < 1.0e-6:
                break
            prev_change = change

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


def _winslow_anisotropic(mesh, metric, pinned_labels, verbose,
                         n_outer=12, relax=0.2, beta=200.0,
                         aniso_cap=2.0, boundary_slip=False,
                         linear_solver="direct", phi_degree=2,
                         move_anisotropy=None, metric_role="M",
                         outer_tol=1.0e-4):
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

    So (3) trades grading *magnitude* for clean anisotropic *cell
    alignment* — exactly its intended role (see the warning above).
    ``relax`` (default 0.2) under-relaxes the per-step displacement;
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
                _use_direct_solver(s, singular)

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

    _zig = (linear_solver != "gamg")

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
    old0 = np.asarray(mesh.X.coords).copy()
    gproj.solve()
    Dcoords = np.asarray(Df.coords)
    gvec = np.asarray(
        uw.function.evaluate(grho.sym, Dcoords)).reshape(-1, cdim)
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
    lam_lo = 1.0 / h0 ** 2                      # coarsest
    lam_hi = 1.0 / (h0 / np.sqrt(aniso_cap)) ** 2  # finest
    Dout = np.empty((Dcoords.shape[0], 2, 2))
    eye2 = np.eye(2)
    for i in range(Dcoords.shape[0]):
        g = gvec[i]
        gni = gn[i]
        if gni > g_eps and gmax > g_eps:
            gh = g / gni
            M = base * (eye2 + beta * (gni / gref) ** 2
                        * np.outer(gh, gh))
        else:
            M = base * eye2
        w, V = np.linalg.eigh(M)
        w = np.clip(w, lam_lo, lam_hi)
        if metric_role == "Minv":
            w = 1.0 / w
        Dout[i] = (V * w) @ V.T
    Df.array[:, 0, 0] = Dout[:, 0, 0]
    Df.array[:, 0, 1] = Dout[:, 0, 1]
    Df.array[:, 1, 0] = Dout[:, 1, 0]
    Df.array[:, 1, 1] = Dout[:, 1, 1]

    for outer in range(n_outer):
        dm = mesh.dm
        pStart, pEnd = dm.getDepthStratum(0)
        n_verts = pEnd - pStart
        is_bnd = _pinned_mask(dm, pinned_labels)
        tris = _tri_cells(dm)
        old_coords = np.asarray(mesh.X.coords).copy()
        _cdim = mesh.cdim

        # Boundary tangential slip — identical per-ring radius
        # projection to _winslow_elliptic (the radial DOF is
        # removed, so slip nodes provably stay on their ring; one
        # node/ring anchors the rotation gauge).
        if boundary_slip and is_bnd.any():
            bc = np.nonzero(is_bnd)[0]
            c0 = old_coords[bc].mean(axis=0)
            rg = np.round(
                np.linalg.norm(old_coords[bc] - c0, axis=1), 6)
            is_anchor = np.zeros(n_verts, dtype=bool)
            slip_center = np.zeros((n_verts, _cdim))
            slip_rtarget = np.zeros(n_verts)
            for rv in np.unique(rg):
                grp = bc[rg == rv]
                rc = old_coords[grp].mean(axis=0)
                is_anchor[grp[np.argmax(
                    (old_coords[grp] - rc)[:, 0])]] = True
                slip_center[grp] = rc
                slip_rtarget[grp] = np.linalg.norm(
                    old_coords[grp] - rc, axis=1)
            is_slip = is_bnd & ~is_anchor
            is_pinned = is_anchor
            _sidx = np.nonzero(is_slip)[0]
            _sctr = slip_center[_sidx]
            _srad = slip_rtarget[_sidx]

            def _project(Y):
                v = Y[_sidx] - _sctr
                nrm = np.linalg.norm(v, axis=1)
                nrm = np.where(nrm > 1.0e-30, nrm, 1.0)
                Y[_sidx] = _sctr + v * (_srad / nrm)[:, None]
                return Y
        else:
            is_pinned = is_bnd

            def _project(Y):
                return Y

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
        ``method="anisotropic"`` the validated knobs are:

        * ``aniso_cap`` (default 2.0) — max cell anisotropy /
          spacing ratio. The **binding stability lever**: ≈2 is
          robust, ≈4 needs a gentler ``relax`` + more ``n_outer``,
          ``≳6`` folds the decoupled direct form.
        * ``relax`` (default 0.2) — per-step under-relaxation of
          the damped MMPDE iteration.
        * ``n_outer`` (default 12) — composed damped steps
          (early-exits on ``outer_tol``).
        * ``linear_solver`` (``"direct"`` default, MUMPS, or
          ``"gamg"`` — validated bit-parity here, the
          parallel-scalable path).
        * ``beta`` (default 200) — how fast the metric saturates
          the ``aniso_cap`` clamp (the clamp, not ``beta``, is the
          lever). ``move_anisotropy`` — optional radial/tangential
          move reweight (quality knob).

        Example::

            smooth_mesh_interior(
                mesh, metric=rho, method="anisotropic",
                method_kwargs=dict(aniso_cap=2.0, relax=0.2,
                                   n_outer=12))
    verbose : bool, default False
        Print per-sweep (Jacobi) or periodic (spring/MA) progress.

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
    if pinned_labels is None:
        pinned_labels = _auto_pinned_labels(mesh)
    pinned_labels = tuple(pinned_labels)

    if metric is not None:
        mk = dict(method_kwargs or {})
        if method == "spring":
            _winslow_spring(mesh, metric, pinned_labels, verbose,
                            boundary_slip=boundary_slip, **mk)
        elif method in ("ma", "monge-ampere", "monge_ampere"):
            _winslow_elliptic(mesh, metric, pinned_labels, verbose,
                              boundary_slip=boundary_slip, **mk)
        elif method in ("anisotropic", "aniso", "tensor"):
            _winslow_anisotropic(mesh, metric, pinned_labels,
                                 verbose,
                                 boundary_slip=boundary_slip, **mk)
        else:
            raise ValueError(
                f"smooth_mesh_interior: unknown method {method!r}; "
                f"use 'spring' (default, fast volumetric), "
                f"'ma' (Monge–Ampère, isotropic, ~60× costlier) or "
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
    amp: float = 8.0,
    lo_percentile: float = 50.0,
    hi_percentile: float = 97.0,
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

        \rho = 1 + \mathrm{amp}\cdot t,\qquad
        t = \mathrm{clip}\!\Big(
            \frac{|\nabla\mathrm{field}| - g_{lo}}
                 {g_{hi} - g_{lo}}, 0, 1\Big),

    with ``g_lo, g_hi`` the lo/hi percentiles of ``|∇field|`` (the
    same percentile-window idea as the adaptation metric).
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
    amp : float, default 8.0
        Bunching intensity: ``ρ_max = 1 + amp`` where ``|∇field|``
        is strongest. Larger ⇒ stronger redistribution.
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

    Returns
    -------
    sympy expression
        ``1 + amp * t.sym[0]`` — Lagrangian, frozen at call time.
    """
    import sympy

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
        _MDG_CACHE[key] = (g, gp, rho0)
    else:
        g, gp, rho0 = cache

    f_sym = (field.sym[0] if hasattr(field, "sym")
             else sympy.sympify(field))
    gp.uw_function = sympy.Matrix(
        [f_sym.diff(X[i]) for i in range(cdim)]).T
    gp.solve()
    gmag = np.linalg.norm(np.asarray(uw.function.evaluate(
        g.sym, rho0.coords)).reshape(-1, cdim), axis=1)
    g_lo = float(np.percentile(gmag, lo_percentile))
    g_hi = float(np.percentile(gmag, hi_percentile))
    # No-op guard: a uniform field has |∇field| ≡ 0, but the L2
    # projection leaves ~1e-18 round-off. Percentile-normalising
    # that noise would fabricate a spurious [0,1] metric (the same
    # failure the mover's own g_eps floor fixes). Any real field
    # gradient is many orders above 1e-9 ⇒ a (near-)constant field
    # yields ρ ≡ 1 (no refinement) exactly.
    if g_hi <= 1.0e-9:
        rho0.data[:, 0] = 0.0
    else:
        rho0.data[:, 0] = np.clip(
            (gmag - g_lo) / max(g_hi - g_lo, 1.0e-30), 0.0, 1.0)
    return 1.0 + float(amp) * rho0.sym[0]
