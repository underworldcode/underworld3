"""Reconnection: repair the element shapes a refinement pass leaves behind (2-D).

Refinement engines choose *where* to put a new vertex; they do not get to choose
how the surrounding cells reconnect. :mod:`underworld3.utilities.edge_split`
splits an edge in every cell incident on it, so a cell that nominated that edge
gains two well-shaped children while a cell dragged along — split at an edge it
did not nominate — gains a thin one. Reconnection is the missing third operation
of the classical refine / swap / smooth triple: UW3 has refine (:meth:`Mesh.adapt`)
and smooth (:meth:`Mesh.relax`), and this is swap.

What it is worth, and where the benefit actually comes from
----------------------------------------------------------
Measured on the production path — ``edge_split`` refinement of a real DM, flat-core
size field, error over the refined core, raw numbers in
``~/+Simulations/mesh_reconnection_study/results_production_repair.txt``.

A flip replaces two cells by two cells and inserts no vertex, so a repair pass run
**after** refinement is cell-count neutral and changes connectivity alone:

==========================  ===================  =====================
base mesh                   99th-pct max angle   core error, same DOFs
==========================  ===================  =====================
gmsh box (the normal case)  124.7 -> 120.5 deg   -0.4 %
gmsh box, regular           116.6 -> 116.6 deg   0 %
grid, aspect ratio 4        156.0 -> 115.1 deg   +0.9 %
non-Delaunay (scrambled)    175.5 -> 118.0 deg   -3.1 %
==========================  ===================  =====================

So as a post-pass this fixes **shape and only shape** — decisively on a poor base
(slivers below q=0.1 go 3.84 % to 0.00 %, and the aspect-ratio-4 row loses 41
degrees of maximum angle) and hardly at all on a gmsh base. Interpolation error
barely moves either way.

Run **between** refinement passes it does more, because a flip changes which edge
of a cell is longest and therefore where the *next* pass inserts a vertex. That
buys roughly 20-30 % lower core error per degree of freedom on a degraded base
(and 20-30 % fewer cells for the same size field), and nothing on a gmsh base. The
gain is therefore a *placement* gain that reconnection unlocks, not a connectivity
gain — the same conclusion this study reached about centroid refinement, in the
opposite direction.

The aspect-ratio-4 row is why the pass exists at all. That base has a maximum
angle of 90 degrees — ideal for P1, since the interpolation bound depends on the
maximum angle (Babuska-Aziz) and not the minimum — and longest-edge refinement
*degrades* it to 156, because repeatedly bisecting the longest edge of a
high-aspect-ratio right triangle manufactures obtuse cells. Refinement creates the
problem; only reconnection removes it.

Scope
-----
The pass considers every edge it is allowed to touch, not only those around
freshly inserted vertices. That is deliberate — the gains on a degraded base come
precisely from repairing connectivity the refinement did not create — but it does
mean a deliberately hand-built triangulation may be re-connected away from the
refined region, which is one reason the pass is opt-in.

Edges carrying an interface label are never flipped, which is what would protect a
fault or a material boundary that is *represented in the mesh*. Note that the
standard adapt-on-top fault workflow does not do that: there a ``Surface`` is a
distance field driving a refinement metric and a constitutive weak zone, and it
labels no mesh edge at all. Repair therefore reconnects freely across such a weak
zone — measured to be harmless, since the weak zone is a smooth function of
distance rather than a discontinuity across a facet, and a sheared weak-zone Stokes
solve gives the same vrms to four significant figures with and without repair. A
fault that must not be crossed has to be a labelled interface, not a distance
field.

Parallel: the frozen seam
-------------------------
A flip cannot be a :c:type:`DMPlexTransform` — a child's cone may only reference
its own parent's closure, while a flip's output cells use the *other* parent's
apex — so there is no inherited star-forest propagation and the DM must be
rebuilt. It is rebuilt **on the same point chart**: a 2-D flip adds and removes no
points, since the quad keeps its four vertices, five edges and two cells and only
the diagonal edge's cone and the two cell cones change. Preserving the numbering
means the point star-forest transfers verbatim, labels transfer by point id and
coordinates transfer unchanged, with no coordinate matching anywhere.

That holds because **no cavity may contain a cell incident on a shared plex
point**. A flip across a partition seam would need one rank's cell to reference an
edge living on another rank, which means enlarging its local chart and rebuilding
the star-forest — a much larger job. Freezing the seam instead costs a measured
0.9-3.5 % of repair sites at 56k cells and np=2..8, and that cost *halves* with
every halving of the target cell size, because repair sites scale with the refined
band while the sites a seam crosses stay O(1).

.. warning::

   **The repaired mesh is not partition-independent.** ``edge_split`` alone is
   bit-confluent — identical at any communicator size — and repair gives that up
   by construction, because which cells are frozen depends on where the
   partitioner drew the seam. Conformity, orientation, volume, labels and the
   star-forest remain exact at every rank count; it is the *choice* of flips near
   a seam that differs. This is the same trade adapt-on-top already makes in
   preferring local adaptation to global remeshing, but it is a change of contract
   relative to the engine, so repair is opt-in rather than automatic.

A related cost: the 99th-percentile maximum angle recovers fully under a frozen
seam, but the absolute maximum does not — a few of the worst cells sit on the seam
and are exactly the ones that may not be touched.

Status
------
2-D only. In 3-D no single flip suffices: the operator set has to become
quality-gated edge removal, and the empty-sphere property is no help either since
a Delaunay tetrahedralisation still contains slivers — measured directly, a
Delaunay tet mesh of a random cloud has 10 % of its cells below q=0.1. See
``docs/developer/design/mesh-reconnection-and-delaunay-adapt.md``.
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw

# Labels PETSc maintains itself. They are rebuilt by ``stratify`` so they must not
# be copied onto a fresh plex, and an edge carrying one is not an interface.
_TOPOLOGY_LABELS = ("depth", "celltype")

# Shewchuk's static filters (Robust Predicates, 1997) with eps = 2^-53. A
# determinant whose magnitude clears the bound has a certain sign; one that does
# not is reported as _UNCERTAIN and the caller declines to act.
#
# Declining is always safe: a flip is an optimisation, never a requirement. That
# is what lets a filter stand in for adaptive-precision arithmetic here. Exact
# arithmetic would resolve more cases and could not resolve any of them wrongly,
# but it is not needed to keep the mesh valid and it is far too slow to sit inside
# a refinement loop. What would NOT be safe is trusting a bare float determinant:
# an inconsistently signed predicate produces a non-conforming mesh.
_EPS = 1.1102230246251565e-16
_ORIENT_BOUND = (3.0 + 16.0 * _EPS) * _EPS

_UNCERTAIN = 0

# A flip must improve the pair's largest angle by at least this much, measured in
# the cosine. Without a margin, two configurations of equal quality could each
# look like an improvement on the other and the sweeps would cycle.
_MIN_GAIN = 1.0e-9


def _orient2d(pa, pb, pc):
    """Sign of the area of triangle ``(pa, pb, pc)``, positive for anticlockwise.

    Returns ``_UNCERTAIN`` when the filter cannot resolve the sign.
    """
    acx, acy = pa[0] - pc[0], pa[1] - pc[1]
    bcx, bcy = pb[0] - pc[0], pb[1] - pc[1]
    left, right = acx * bcy, acy * bcx
    det = left - right
    if det == 0.0:
        # Collinear as far as this arithmetic can tell. Reported as unresolved,
        # never as a sign: the filter below reduces to `0 >= 0` when both products
        # vanish — which they do for any axis-aligned collinear triple, an
        # ordinary configuration on a structured mesh — and would then return a
        # confident "clockwise" for points that are not clockwise at all.
        return _UNCERTAIN
    if abs(det) >= _ORIENT_BOUND * (abs(left) + abs(right)):
        return 1 if det > 0.0 else -1
    return _UNCERTAIN


def _smallest_cosine(triangles):
    """The most negative cosine of any interior angle across ``triangles``.

    A monotone stand-in for "the largest angle": angle is largest exactly where
    its cosine is smallest, and comparing cosines avoids an ``arccos`` per angle
    and the tiny non-monotonicity its rounding would introduce near 180 degrees.
    """
    worst = 1.0
    for P in triangles:
        for i in range(3):
            u = P[(i + 1) % 3] - P[i]
            v = P[(i + 2) % 3] - P[i]
            denom = np.hypot(u[0], u[1]) * np.hypot(v[0], v[1])
            if denom == 0.0:
                return -1.0
            worst = min(worst, float((u[0] * v[0] + u[1] * v[1]) / denom))
    return worst


# --------------------------------------------------------------- topology reads

def _coords(dm):
    return np.asarray(dm.getCoordinatesLocal().array).reshape(
        -1, dm.getCoordinateDim())


def _shared_points(dm):
    """Chart-indexed 0/1 flags marking points held by more than one rank.

    Marking the local leaves and OR-ing over the star-forest also flags the roots
    on the owning side, so every rank agrees on the seam. Reuses
    ``edge_split._sf_logical_or``, whose leaf/root convention — the same array
    passed as both leaf and root data — is the one proven correct against
    ``uwnvb_sf_lor`` in the C.
    """
    from underworld3.utilities import edge_split

    pStart, pEnd = dm.getChart()
    flag = np.zeros(pEnd - pStart, dtype=np.int32)
    if uw.mpi.size == 1:
        return flag
    try:
        _nroots, ilocal, _iremote = dm.getPointSF().getGraph()
    except (ValueError, TypeError):
        # An unpopulated star-forest reports a root count petsc4py cannot shape an
        # array from. Nothing is shared, so there is no seam.
        return flag
    if ilocal is not None and len(ilocal):
        flag[np.asarray(ilocal, dtype=np.int64) - pStart] = 1
    # COLLECTIVE, and reached on every rank: one that shares nothing still has to
    # participate or its peers block. Gate on communicator size, never on what
    # this rank happens to own.
    edge_split._sf_logical_or(dm, flag)
    return flag


def _labelled_points(dm):
    """Chart-indexed flags for points belonging to an **interface** label.

    A labelled interior edge is an interface — a named boundary, or a registered
    surface — and must never be flipped, since that is what protects a fault or a
    material boundary from being reconnected across.

    A label value carried by a **cell** is excluded, because it describes a
    *volume* and not an interface. That distinction is load-bearing rather than
    fastidious. ``Elements`` labels every cell of a gmsh mesh, and the
    ``uwnvb_bisect`` transform propagates a parent's labels to its children, so
    after refinement every new *interior edge* carries ``Elements`` as well.
    Treating any labelled point as an interface therefore locked 81 % of the
    interior edges of a plain refined box, and repair quietly did almost nothing
    on every real UW3 mesh — while hand-built fixtures, which have no such label,
    kept working. Over-locking is safe in the sense that it cannot corrupt a mesh,
    but it is not safe in the sense that matters: it disables the feature silently.

    A region *join* is handled separately, by :func:`_cell_regions`, which
    compares the two cells rather than reading the edge.
    """
    pStart, pEnd = dm.getChart()
    cS, cE = dm.getHeightStratum(0)
    flag = np.zeros(pEnd - pStart, dtype=bool)
    for i in range(dm.getNumLabels()):
        if dm.getLabelName(i) in _TOPOLOGY_LABELS:
            continue
        label = dm.getLabel(dm.getLabelName(i))
        values = label.getValueIS()
        if values is None:
            continue
        for val in values.getIndices():
            points = label.getStratumIS(int(val))
            if points is None:
                continue
            idx = np.asarray(points.getIndices(), dtype=np.int64)
            if not len(idx):
                continue
            if ((idx >= cS) & (idx < cE)).any():
                continue                 # a volume label, not an interface
            flag[idx - pStart] = True
    return flag


def _cell_regions(dm):
    """Per-cell label signature, or ``None`` when every cell carries the same one.

    An edge between two cells with different region values is a material
    interface even when the edge itself is unlabelled, so the signature is what
    lets those edges be locked. Built from label strata rather than a per-cell
    query, which would be one PETSc call per cell per label.
    """
    cS, cE = dm.getHeightStratum(0)
    names = [dm.getLabelName(i) for i in range(dm.getNumLabels())
             if dm.getLabelName(i) not in _TOPOLOGY_LABELS]
    sig = np.zeros((cE - cS, len(names)), dtype=np.int64)
    for j, name in enumerate(names):
        label = dm.getLabel(name)
        values = label.getValueIS()
        if values is None:
            continue
        for val in values.getIndices():
            points = label.getStratumIS(int(val))
            if points is None:
                continue
            idx = np.asarray(points.getIndices(), dtype=np.int64)
            cells = idx[(idx >= cS) & (idx < cE)]
            if len(cells):
                sig[cells - cS, j] = int(val)
    if sig.shape[1] == 0 or np.all(sig == sig[0]):
        return None
    return sig


def _cell_vertices_and_seam(dm, X, shared):
    """One closure pass: anticlockwise vertices of every cell, and the seam mask.

    Both need the transitive closure of every cell, so they are computed together
    rather than in two passes.
    """
    cS, cE = dm.getHeightStratum(0)
    vS, vE = dm.getDepthStratum(0)
    pStart, _pEnd = dm.getChart()
    any_shared = bool(shared.any())

    verts = np.empty((cE - cS, 3), dtype=np.int64)
    frozen = np.zeros(cE - cS, dtype=bool)
    for c in range(cS, cE):
        closure = np.asarray(dm.getTransitiveClosure(c)[0], dtype=np.int64)
        v = [int(p) for p in closure if vS <= p < vE]
        if _orient2d(X[v[0] - vS], X[v[1] - vS], X[v[2] - vS]) < 0:
            v = [v[0], v[2], v[1]]
        verts[c - cS] = v
        if any_shared:
            frozen[c - cS] = bool(shared[closure - pStart].any())
    return verts, frozen


# ------------------------------------------------------------------- the rebuild

def rebuild_with_cones(dm, new_cells, new_edges):
    """Build a fresh plex on the **same point chart** with the given cones replaced.

    Parameters
    ----------
    dm : PETSc.DMPlex
        Source mesh. Not modified.
    new_cells : dict
        ``{cell point: (v0, v1, v2)}`` with the vertices anticlockwise.
    new_edges : dict
        ``{edge point: (va, vb)}``.

    Returns
    -------
    PETSc.DMPlex
        A new mesh whose chart, coordinates, labels and point star-forest match
        the source, differing only in the replaced cones.

    Notes
    -----
    Surgery on the source is not possible: ``DMPlexSymmetrize`` refuses to run on
    a plex that already has supports, and nothing outside ``DMDestroy`` frees
    them. Hence a fresh plex with every untouched cone copied across.

    The cone-orientation convention is derived, not assumed. For a triangle the
    closure vertex order is anticlockwise, cone entry ``i`` is the edge joining
    closure vertices ``i`` and ``i+1`` (mod 3), and its orientation is ``0`` when
    the edge's own cone runs that way and ``-1`` when reversed. A wrong
    orientation does not raise — it silently yields wrong geometry — so it is
    computed from the edge cone every time.
    """
    pStart, pEnd = dm.getChart()
    vS, vE = dm.getDepthStratum(0)
    eS, eE = dm.getDepthStratum(1)
    cdim = dm.getCoordinateDim()

    new = PETSc.DMPlex().create(comm=dm.comm)
    new.setDimension(dm.getDimension())
    new.setChart(pStart, pEnd)
    for p in range(pStart, pEnd):
        new.setConeSize(p, dm.getConeSize(p))
    new.setUp()

    # Edges first: the cell wiring below reads edge cones back to derive
    # orientations, so they have to be the new ones already.
    for p in range(pStart, pEnd):
        if p in new_cells:
            continue
        if p in new_edges:
            new.setCone(p, [int(v) for v in new_edges[p]])
            continue
        new.setCone(p, [int(x) for x in dm.getCone(p)])
        orientation = [int(o) for o in dm.getConeOrientation(p)]
        if orientation:
            new.setConeOrientation(p, orientation)

    edge_of = {}
    for e in range(eS, eE):
        a, b = (int(v) for v in new.getCone(e))
        edge_of[(a, b) if a < b else (b, a)] = e

    for c, (v0, v1, v2) in new_cells.items():
        cone, orientation = [], []
        for x, y in ((v0, v1), (v1, v2), (v2, v0)):
            e = edge_of[(x, y) if x < y else (y, x)]
            cone.append(e)
            orientation.append(0 if int(new.getCone(e)[0]) == x else -1)
        new.setCone(c, cone)
        new.setConeOrientation(c, orientation)

    new.symmetrize()
    new.stratify()

    # Coordinates verbatim: the vertex points are unchanged, so this is the same
    # section over the same chart holding the same values.
    new.setCoordinateDim(cdim)
    section = new.getCoordinateSection()
    section.setNumFields(1)
    section.setFieldComponents(0, cdim)
    section.setChart(vS, vE)
    for v in range(vS, vE):
        section.setDof(v, cdim)
        section.setFieldDof(v, 0, cdim)
    section.setUp()
    coords = PETSc.Vec().createSeq(section.getStorageSize(),
                                   comm=PETSc.COMM_SELF)
    coords.array[:] = np.asarray(dm.getCoordinatesLocal().array)
    new.setCoordinatesLocal(coords)

    # Labels by point id. No coordinate matching is involved, which is the whole
    # reason for preserving the numbering.
    for i in range(dm.getNumLabels()):
        name = dm.getLabelName(i)
        if name in _TOPOLOGY_LABELS:
            continue
        new.createLabel(name)
        source, target = dm.getLabel(name), new.getLabel(name)
        values = source.getValueIS()
        if values is None:
            continue
        for val in values.getIndices():
            points = source.getStratumIS(int(val))
            if points is None:
                continue
            for p in points.getIndices():
                target.setValue(int(p), int(val))

    # The star-forest transfers verbatim: every rank preserves its numbering, so
    # the remote point numbers it carries are still the right ones.
    if uw.mpi.size > 1:
        new.setPointSF(dm.getPointSF())
    return new


# ---------------------------------------------------------------- the flip pass

def _flippable(dm, X, verts, frozen, locked, regions):
    """Edges worth flipping, as ``(edge, cell_t, cell_u, p, a, q, b, gain)``.

    ``(p, a, q, b)`` is the quad anticlockwise with ``(a, b)`` the current
    diagonal, so the flip replaces cells ``(p, a, b)`` and ``(a, q, b)`` by
    ``(p, a, q)`` and ``(p, q, b)``. An edge qualifies on two counts:

    * the quad is **strictly convex**, so the flip cannot invert a cell. Declined
      whenever the filtered orientation predicate cannot resolve a sign;
    * the flip **strictly reduces the largest of the pair's six angles**.

    The second test is deliberately not the Delaunay (in-circle) criterion, even
    though this is a Lawson flip. Delaunay maximises the *minimum* angle and says
    nothing about the maximum, while the P1 interpolation bound depends on the
    maximum angle and not the minimum (Babuska-Aziz). The two disagree in practice
    and not marginally: flipping a gmsh-generated mesh towards Delaunay was
    measured to *raise* the 99th-percentile maximum angle from 126.8 to 129.3
    degrees, because gmsh optimises element shape rather than the empty-circle
    property and its triangulation is therefore locally non-Delaunay exactly where
    it has chosen a better-shaped configuration. Since every UW3 mesh starts from
    gmsh, a repair pass that can degrade such a mesh is unusable. Gating on the
    angle instead makes the pass monotone by construction: it can decline, but it
    cannot make a mesh worse.
    """
    eS, eE = dm.getDepthStratum(1)
    cS, _cE = dm.getHeightStratum(0)
    vS, _vE = dm.getDepthStratum(0)
    pStart, _pEnd = dm.getChart()

    out = []
    for e in range(eS, eE):
        if locked[e - pStart]:
            continue
        support = [int(c) for c in dm.getSupport(e)]
        if len(support) != 2:
            continue                     # boundary edge: nothing to flip into
        t, u = support
        if frozen[t - cS] or frozen[u - cS]:
            continue
        if regions is not None and not np.array_equal(regions[t - cS],
                                                     regions[u - cS]):
            continue                     # region interface
        a, b = (int(v) for v in dm.getCone(e))
        p = next(v for v in verts[t - cS] if v not in (a, b))
        q = next(v for v in verts[u - cS] if v not in (a, b))

        # Order the diagonal so the quad p-a-q-b runs anticlockwise.
        if _orient2d(X[p - vS], X[a - vS], X[q - vS]) < 0:
            a, b = b, a
        if _orient2d(X[p - vS], X[a - vS], X[q - vS]) <= 0:
            continue                     # not strictly convex, or unresolved
        if _orient2d(X[p - vS], X[q - vS], X[b - vS]) <= 0:
            continue

        Xp, Xa, Xq, Xb = X[p - vS], X[a - vS], X[q - vS], X[b - vS]
        before = _smallest_cosine(((Xp, Xa, Xb), (Xa, Xq, Xb)))
        after = _smallest_cosine(((Xp, Xa, Xq), (Xp, Xq, Xb)))
        if after <= before + _MIN_GAIN:
            continue                     # no shape gain worth the flip
        out.append((e, t, u, int(p), a, int(q), b, after - before))

    # Best gain first, so that when two candidate flips share a cell and only one
    # can run this sweep, the sweep keeps the better of the two rather than
    # whichever the edge loop happened to reach first.
    out.sort(key=lambda row: -row[7])
    return out


def flip_to_reduce_max_angle(dm, max_sweeps=12):
    """Flip edges to reduce the largest element angles, leaving the seam alone.

    Parameters
    ----------
    dm : PETSc.DMPlex
        A 2-D simplex mesh. Not modified.
    max_sweeps : int
        Cap on sweeps. Reaching it warns rather than failing silently.

    Returns
    -------
    repaired : PETSc.DMPlex
        A new mesh on the same point chart, or ``dm`` itself if nothing flipped.
    n_flips : int
        Flips performed across all ranks.

    Notes
    -----
    Each sweep applies an **independent** set of flips — no two sharing a cell —
    and rebuilds once, instead of rebuilding per flip. Two flips sharing a cell
    would each rewire it from a stale reading of the other's result. Deferring the
    loser to the next sweep costs a sweep; not deferring it costs correctness.

    Every accepted flip strictly reduces its own pair's largest angle, so the pass
    cannot degrade a mesh. It is not guaranteed to reach a global optimum: reducing
    one pair's largest angle can raise a neighbouring pair's, so the sequence is a
    local improvement and ``max_sweeps`` is the guard against a pathological case
    cycling between configurations. In practice it converges in a few sweeps.
    """
    if dm.getDimension() != 2:
        raise NotImplementedError(
            "reconnect.flip_to_reduce_max_angle is 2-D only. In 3-D no single "
            "flip is enough — the operator set has to change to quality-gated "
            "edge removal, and a Delaunay tetrahedralisation still contains "
            "slivers so the empty-sphere test is no help either. See "
            "docs/developer/design/mesh-reconnection-and-delaunay-adapt.md")

    total = 0
    for _sweep in range(max_sweeps):
        X = _coords(dm)
        verts, frozen = _cell_vertices_and_seam(dm, X, _shared_points(dm))
        candidates = _flippable(dm, X, verts, frozen, _labelled_points(dm),
                                _cell_regions(dm))

        claimed = set()
        new_cells, new_edges = {}, {}
        for e, t, u, p, a, q, b, _gain in candidates:
            if t in claimed or u in claimed:
                continue
            claimed.update((t, u))
            new_edges[e] = (p, q)
            new_cells[t] = (p, a, q)
            new_cells[u] = (p, q, b)

        # COLLECTIVE, and reached on every rank: one with nothing to flip still
        # has to vote or its peers block waiting for it.
        n = uw.mpi.comm.allreduce(len(new_edges), op=MPI.SUM)
        if n == 0:
            break
        dm = rebuild_with_cones(dm, new_cells, new_edges)
        total += n
    else:
        uw.pprint(0, f"[reconnect] reached the {max_sweeps}-sweep cap with flips "
                     f"still pending. The mesh is valid and conforming but not "
                     f"fully repaired; raise max_sweeps if this matters.")

    return dm, total
