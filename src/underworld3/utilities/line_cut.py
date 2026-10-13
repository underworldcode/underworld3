"""Make a line a chain of mesh edges, on a mesh that already exists.

A weak zone whose boundary runs *through* elements cannot be represented by a
linear element: inside such an element the discrete stress is the interpolated
viscosity times the interpolated strain rate, whose cell average differs from the
honest one by

.. math::  -2\\,\\mathrm{Cov}(\\eta, \\dot\\varepsilon)

per cell. That covariance is zero for any element lying wholly inside or wholly
outside the zone and positive only for elements that **straddle** it, so the
artefact is not a resolution problem — refining shrinks the straddling band but
never empties it. The cure is to stop straddling: split each edge the line crosses
**at the crossing point**, so the line becomes a chain of element edges and every
element lies cleanly on one side.

The point of doing this *on top of* an existing mesh, rather than building the
line into the mesh generator, is that the line's position may be a design variable
in an outer optimisation: the base mesh — and the multigrid hierarchy resting on
it — has to stay fixed while the line moves. Nothing here modifies the mesh it is
given; the cut is a new mesh.

Mechanism
---------
No new topology code is needed. The ``uwnvb_bisect`` transform inserts a vertex on
every marked edge, and for a triangle carrying **two** marks it emits the segment
joining the two inserted vertices — which is exactly the cut. A triangle carrying
one mark emits the segment from the inserted vertex to the **opposite** vertex,
which is the cut for a triangle the line enters through an edge and leaves through
a corner. Marking every crossed edge and applying the transform once therefore
produces the whole chain. The transform places each new vertex at the edge
midpoint; moving it to the true crossing is a coordinate write, and the topology
does not care.

Snap or cut
-----------
A crossing landing close to an existing vertex leaves a sliver — in the worst case
measured here, a cell of area 1e-24 and a zero interior angle. So a crossing
within ``snap_frac`` of an edge's end, *measured along that edge*, moves the
vertex onto the line instead of splitting beside it (in the cut mesh; the mesh
passed in is untouched).

The along-edge measure is the one that matters: it is exactly the short side of
the sliver that would otherwise be created, and unlike an absolute distance it
carries no length scale, so the same tolerance works on any mesh. Note this is a
*discrete* switch — as a line sweeps across the mesh the topology changes in
jumps, which anything optimising over the line's position has to live with.
:func:`sliver_report` measures what a given tolerance buys.

Cutting without snapping costs about 60 % more iterations of an algebraic solver
on the slivers it leaves behind, and snapping buys that back — which is where the
0.10 default comes from. A Lawson flip pass
(:func:`~underworld3.utilities.reconnect.flip_to_reduce_max_angle`, which locks
the cut automatically because :data:`CUT_LABEL` is an edge label) helps less, so
raising the snap fraction is the better lever and repair is a second-order
touch-up rather than a requirement. The measurements behind all of that are in
``docs/developer/subsystems/conforming-surfaces-and-fault-zones.md``.

Scope
-----
Two dimensions, and lines that cross the mesh from boundary to boundary. A line
*ending* inside the mesh (a fault tip) leaves a triangle the line enters but does
not leave, which bisects without cutting; that is refused rather than silently
mis-meshed, as are triangles crossed three times.

Parallel
--------
Two rules hold everywhere in this module, and both are load-bearing rather than
defensive.

**Every refusal is global.** A rank-local ``raise`` aborts one rank while its
peers walk on into the next collective and block there, so what should be a clear
error message becomes a hang. Every condition tested here is a property of one
rank's cells, so each is reduced *before* it is tested: either every rank raises
or none does. The happy path is not evidence — a parallel test that never takes an
error path cannot see this class of defect at all, and nine of them have been
found this way so far. np=1 and np=2 both pass every one; np=3 is what exposes
them.

**Every tolerance is built from a GLOBAL length.** The cut is partition
independent because each quantity is a pure function of the coordinates and the
line, so every rank holding a shared edge computes the same crossing — which is
what ``uwnvb_bisect`` needs to keep the child point star-forest conforming. A
rank-local coordinate extent is not that: it varies with the partition (measured
0.58-0.67 against 1.0 in serial), and it raises outright on a rank owning no
vertices. :func:`_global_extent` is the one source of that number.
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities.edge_split import (_independent_edges, _owned_count,
                                              _sf_logical_or)

_BISECT_LABEL = "uwnvb_bisect_edges"

#: Default name for the label marking the mesh edges along a cut. Callers adding a
#: NAMED surface pass their own name instead, so the label doubles as the boundary
#: label a solver applies conditions on. Downstream passes read whatever it is
#: called: ``relax(pin_bands=...)`` holds it, and the reconnection pass refuses to
#: flip across it (it locks every non-topology EDGE label).
CUT_LABEL = "uw_cut_edges"


def _edge_vertices(dm):
    """(n_edges, 2) local vertex indices of every edge, in cone order."""
    vS, _vE = dm.getDepthStratum(0)
    eS, eE = dm.getDepthStratum(1)
    return np.array([dm.getCone(e) for e in range(eS, eE)], dtype=np.int64) - vS


def _coords(dm):
    return np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dm.getCoordinateDim())


def _global_extent(dm):
    """Longest side of the mesh's bounding box, reduced over every rank.

    COLLECTIVE. Every tolerance in this module is a fraction of this length, and
    the module's central invariant is that the crossings are a pure function of
    the coordinates and the line — so the length has to be the same number on
    every rank. A local ``np.ptp`` is not: it measures this rank's piece of the
    mesh, which spread 0.58-0.67 against 1.0 in serial on a three-way partition,
    and it raises on a rank owning no vertices, which is where the small coarse
    levels of a cut hierarchy end up.

    Reduced as a bounding BOX rather than as each rank's own longest side: the
    maximum of local extents is not the extent of the union.
    """
    cdim = dm.getCoordinateDim()
    X = _coords(dm)
    # Pack as [lo, -hi] so one MIN reduction serves both ends. An empty rank
    # contributes the identity, +inf, and must still take part in the reduce.
    box = (np.concatenate([X.min(axis=0), -X.max(axis=0)]) if len(X)
           else np.full(2 * cdim, np.inf))
    uw.mpi.comm.Allreduce(MPI.IN_PLACE, box, op=MPI.MIN)
    return float((-box[cdim:] - box[:cdim]).max())


def _segments(lines):
    """Every (A, B) segment of every polyline."""
    for pts in lines:
        pts = np.asarray(pts, dtype=float)[:, :2]
        for A, B in zip(pts[:-1], pts[1:]):
            if np.any(B != A):
                yield A, B


def _distance_to_lines(X, lines):
    """Distance from each point to the nearest point of the polylines.

    Measured to the *segments*, not to their infinite extensions, so a line stops
    attracting vertices beyond its own end.
    """
    best = np.full(len(X), np.inf)
    for A, B in _segments(lines):
        d = B - A
        u = np.clip(((X - A) @ d) / (d @ d), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(X - (A + u[:, None] * d), axis=1))
    return best


def _project_onto_lines(X, lines):
    """The nearest point of the polylines to each point."""
    best = np.full(len(X), np.inf)
    out = X.copy()
    for A, B in _segments(lines):
        d = B - A
        u = np.clip(((X - A) @ d) / (d @ d), 0.0, 1.0)
        foot = A + u[:, None] * d
        dist = np.linalg.norm(X - foot, axis=1)
        closer = dist < best
        best = np.where(closer, dist, best)
        out[closer] = foot[closer]
    return out


def _crossing_parameters(X, ends, lines, on_line):
    """Parameter along each edge where a line crosses it.

    ``NaN`` where the edge is not crossed. An edge with an endpoint already on a
    line is never counted: the line meets it at that vertex, and splitting it as
    well would put two cut vertices a hair apart.

    Every quantity here is a function of the coordinates and the line alone, so
    every rank holding a shared edge computes the same crossing. That is what
    makes the cut partition-independent, and what ``uwnvb_bisect`` needs to keep
    the child point star-forest conforming.
    """
    P, Q = X[ends[:, 0]], X[ends[:, 1]]
    touches = on_line[ends[:, 0]] | on_line[ends[:, 1]]

    t = np.full(len(P), np.nan)
    multiply_crossed = np.zeros(len(P), dtype=bool)
    for A, B in _segments(lines):
        d = B - A
        nrm = np.array([-d[1], d[0]])
        sP, sQ = (P - A) @ nrm, (Q - A) @ nrm
        straddles = (sP * sQ < 0.0) & ~touches

        with np.errstate(invalid="ignore", divide="ignore"):
            tk = np.where(straddles, sP / (sP - sQ), np.nan)
        # The crossing must lie between A and B, not merely on the infinite line
        # through them: a polyline is a chain of finite segments.
        foot = P + tk[:, None] * (Q - P)
        u = ((foot - A) @ d) / (d @ d)
        hit = straddles & (u >= 0.0) & (u <= 1.0)

        multiply_crossed |= hit & np.isfinite(t)
        t = np.where(hit, tk, t)
    return t, np.flatnonzero(multiply_crossed)


def _vertex_h(X, ends):
    """Mean length of the edges meeting each vertex: the local h AT a vertex."""
    L = np.linalg.norm(X[ends[:, 0]] - X[ends[:, 1]], axis=1)
    total = np.zeros(len(X))
    count = np.zeros(len(X))
    for k in (0, 1):
        np.add.at(total, ends[:, k], L)
        np.add.at(count, ends[:, k], 1.0)
    return total / np.maximum(count, 1.0)


def _resolve_snapping(dm, X, ends, lines, snap_frac, scale, snap_quality=0.5,
                      snap_dist=0.0):
    """Which vertices to move onto the line, and where the crossings then land.

    A crossing at parameter ``t`` on an edge sits ``t`` of the way along it, so
    ``t < snap_frac`` means the split would carve off a sliver whose short side is
    that fraction of an edge. In that case the edge's endpoint is moved onto the
    line instead and the edge is left whole.

    Snapping a vertex changes the crossings on every edge that touches it, which
    can bring a further crossing close to a vertex, so this repeats until the set
    settles. It settles quickly — each round only ever adds vertices, and the mesh
    is finite — but the loop is capped rather than trusted.

    The VETO is what lets ``snap_frac`` be large. Without it the tolerance does
    two jobs at once: it decides which crossings are near enough to snap, and — by
    having no say in the matter afterwards — how much damage is acceptable. A cell
    thinner than the tolerance band has every corner pulled onto the line from
    both sides and is flattened; measured on a graded mesh, every collapsed cell
    at ``snap_frac=0.4`` had all three corners snapped, and the cut was refused
    outright. So a proposed move that would drive an incident cell's quality below
    ``snap_quality`` is rejected, and that crossing falls back to being split —
    the path that already works. The tolerance then chooses, and the guard vetoes.

    An offending cell vetoes ALL of its moving corners at once rather than
    searching for the cheapest one to give up. Over-vetoing costs splits, which is
    the conservative direction, and it makes the outcome independent of the order
    cells are visited — which a partition would otherwise change.

    The chosen set is reconciled over the point star-forest EVERY round. The
    decision "this crossing is too close to that end" is read off one edge, and a
    rank holding only one side of a shared vertex can decide differently from its
    neighbour. The vertex then moves on one rank and not the other, the two ranks
    disagree about which edges are crossed, and the caller's split loop never
    empties its crossing set — measured, at np=3, as a cut that converged at
    snap_frac=0 and never converged at snap_frac=0.1.

    The veto is reconciled the same way and for a sharper reason: a vertex's
    incident cells are spread across the ranks that share it, so a rank can hold
    the ruined cell that another rank cannot see. Vetoes are OR-reduced, so one
    rank objecting stops the move everywhere.
    """
    cell_verts = _cell_vertices(dm)
    # The floor a cell must not drop below: the absolute one, or its own current
    # quality if it already sits under that. "Never below the floor, and never
    # worse if already below it" — which, unlike a floor expressed as a FRACTION
    # of the current quality, does not compound when this routine is applied
    # repeatedly. Measured: a relative 0.5 over six refine-and-snap rounds
    # licenses 0.5**6 of the original, and the worst angle duly fell 15.4 -> 2.3
    # degrees while every individual round looked well behaved.
    floor = (None if snap_quality is None else
             np.minimum(snap_quality, np.abs(_cell_quality(X, cell_verts))))
    # Vertices ALREADY on the line — a tip or junction placed by
    # `pull_vertex_onto` — do not move, so they cannot ruin anything and must
    # never be vetoed. Vetoing one would unplace the very point that makes a
    # terminating chain legal.
    fixed = _distance_to_lines(X, lines) < 1e-12 * scale
    vetoed = np.zeros(len(X), dtype=bool)
    # Distance from each vertex to the line, in units of the local h. The
    # along-edge criterion cannot see this: a vertex can sit a fraction of an
    # element from the line while every edge meeting it is crossed near its
    # MIDPOINT, so no crossing is ever "close to an end" and nothing proposes it.
    # The cut then runs past it and leaves a cell with one edge on the surface
    # and its apex a fraction of h away. Measured: every cell below 15 degrees
    # had exactly that shape — two corners on the cut, elongated along it, apex
    # 0.45 W off — and no value of snap_frac touched a single one of them.
    reach = snap_dist * _vertex_h(X, ends) if snap_dist > 0.0 else None
    # Seed with the vertices that are ALREADY on the surface, not just the ones
    # snapping will move. A junction or a tip placed on a vertex lies exactly on
    # the line, so the edges radiating from it show `s == 0` and register no
    # strict sign change — nothing proposes them for snapping, and they would be
    # invisible here. The validation then reads such a cell as "entered but not
    # left" and refuses a perfectly legal branch: measured on a three-way (Y)
    # junction, which this makes work.
    on_line = fixed.copy()
    vS, _vE = dm.getDepthStratum(0)
    pStart, pEnd = dm.getChart()

    def reconcile(mask):
        """OR the mask over every rank sharing each vertex."""
        flag = np.zeros(pEnd - pStart, dtype=np.int32)
        flag[np.flatnonzero(mask) + vS - pStart] = 1
        _sf_logical_or(dm, flag)
        return flag[np.arange(len(X)) + vS - pStart] == 1

    # Rounds are spent on vetoes as well as on proposals now, and a veto can
    # re-open a crossing that had settled, so the cap is larger than the ten a
    # pure proposal loop needed.
    for _ in range(30):
        X_snapped = X.copy()
        if on_line.any():
            X_snapped[on_line] = _project_onto_lines(X[on_line], lines)

        t, multiply_crossed = _crossing_parameters(X_snapped, ends, lines, on_line)
        near = np.isfinite(t) & ((t < snap_frac) | (t > 1.0 - snap_frac))

        # The end the crossing is nearest is the one that would form the sliver.
        rows = np.flatnonzero(near)
        pick = ends[rows, np.where(t[rows] < 0.5, 0, 1)]
        proposed = on_line.copy()
        proposed[pick[~vetoed[pick]]] = True
        if reach is not None:
            close = (_distance_to_lines(X, lines) < reach) & ~vetoed
            proposed |= close

        # COLLECTIVE, so every rank must reach it — including one that proposes
        # nothing. An early `if not near.any(): return` here deadlocked at np=3:
        # the rank owning no part of the line walked out while its peers waited
        # in the reduce.
        proposed = reconcile(proposed) & ~vetoed

        # Would the proposal ruin a cell? Test it on the fully moved coordinates
        # rather than one vertex at a time: it is the cell with several corners
        # coming in from both sides that collapses, and no single move of that set
        # looks bad on its own.
        X_try = X.copy()
        if proposed.any():
            X_try[proposed] = _project_onto_lines(X[proposed], lines)
        fresh = np.zeros(len(X), dtype=bool)
        quality = _cell_quality(X_try, cell_verts)
        ruined = (np.zeros(len(cell_verts), dtype=bool) if floor is None
                  else (quality <= 0.0) | (quality < floor))
        if ruined.any():
            corners = np.unique(cell_verts[ruined].ravel())
            fresh[corners[proposed[corners] & ~fixed[corners]]] = True
        # Reduced whether or not this rank found anything: a rank seeing no
        # ruined cell still has to reach the exchange, and a vertex whose bad
        # cell lives on a neighbour must be vetoed here too.
        fresh = reconcile(fresh)
        if uw.mpi.comm.allreduce(int(fresh.any()), op=MPI.MAX):
            vetoed |= fresh
            continue

        # Settled is a GLOBAL property: one rank still moving means another round
        # for everyone, or the reconcile above goes unmatched.
        settled = uw.mpi.comm.allreduce(
            int(np.array_equal(proposed, on_line)), op=MPI.MIN)
        if settled:
            return on_line, X_snapped, t, multiply_crossed
        on_line = proposed

    raise RuntimeError(
        "snapping did not settle in 30 rounds; snap_frac is large enough that "
        "moving one vertex keeps dragging the next crossing into tolerance.")


def _cell_vertices(dm):
    """(n_cells, 3) local vertex indices of every triangle."""
    vS, vE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    return np.array([[int(p) - vS for p in dm.getTransitiveClosure(c)[0]
                      if vS <= p < vE] for c in range(cS, cE)],
                    dtype=np.int64).reshape(cE - cS, 3)


def _minimal_face_vertices(X, cell_verts, point):
    """Vertices of the smallest mesh face whose closure contains ``point``.

    A point sits in the relative interior of exactly one face: a triangle, an
    edge, or a vertex. This returns that face's vertices — the three corners
    of the cell holding it, the two ends of the edge it lies on, or the single
    vertex it coincides with. Empty when the point lies outside the mesh.

    The face is read off the BARYCENTRIC coordinates of the containing cell: a
    coordinate is zero exactly when the point lies on the opposite edge, so
    keeping the corners with a nonzero coordinate drops the corners the point
    is not "on". Doing it this way means one cell answers the question. Taking
    the union of the corners of every containing cell does NOT: a point on an
    edge lies in two cells, and the union then offers each cell's opposite
    apex, which belongs to one of them and not the other. Moving such an apex
    leaves the other cell without a corner on the point, which is the defect
    this whole path exists to avoid (issue #542).

    The coordinates are scale-free, so one relative tolerance serves cells of
    any size — which embedded meshes need, their cell sizes differing by an
    order of magnitude either side of a zone skin.
    """
    A = X[cell_verts[:, 0], :2]
    B = X[cell_verts[:, 1], :2]
    C = X[cell_verts[:, 2], :2]
    v0, v1, v2 = C - A, B - A, np.asarray(point, dtype=float)[:2] - A
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    # A degenerate cell has no interior to be inside of; it cannot hold the
    # point and must not divide.
    alive = np.abs(den) > 0.0
    den = np.where(alive, den, 1.0)
    lc = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    lb = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    la = 1.0 - lb - lc

    tol = 1e-9
    inside = np.flatnonzero(alive & (la >= -tol) & (lb >= -tol) & (lc >= -tol))
    if not len(inside):
        return np.empty(0, dtype=np.int64)
    # One containing cell settles it; the first keeps the answer independent
    # of how the cells happen to be ordered, since every containing cell names
    # the same face.
    c = int(inside[0])
    on = np.array([la[c] > tol, lb[c] > tol, lc[c] > tol])
    return cell_verts[c][on]


def _cell_quality(X, cell_verts):
    """Scale-free triangle quality: 1 equilateral, 0 degenerate, <0 inverted.

    ``4 sqrt(3) A / sum(l^2)``, signed through the area. The sign matters: an
    inverted cell reports a negative number instead of a small positive one, and
    a *flattened* cell reports ~0 either way. Testing area > 0 does neither — a
    cell snapping flat onto the line lands at area 1e-19 of random sign, which
    passes an inversion test about half the time. Measured: guarding on inversion
    alone still left a zero interior angle.
    """
    P = X[cell_verts]
    e = np.stack([P[:, 2] - P[:, 1], P[:, 0] - P[:, 2], P[:, 1] - P[:, 0]],
                 axis=1)
    twice_area = ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                  - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
    return 2.0 * np.sqrt(3.0) * twice_area / np.maximum(
        (e ** 2).sum(axis=(1, 2)), np.finfo(float).tiny)


def _cell_edge_counts(dm, crossed_edges, on_line_vertices):
    """Per cell: how many of its edges are crossed, how many corners are on a line."""
    cS, cE = dm.getHeightStratum(0)
    vS, vE = dm.getDepthStratum(0)
    pEnd = dm.getChart()[1]

    is_crossed = np.zeros(pEnd, dtype=bool)
    is_crossed[crossed_edges] = True
    is_on = np.zeros(pEnd, dtype=bool)
    is_on[np.flatnonzero(on_line_vertices) + vS] = True

    n_cross = np.zeros(cE - cS, dtype=np.int64)
    n_corner = np.zeros(cE - cS, dtype=np.int64)
    for c in range(cS, cE):
        edges = np.asarray(dm.getCone(c), dtype=np.int64)
        n_cross[c - cS] = is_crossed[edges].sum()
        verts = np.array([int(p) for p in dm.getTransitiveClosure(c)[0]
                          if vS <= p < vE], dtype=np.int64)
        n_corner[c - cS] = is_on[verts].sum()
    return n_cross, n_corner


def _child_vertex_of(child, positions, scale):
    """Child vertices nearest the given positions, and how many did not match.

    Parent vertices keep their coordinates through the transform and inserted
    vertices land on their parent edge's midpoint, so position identifies both.
    petsc4py does not expose ``DMPlexTransformGetTargetPoint``, so this is the
    available route, and matching on geometry keeps it independent of the
    transform's internal point numbering.

    The mismatch COUNT is returned rather than raised on. Raising here would be
    rank-local, and it would sit inside the caller's rank-local "did this rank
    split anything?" guard as well — two ways for one rank to leave while its
    peers wait in the next reduce. The caller reduces the count and raises for
    everyone. A rank with nothing to look up returns empty and zero, which is a
    result, not a special case.
    """
    if not len(positions):
        return np.empty(0, dtype=np.int64), 0

    Xc = _coords(child)
    vS, vE = child.getDepthStratum(0)
    tree = uw.kdtree.KDTree(np.ascontiguousarray(Xc[: vE - vS]))
    idx, dist_sqr, found = tree.find_closest_point(np.ascontiguousarray(positions))

    bad = (~np.asarray(found).ravel()
           | (np.asarray(dist_sqr).ravel() > (1e-9 * scale) ** 2))
    return np.asarray(idx, dtype=np.int64).ravel(), int(bad.sum())


def _set_coordinates(dm, indices, values):
    """Move a set of vertices. The coordinate vector is written whole."""
    vec = dm.getCoordinatesLocal()
    arr = np.asarray(vec.array).reshape(-1, dm.getCoordinateDim()).copy()
    arr[indices] = values
    new = vec.duplicate()
    new.array[:] = arr.reshape(-1)
    dm.setCoordinatesLocal(new)


def _label_cut_edges(dm, lines, tol, name, value):
    """Mark the edges lying along the cut; return the edge points marked.

    An edge is on the cut when both its endpoints and its midpoint lie on a line.
    The midpoint test is what distinguishes the cut from a chord: where a polyline
    turns, two vertices on different segments can be joined by an edge that is not
    part of the line at all.

    The points are returned rather than counted here because a shared edge is held
    by every rank on the seam: counting locally and summing would report it once
    per sharer. The caller counts the OWNED ones.
    """
    X = _coords(dm)
    on = _distance_to_lines(X, lines) < tol
    eS, _eE = dm.getDepthStratum(1)
    ends = _edge_vertices(dm)
    mid_on = _distance_to_lines(0.5 * (X[ends[:, 0]] + X[ends[:, 1]]), lines) < tol
    keep = on[ends[:, 0]] & on[ends[:, 1]] & mid_on

    if not dm.hasLabel(name):
        dm.createLabel(name)
    label = dm.getLabel(name)
    label.setDefaultValue(0)
    marked = np.flatnonzero(keep) + eS
    for e in marked:
        label.setValue(int(e), int(value))
    return marked


def cell_areas(dm):
    """Signed area of every triangle. Negative means the cell is inverted.

    A cell's cone holds its edges together with an orientation saying which way
    the cell traverses each; taking the first vertex of every edge regardless
    returns a ring that is not the cell, and reports zero area for perfectly good
    cells — so an inversion check built that way always passes.
    """
    X = _coords(dm)
    vS, _vE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    out = np.empty(cE - cS)
    for c in range(cS, cE):
        ring = []
        for e, o in zip(dm.getCone(c), dm.getConeOrientation(c)):
            a, b = (int(v) - vS for v in dm.getCone(e))
            ring.append(a if o >= 0 else b)
        p, q, r = X[ring[0]], X[ring[1]], X[ring[2]]
        out[c - cS] = 0.5 * ((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return out


def min_angles(dm):
    """Smallest interior angle of every triangle, in degrees.

    Area alone does not identify a sliver — a small cell near a refined feature is
    not one. A collapsing angle is what costs the solver.
    """
    X = _coords(dm)
    vS, vE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    out = np.empty(cE - cS)
    for c in range(cS, cE):
        v = np.array([int(p) - vS for p in dm.getTransitiveClosure(c)[0]
                      if vS <= p < vE])
        P = X[v]
        e = np.array([P[2] - P[1], P[0] - P[2], P[1] - P[0]])
        L = np.linalg.norm(e, axis=1)
        cosines = [(L[1] ** 2 + L[2] ** 2 - L[0] ** 2) / (2 * L[1] * L[2]),
                   (L[2] ** 2 + L[0] ** 2 - L[1] ** 2) / (2 * L[2] * L[0]),
                   (L[0] ** 2 + L[1] ** 2 - L[2] ** 2) / (2 * L[0] * L[1])]
        out[c - cS] = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0))).min()
    return out


def cut_along_lines(dm, lines, snap_frac=0.10, label=CUT_LABEL, label_value=1,
                    snap_quality=0.15, snap_dist=0.0):
    """Split every edge the given lines cross, at the crossing point.

    Parameters
    ----------
    dm : PETSc.DMPlex
        A 2-D simplex mesh. **Not modified** — the cut is returned as a new mesh,
        so a line can be moved and re-cut against the same fixed base.
    lines : sequence of array_like
        One or more polylines, each an ``(N, 2)`` array of points. They must not
        intersect one another or themselves, and each must cross the mesh from
        boundary to boundary.
    snap_frac : float
        A crossing landing within this fraction of an edge's length from either
        end moves that end onto the line instead of splitting the edge. This is
        what keeps slivers out; ``0.0`` disables it and will produce degenerate
        cells wherever a line passes near a vertex. The default is the smallest
        value measured to bring an algebraic solver back within about 15 % of the
        uncut mesh's cost (see the table above). The cut stays exactly on the
        line either way — a snapped vertex is moved *onto* the line, not the line
        onto the vertex — so what a larger value costs is displacement of the
        surrounding mesh, not accuracy of the interface. See :func:`sliver_report`.
    snap_quality : float
        Floor on triangle quality (``4 sqrt(3) A / sum(l^2)``: 1 equilateral, 0
        degenerate). A snap is refused if it would drive any cell touching it
        below this — or below that cell's present quality, if it is already worse
        — and the crossing is split instead. Roughly, 0.75 is a 30 degree worst
        angle, 0.43 is 15 degrees and 0.15 is 5 degrees.

        This is what makes a large ``snap_frac`` safe: without it, a cell thinner
        than the tolerance band has every corner pulled onto the line and is
        flattened. ``0.0`` leaves only the inversion veto, which is measurably
        *not* enough: a flattened cell lands at quality ~1e-16 of either sign, so
        half of them survive an inversion test, the worst interior angle still
        reaches zero, and — worse than the refusal it replaces — the returned mesh
        looks fine while the cut chain has silently broken. ``None`` removes the
        guard altogether, restoring the behaviour from before it existed — where
        a tolerance this large refuses the cut outright, which is the refusal
        the guard was written to avoid.

        Keep it LOW. The guard protects the SNAPPED mesh, which is not the mesh
        that comes back: every snap it vetoes becomes a split, and the splits are
        what make the cut's slivers. Measured on one graded cut, raising the floor
        from 0.15 to 0.55 held the snapped mesh's worst angle up (15.6 -> 24.5
        degrees) while driving the CUT's down (10.9 -> 0.16) and the split count
        up (139 -> 359). This is a backstop against flattening, not a quality
        target.
    snap_dist : float
        Also snap any vertex lying within this multiple of its own local h of the
        line, whatever the crossings on its edges look like. ``snap_frac`` is
        measured ALONG an edge and is blind to a vertex that sits close to the
        line while every edge meeting it is crossed near its midpoint — which is
        the configuration that produces the cut's worst cells, and which no value
        of ``snap_frac`` reaches. The quality guard applies to these proposals
        too, so a vertex whose move would flatten a cell is split around instead.
    label, label_value : str, int
        Name and stratum value of the label put on the cut edges. Naming it after
        the surface lets a solver apply a boundary condition there directly; the
        default is a generic name for callers that only want the geometry.

    Returns
    -------
    cut : PETSc.DMPlex
        A new mesh in which every segment of every line between consecutive
        crossings is an edge. Those edges carry ``label`` with value
        ``label_value``.
    info : dict
        ``n_split`` edges split, ``n_on_surface`` vertices lying on a line,
        ``n_cut_edges`` edges labelled, ``min_area`` and ``min_angle`` of the
        result. Every entry is GLOBAL, and counts are over owned points, so the
        numbers are the same at any communicator size.

        ``n_on_surface`` counts vertices moved onto a line by snapping AND
        vertices that were already on one — a tip or a junction placed on a
        vertex, which is how those are represented. Both are cut vertices; the
        distinction does not survive into the result.

    Raises
    ------
    ValueError
        If a line ends inside the mesh, if a triangle is crossed three times, or
        if an edge is crossed more than once. Each is a case this routine cannot
        cut correctly, and each would otherwise give a mesh that looks plausible
        and still leaks stress.
    RuntimeError
        If snapping inverts a cell or fails to settle, which means ``snap_frac``
        is too large for this mesh.

    Examples
    --------
    A single line crossing the mesh cuts ONE chain, so its facets number one
    fewer than the vertices along it — and every vertex along it is either one
    this routine inserted or one already on the line:

    >>> cut, info = cut_along_lines(mesh.dm, [np.array([[0.5, -0.1], [0.5, 1.1]])])
    >>> info["n_cut_edges"] == info["n_split"] + info["n_on_surface"] - 1
    True
    """
    if dm.getDimension() != 2:
        raise ValueError(
            f"cut_along_lines is 2-D; this mesh is {dm.getDimension()}-D. Cutting "
            "tetrahedra along a surface is a different pattern problem.")

    from underworld3.utilities import _nvb_transform  # noqa: F401  (registers the type)

    X = _coords(dm)
    ends = _edge_vertices(dm)
    # One global length, computed once and threaded through every tolerance
    # below. Cutting never moves a vertex outside the bounding box, so the same
    # number is valid for the child meshes the pass loop produces.
    scale = _global_extent(dm)

    on_line, X_snapped, t, multiply_crossed = _resolve_snapping(
        dm, X, ends, lines, snap_frac, scale, snap_quality, snap_dist)
    eS, _eE = dm.getDepthStratum(1)
    crossed = np.flatnonzero(np.isfinite(t)) + eS

    n_cross, n_corner = _cell_edge_counts(dm, crossed, on_line)
    # A triangle the line passes through leaves by an edge (two crossings) or by a
    # corner (one crossing, one on-line vertex). Anything else it cannot cut.
    #
    # Every one of these is REDUCED before it is tested. Each condition is a
    # property of one rank's cells, so a rank-local raise would abort that rank
    # while its peers walked on into the next collective and hung — which is
    # exactly what a three-way partition produced. Reducing first means every rank
    # raises, or none does.
    faults = np.array([len(multiply_crossed),
                       int(((n_cross == 1) & (n_corner == 0)).sum()),
                       int((n_cross == 3).sum())], dtype=np.int64)
    n_multi, n_tip, n_triple = uw.mpi.comm.allreduce(faults, op=MPI.SUM)

    if n_multi:
        raise ValueError(
            f"{n_multi} edge(s) are crossed more than once; an edge can only be "
            "split at one point. Refine the mesh near the line, or simplify the "
            "line.")
    if n_tip:
        raise ValueError(
            f"{n_tip} triangle(s) are entered but not left, which means a line "
            "ends inside the mesh. A line must cross from boundary to boundary; a "
            "terminating tip needs a vertex placed at the tip and is not "
            "supported.")
    if n_triple:
        raise ValueError(
            f"{n_triple} triangle(s) are crossed three times. The transform "
            "splits these into four rather than cutting them. Refine the mesh "
            "near the line so no triangle sees more than one line segment.")

    # Both reduced before either is tested: a rank that owns no part of the line
    # must not take a different branch from one that does. Counted over OWNED
    # points only — a shared edge or vertex sits on every rank of the seam, and
    # summing local counts would report it once per sharer.
    vS, _vE = dm.getDepthStratum(0)
    totals = np.array([_owned_count(dm, crossed),
                       _owned_count(dm, np.flatnonzero(on_line) + vS)],
                      dtype=np.int64)
    n_crossed_total, n_on_surface = uw.mpi.comm.allreduce(totals, op=MPI.SUM)
    if n_crossed_total == 0 and n_on_surface == 0:
        raise ValueError("no mesh edge is crossed by any line: nothing to cut.")

    # Apply the snapping to a WORKING COPY. The caller's mesh is never touched, so
    # a line can be moved and re-cut against the same fixed base. Unconditional:
    # an empty index set is a no-op, and a rank-local guard around mesh surgery is
    # the shape every deadlock in this module has had.
    work = dm.clone()
    _set_coordinates(work, np.flatnonzero(on_line), X_snapped[on_line])

    # Split in PASSES of pairwise-INDEPENDENT edges, never two edges of one cell
    # at once.
    #
    # The transform can split two edges of a triangle in one go, and doing so
    # emits the segment joining the two inserted vertices — the cut, in a single
    # pass. That works perfectly in serial and is WRONG IN PARALLEL: the
    # double-split path leaves the child's point star-forest inconsistent, and
    # wrapping the result as a Mesh dies in PetscSectionCreateGlobalSection
    # ("Global dof 0 for point N is not the unconstrained 2") at np>=3. Its own
    # source calls those tables "a safety net", and nothing had exercised them
    # across a partition.
    #
    # Independent single splits are the validated path, and they still build the
    # cut: splitting the entry edge inserts a vertex, and the SECOND pass splits
    # the exit edge and joins its new vertex to the OPPOSITE vertex of the cell —
    # which is the first vertex. The cut segment appears as an edge either way; it
    # just takes two passes rather than one.
    n_split, cut = 0, work
    for _pass in range(12):
        X_now = _coords(cut)
        ends_now = _edge_vertices(cut)
        on_now = _distance_to_lines(X_now, lines) < 1e-12 * scale
        t_now, _multi = _crossing_parameters(X_now, ends_now, lines, on_now)

        eS_now = cut.getDepthStratum(1)[0]
        want = np.flatnonzero(np.isfinite(t_now)) + eS_now
        chosen = _independent_edges(cut, want)

        n_this = uw.mpi.comm.allreduce(int(_owned_count(cut, chosen)), op=MPI.SUM)
        if n_this == 0:
            break
        n_split += n_this

        work_pass = cut.clone()
        work_pass.createLabel(_BISECT_LABEL)
        bisect_label = work_pass.getLabel(_BISECT_LABEL)
        bisect_label.setDefaultValue(0)
        for e in chosen:
            bisect_label.setValue(int(e), 1)

        transform = PETSc.DMPlexTransform().create(comm=work_pass.comm)
        transform.setType("uwnvb_bisect")
        transform.setDM(work_pass)
        transform.setUp()
        child = transform.apply(work_pass)
        transform.destroy()
        if child.hasLabel(_BISECT_LABEL):
            child.removeLabel(_BISECT_LABEL)

        # Move each inserted vertex from the midpoint, where the transform put it,
        # to the crossing. Everything else is already where it belongs.
        #
        # No `if len(chosen):` guard. A pass is entered by GLOBAL agreement, so a
        # rank that happens to have split nothing still has to reach the reduce
        # below; guarding the block would walk it straight past.
        ce = ends_now[chosen - eS_now]
        tc = t_now[chosen - eS_now][:, None]
        midpoints = 0.5 * (X_now[ce[:, 0]] + X_now[ce[:, 1]])
        targets, n_missing = _child_vertex_of(child, midpoints, scale)
        if uw.mpi.comm.allreduce(n_missing, op=MPI.SUM):
            raise RuntimeError(
                "expected vertex position(s) have no child vertex; the transform "
                "did not place points where this routine assumes it does.")
        _set_coordinates(child, targets,
                         (1.0 - tc) * X_now[ce[:, 0]] + tc * X_now[ce[:, 1]])
        cut = child
    else:
        raise RuntimeError(
            "the cut did not converge in 12 passes; every pass must split at "
            "least one edge and remove it from the crossing set.")

    # Reduced before it is tested. Whether a rank holds an inverted cell depends
    # on the partition, so this raise was rank-local at np>1 while its peers went
    # on into `Mesh(cut_dm, ...)` and waited there. Measured: snap_frac=0.49 on a
    # 1/12 box inverts a cell.
    areas = cell_areas(cut)
    n_inverted = uw.mpi.comm.allreduce(int((areas <= 0.0).sum()), op=MPI.SUM)
    if n_inverted:
        raise RuntimeError(
            f"snapping inverted {n_inverted} cell(s); snap_frac={snap_frac} is "
            "too large for this mesh.")

    marked = _label_cut_edges(cut, lines, 1e-9 * scale, label, label_value)

    # Every reported number is GLOBAL. They are printed together as one summary,
    # so a mix of rank-local and reduced values would be read as agreeing when
    # they do not — and the documented identity between them cannot hold.
    # `min_angles` is O(cells), so it is computed once and reduced with the rest.
    # An empty rank contributes the identity of each reduction, never a raise.
    angles = min_angles(cut)
    worst = np.array([areas.min() if areas.size else np.inf,
                      angles.min() if angles.size else np.inf])
    uw.mpi.comm.Allreduce(MPI.IN_PLACE, worst, op=MPI.MIN)
    return cut, {
        "n_split": int(n_split),
        "n_on_surface": int(n_on_surface),
        "n_cut_edges": int(uw.mpi.comm.allreduce(_owned_count(cut, marked),
                                                 op=MPI.SUM)),
        "min_area": float(worst[0]),
        "min_angle": float(worst[1]),
    }


def pull_vertex_onto(dm, targets):
    """Move the nearest mesh vertex onto each target point; return a new mesh.

    COLLECTIVE. This is how a fault TIP or a network JUNCTION is placed, and both
    are the same problem: a distinguished point of the geometry that has to
    coincide with a mesh vertex. Once it does, every branch meeting there arrives
    at the already-legal "one crossed edge, one on-surface corner" case, and
    :func:`cut_along_lines` terminates the chain cleanly instead of refusing it.

    Prefer this to snapping the tip to the nearest vertex. Moving the MESH keeps
    the tip exactly where it was asked for — measured on a 1/12 box, tip error
    0.0000 against 0.0306, and a better worst angle (8.79 deg against 5.29) — and
    it costs mesh displacement rather than geometric accuracy, the same trade as
    ``snap_frac``. The tip is where the stress concentrates, so accuracy there is
    worth more than tidiness.

    Parameters
    ----------
    dm : PETSc.DMPlex
        **Not modified.** The pull is returned as a new mesh.
    targets : array_like
        An ``(N, 2)`` array of points, or one point.

    Returns
    -------
    PETSc.DMPlex
        A copy with one vertex moved onto each target.

    Notes
    -----
    Which vertex gets chosen JUMPS as the target moves, so this is a discrete
    switch in what may be a continuous design variable — the same class of
    behaviour as ``snap_frac``, and anything optimising over fault geometry has
    to live with it.

    The vertex is chosen from the corners of the cells CONTAINING the target,
    not from the mesh at large. The globally nearest vertex can belong to a
    neighbouring cell, and moving that one leaves the containing cell with no
    corner on the line, so :func:`cut_along_lines` refuses the very tip this
    call was made to place (issue #542). Restricting the candidates ties the
    choice to the containment relation the cut actually tests. A target outside
    the mesh falls back to the nearest vertex anywhere.

    The choice is made from the coordinates alone and reduced globally, so every
    rank moves the same vertex: a rank-local nearest-vertex search picks a
    different one on each rank, which is how the mesh stops being
    partition-independent. Exact ties in distance are broken by coordinate order;
    two distinct vertices at bit-identical distance would be arbitrary, and that
    is measure-zero rather than handled.
    """
    X = _coords(dm)
    arr = X.copy()
    scale = _global_extent(dm)
    cell_verts = _cell_vertices(dm)

    for t in np.atleast_2d(np.asarray(targets, dtype=float))[:, :2]:
        # Candidates are the vertices of the smallest face CONTAINING the
        # target, not every vertex in the mesh. The nearest vertex overall can
        # belong to a neighbouring cell, and moving it leaves the containing
        # cell without a corner on the line — precisely the "entered but not
        # left" state cut_along_lines refuses, reported for a tip that does
        # have a vertex on it (issue #542). Measured inside a ribbon zone of
        # width 0.02: the target sat on an edge whose two ends were at exactly
        # 1.000e-02, being the ribbon's own edge vertices, and the apex of one
        # of the two cells sharing that edge won by 0.4% at 9.963e-03.
        face = _minimal_face_vertices(X, cell_verts, t)
        held = len(face) > 0
        candidates = face if held else np.arange(len(X))
        d = np.linalg.norm(X[candidates, :2] - t, axis=1)
        # Reduced as (outside, distance, x, y) so both decisions ride one
        # reduction: tuples compare lexicographically, so MIN prefers ANY rank
        # that holds a containing cell over every rank that does not, then the
        # closest corner, then the one lowest in coordinate order. The
        # outside-flag has to lead: a rank with no part of the containing cell
        # proposes its own nearest vertex, which can easily be closer than the
        # right answer, and a reduction on distance alone would take it.
        #
        # All ranks flagging 1 means the target lies outside the mesh, and the
        # fallback is the nearest vertex anywhere — the behaviour before the
        # containment rule, kept because a target off the mesh still has to
        # land somewhere.
        local = ((0 if held else 1, float(d.min()),
                  *X[candidates[int(d.argmin())], :2])
                 if d.size else (1, np.inf, np.inf, np.inf))
        _outside, _dist, tx, ty = uw.mpi.comm.allreduce(local, op=MPI.MIN)

        # Move it by POSITION, not by index: the chosen vertex may be a ghost
        # here and an owned point there, and both copies have to end up in the
        # same place without a star-forest exchange.
        hit = np.flatnonzero(np.linalg.norm(X[:, :2] - np.array([tx, ty]),
                                            axis=1) < 1e-12 * scale)
        arr[hit, :2] = t

    out = dm.clone()
    _set_coordinates(out, np.arange(len(arr)), arr)
    return out


def sliver_report(dm, lines, snap_fracs):
    """How the cut's worst cell varies with the snap tolerance.

    The tolerance trades slivers against a perturbed mesh, and neither cost is
    knowable in advance — it depends on how the line happens to fall relative to
    this mesh's vertices. Measure it rather than guess.

    Returns a list of ``(snap_frac, info)``; entries where the cut failed carry the
    exception message under ``"error"`` instead.
    """
    out = []
    for frac in snap_fracs:
        try:
            _cut, info = cut_along_lines(dm, lines, snap_frac=frac)
            out.append((frac, info))
        except (ValueError, RuntimeError) as exc:
            # A tolerance can legitimately be unusable on a given mesh: too small
            # leaves a triple crossing, too large inverts a cell. Both are results.
            out.append((frac, {"error": str(exc)}))
    return out
