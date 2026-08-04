"""Embed a surface in a mesh by PLACING its points, not by cutting for them.

:mod:`underworld3.utilities.line_cut` embeds a surface by **splitting** every
edge the surface crosses. That works, but every restriction the cut carries is a
consequence of it: an edge can be split at exactly one point, so two flanks
running closer than one element apart compete for the *same* edge and the cut is
refused; the surface can never be finer than the local ``h``; and a triangle the
surface enters but does not leave — a fault tip — has no split that represents
it.

This module does the same job with the opposite move. It **asserts the
surface's own points**, deletes the mesh vertices in the way, and retriangulates
the cavity so that the placed segments survive as element edges. Nothing is
crossed, so nothing has to be split twice, and none of those restrictions apply:

* a tip terminates *inside* the mesh, because the tip is a placed vertex like
  any other and the cavity closes around it;
* the surface's point spacing is chosen by the caller, not by wherever the mesh
  happened to put its edges;
* two surfaces may run arbitrarily close together, because neither is competing
  for an edge of the original mesh.

The operation is the same on a curve in 2-D and on a sheet in 3-D — place,
delete, retriangulate — which is the other reason to prefer it: cutting
tetrahedra along a surface is an unsolved pattern problem, while filling a
cavity is standard meshing practice. Only the 2-D half exists here.

The three shapes of the cavity
------------------------------
Deleting the vertices in the way leaves one hole. What has to be triangulated
inside it depends only on how many ends of the surface reach the domain
boundary, and all three cases are the **same walk** between two chains:

====================  =========================================================
ends on the boundary  the region to fill
====================  =========================================================
none (a fault)        an annulus — the cavity ring outside, the surface
                      traversed out and back inside. Opened at one rung so that
                      it, too, is a pair of chains.
one (one tip inside)  a disc, its boundary the cavity ring and the out-and-back
                      traverse, the two meeting at the boundary end.
two (a crossing)      two discs, one per flank, each bounded by half the cavity
                      ring and by the surface.
====================  =========================================================

The surface is traversed **out and back**, and the two passes name the SAME
vertices. That is what makes a placed segment one edge with a cell on each side
rather than two edges with a gap between them, and it is the whole of "zero
thickness": the finite-width ribbon this construction was prototyped on is the
same walk with the two passes held apart.

Ordering the walk: the parameter
--------------------------------
The two chains have to be marched in step, which needs one parameter that orders
*both* — a point of the surface and a point of the cavity ring alike. Two
cheaper choices fail, for instructive reasons. Arc length along each chain
measures the cavity ring's wiggle rather than its progress, and the two rings
drift apart. Position along strike is discontinuous at the tips, which is
exactly where the difficulty lives.

:func:`_traverse_parameter` uses arc length around the surface's own boundary —
down one flank, **around the tip**, back up the other. It has neither problem,
because it goes round the tip for the same reason the surface does. At a
zero-thickness tip the two flanks meet at a point, so the turn through 180
degrees is given a window of its own (``cap``, one point spacing wide) and the
parameter is interpolated across it by ANGLE about the tip. That window is what
turns the tip into a **fan** — one placed vertex, many cavity vertices — instead
of a plateau on which the walk has nothing left to order by.

Scope
-----
Two dimensions, serial. Placement ADDS points, and the chart-expansion rebuild a
shared point would need does not exist yet, so a parallel call is refused rather
than silently returning a mesh whose star-forest is wrong. See
:func:`~underworld3.utilities.reconnect.rebuild_cavities`.

Several surfaces are placed **one at a time**, each against the result of the
last. The already-placed segments carry an edge label, and a labelled edge is an
interface, so a later placement will not delete a vertex out of an earlier one.
"""

import numpy as np

import underworld3 as uw
from underworld3.utilities import reconnect
from underworld3.utilities.line_cut import (CUT_LABEL, _coords,
                                            _distance_to_lines, _edge_vertices,
                                            _vertex_h, cell_areas, min_angles)


# ------------------------------------------------------------- mesh reads (2-D)

def _cells_anticlockwise(dm, X):
    """(n_cells, 3) local vertex indices of every triangle, wound anticlockwise.

    The winding is not cosmetic: :func:`_cavity_ring` reads the boundary of the
    dropped set off the cells' own directed edges, and a clockwise cell would
    contribute its edges the wrong way round and break the cancellation.
    """
    vS, vE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    cells = np.array([[int(p) - vS for p in dm.getTransitiveClosure(c)[0]
                       if vS <= p < vE] for c in range(cS, cE)],
                     dtype=np.int64).reshape(cE - cS, 3)
    P = X[cells]
    turn = ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
            - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
    cells[turn < 0] = cells[turn < 0][:, [0, 2, 1]]
    return cells


def _boundary_edges(dm):
    """Every facet of the domain boundary, as ``(edge point, va, vb)``.

    A facet with one cell in its support is on the boundary of the domain. Its
    vertices may never be deleted — that would change the domain — so they are
    the one class of vertex the placement has to work around rather than through.
    """
    vS, _vE = dm.getDepthStratum(0)
    out = []
    for e in range(*dm.getDepthStratum(1)):
        if len(dm.getSupport(e)) == 1:
            a, b = (int(v) - vS for v in dm.getCone(e))
            out.append((int(e), a, b))
    return out


def _boundary_vertices(dm, n_vertices):
    """Mask of the vertices lying on the domain boundary."""
    on = np.zeros(n_vertices, dtype=bool)
    for _e, a, b in _boundary_edges(dm):
        on[a] = on[b] = True
    return on


def _interface_vertices(dm, n_vertices):
    """Vertices carrying an edge of an already-embedded surface.

    Deleting one would put a gap in a surface placed earlier, so these are
    refused as victims. Read through :func:`reconnect._interface_edges`, which
    already knows which labels mean *interface* and which are PETSc's or UW3's
    own bookkeeping — a distinction that has produced three separate silent
    failures when guessed at.
    """
    pStart, _pEnd = dm.getChart()
    vS, _vE = dm.getDepthStratum(0)
    locked = reconnect._interface_edges(dm)
    on = np.zeros(n_vertices, dtype=bool)
    for e in range(*dm.getDepthStratum(1)):
        if locked[e - pStart]:
            for v in dm.getCone(e):
                on[int(v) - vS] = True
    return on


# ------------------------------------------------------------------ the polyline

def _arc_length(pts):
    """Segment vectors, their lengths, and the cumulative arc length."""
    seg = pts[1:] - pts[:-1]
    seglen = np.linalg.norm(seg, axis=1)
    return seg, seglen, np.concatenate([[0.0], np.cumsum(seglen)])


def _point_at(pts, a):
    """The point of the polyline at arc length ``a``."""
    _seg, seglen, cum = _arc_length(pts)
    k = int(np.clip(np.searchsorted(cum, a, side="right") - 1, 0, len(seglen) - 1))
    u = (a - cum[k]) / seglen[k]
    return pts[k] + u * (pts[k + 1] - pts[k])


def _nearest_foot(pts, X):
    """Arc length of, and signed distance to, the nearest point of the polyline.

    The sign is taken from the nearest SEGMENT's normal. Where a polyline turns
    sharply the two segments meeting at the corner disagree about the sign in
    the reflex wedge; a fault trace turns gently, and the cavity keeps its ring
    a clear element away from the corner.
    """
    seg, seglen, cum = _arc_length(pts)
    best = np.full(len(X), np.inf)
    a = np.zeros(len(X))
    d = np.zeros(len(X))
    for k in range(len(seg)):
        u = np.clip(((X - pts[k]) @ seg[k]) / (seg[k] @ seg[k]), 0.0, 1.0)
        foot = pts[k] + u[:, None] * seg[k]
        dist = np.linalg.norm(X - foot, axis=1)
        closer = dist < best
        normal = np.array([-seg[k][1], seg[k][0]]) / seglen[k]
        best[closer] = dist[closer]
        a[closer] = (cum[k] + u * seglen[k])[closer]
        d[closer] = ((X - pts[k]) @ normal)[closer]
    return a, d


def _resample(pts, spacing):
    """The polyline with every segment cut into pieces no longer than ``spacing``.

    The original control points are kept, so the geometry is exact and only the
    point DENSITY changes. Spacing is what sets the placed surface's resolution
    — under the cut that was decided for you by wherever the mesh happened to
    put its edges.
    """
    out = [pts[0]]
    for A, B in zip(pts[:-1], pts[1:]):
        span = float(np.linalg.norm(B - A))
        # A repeated control point contributes no segment, and a segment of zero
        # length has no direction: it would give the tip cap a normal of 0/0 and
        # the whole walk a parameter of NaN, silently.
        if span == 0.0:
            continue
        n = max(int(np.ceil(span / spacing)), 1)
        for i in range(1, n + 1):
            out.append(A + (i / n) * (B - A))
    return np.array(out)


def _inside_mesh(X, cells, points):
    """Which of ``points`` lie inside some cell of the mesh."""
    P = X[cells]
    v0, v1 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    inside = np.zeros(len(points), dtype=bool)
    for i, q in enumerate(points):
        w = q - P[:, 0]
        s = (w[:, 0] * v1[:, 1] - w[:, 1] * v1[:, 0]) / det
        t = (v0[:, 0] * w[:, 1] - v0[:, 1] * w[:, 0]) / det
        inside[i] = bool(((s >= 0.0) & (t >= 0.0) & (s + t <= 1.0)).any())
    return inside


def _clip_to_domain(dm, X, cells, pts):
    """The single run of the polyline that lies inside the mesh.

    A trace is normally specified with its ends outside the domain, so that it
    crosses cleanly rather than stopping a hair short of the wall. Placement
    needs the ends themselves — they become vertices — so the polyline is cut
    where it meets the boundary.

    Crossings are found against the boundary FACETS rather than against an
    assumed box, so this works on whatever domain the mesh describes. A trace
    that leaves and re-enters is refused: it is two surfaces, and they want two
    names.
    """
    _seg, seglen, cum = _arc_length(pts)
    ends = _boundary_edges(dm)

    breaks = [0.0, cum[-1]]
    for k in range(len(seglen)):
        A, d = pts[k], pts[k + 1] - pts[k]
        for _e, ia, ib in ends:
            p, e = X[ia], X[ib] - X[ia]
            det = d[0] * (-e[1]) - d[1] * (-e[0])
            if det == 0.0:
                continue                  # parallel: no single crossing point
            rhs = p - A
            t = (rhs[0] * (-e[1]) - rhs[1] * (-e[0])) / det
            u = (d[0] * rhs[1] - d[1] * rhs[0]) / det
            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                breaks.append(cum[k] + t * seglen[k])

    breaks = np.unique(np.round(breaks, 12))
    mids = np.array([_point_at(pts, 0.5 * (breaks[i] + breaks[i + 1]))
                     for i in range(len(breaks) - 1)])
    live = _inside_mesh(X, cells, mids)
    if not live.any():
        raise ValueError(
            "no part of the surface lies inside the mesh: the trace and the "
            "mesh do not overlap.")
    starts = np.flatnonzero(
        np.diff(np.concatenate([[0], live.astype(int), [0]])) == 1)
    if len(starts) > 1:
        raise ValueError(
            f"the surface leaves the domain and re-enters it, in {len(starts)} "
            "separate pieces. Each piece is a surface of its own and wants its "
            "own name; place them one at a time.")

    live_at = np.flatnonzero(live)
    # Clamped back onto the polyline: the break positions are rounded so that a
    # crossing found twice on two segments is one break, and a rounded end can
    # land a hair PAST the trace — which puts the last control point strictly
    # inside the kept run, repeats it, and leaves a zero-length final segment.
    lo = max(float(breaks[live_at[0]]), 0.0)
    hi = min(float(breaks[live_at[-1] + 1]), float(cum[-1]))
    interior = [p for a, p in zip(cum, pts) if lo < a < hi]
    return np.array([_point_at(pts, lo)] + interior + [_point_at(pts, hi)])


def _spacing_near(dm, X, cells, pts):
    """The mean cell diameter along the surface: the mesh's own resolution there."""
    from underworld3.utilities.edge_split import cell_diameters

    diameter = cell_diameters(dm)
    near = _distance_to_lines(X[cells].mean(axis=1), [pts]) < diameter
    return float(diameter[near].mean() if near.any() else diameter.mean())


# ------------------------------------------------------------------- the cavity

def _cavity_ring(cells, drop):
    """The boundary of the dropped cell set, as a closed anticlockwise ring.

    Each dropped cell contributes its directed edges; an edge whose reverse also
    belongs to a dropped cell is interior to the cavity and cancels. What is left
    traverses the hole with the cavity on its left — the orientation the walk
    needs, obtained without a single geometric test.

    ``None`` when the survivors do not leave one simple hole: a vertex appearing
    twice on the ring, or two disconnected cavities. Either means the caller has
    to widen the clearance, not that this can be patched up.
    """
    directed = {}
    for ci in drop:
        v0, v1, v2 = cells[ci]
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            directed[(int(a), int(b))] = int(ci)
    ring_edges = [(a, b) for (a, b) in directed if (b, a) not in directed]
    if not ring_edges:
        return None
    step = dict(ring_edges)
    if len(step) != len(ring_edges):
        return None                       # a vertex leaves the cavity twice
    start = min(step)
    ring, cur = [start], step[start]
    while cur != start:
        if cur not in step or len(ring) > len(step):
            return None
        ring.append(cur)
        cur = step[cur]
    return ring if len(ring) == len(step) else None


def _cells_meeting(X, cells, pts, candidates):
    """Which candidate cells a segment of the polyline enters.

    The clearance test alone is not enough. It deletes the vertices NEAR the
    surface, but a cell can be crossed by the surface while all three of its
    corners sit further away than the clearance — on an obtuse cell, or where
    the surface runs nearly parallel to an edge. Such a cell would survive and
    straddle, which is the one thing the construction exists to prevent, so the
    cells the surface passes through are added outright.
    """
    P = X[cells[candidates]]
    hit = np.zeros(len(candidates), dtype=bool)

    def side(a, b, c):
        return ((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
                - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))

    for A, B in zip(pts[:-1], pts[1:]):
        for i in range(3):
            p, q = P[:, i], P[:, (i + 1) % 3]
            # Two segments cross when each separates the other's ends. Touching
            # counts, which over-selects by at most a ring of cells and is the
            # conservative direction.
            hit |= ((side(A, B, p) * side(A, B, q) <= 0.0)
                    & (side(p, q, A) * side(p, q, B) <= 0.0))
    return np.asarray(candidates)[hit]


# ---------------------------------------------------------------- the parameter

def _traverse_parameter(pts, X, cap, cap_near, cap_far):
    """Arc length around the surface's boundary, for points off the surface.

    The surface's boundary is its two flanks and, where an end terminates inside
    the mesh, a cap joining them. The parameter runs from 0 along the flank on
    the NEGATIVE side, through the far cap, back along the positive side, and
    through the near cap. Its two properties are the ones the walk needs and
    nothing cheaper has: it is continuous across a tip, and it orders a cavity
    vertex and a surface vertex on the same scale.

    ``cap`` is the arc length allotted to a tip. Any positive value orders the
    walk correctly; one point spacing gives the tip the same share of the
    parameter as an ordinary segment, so the fan around it comes out with about
    as many triangles as a segment gets.

    Returns the parameter and the raw arc length of each foot, the latter
    because a caller that knows a point's flank from the TOPOLOGY has a better
    source for the side than the sign of a distance.
    """
    seg, seglen, cum = _arc_length(pts)
    length = cum[-1]

    a, d = _nearest_foot(pts, X)
    s = np.where(d < 0.0, a, 2.0 * length + cap * cap_far - a)

    # Beyond a tip every point has the SAME foot — the tip — so arc length says
    # nothing about where it sits around the turn. The angle about the tip does,
    # and it agrees with the flank formula at both ends of the window.
    if cap_far:
        t = seg[-1] / seglen[-1]
        n = np.array([-t[1], t[0]])
        r = X - pts[-1]
        phi = np.arctan2(r @ n, r @ t)
        s = np.where(r @ t > 0.0, length + cap * (phi + 0.5 * np.pi) / np.pi, s)
    if cap_near:
        t = seg[0] / seglen[0]
        n = np.array([-t[1], t[0]])
        r = X - pts[0]
        psi = np.arctan2(r @ n, -(r @ t))
        s = np.where(r @ t < 0.0,
                     2.0 * length + cap * cap_far
                     + cap * (0.5 * np.pi - psi) / np.pi, s)
    return s, a


def _traverse(index, pts, cap, cap_near, cap_far):
    """The surface out along one flank and back, as ``(vertex, parameter)``.

    ``index`` gives the vertex carrying each point of the polyline. A point in
    the middle of the surface appears TWICE — once per flank — as the same
    vertex; a tip inside the mesh appears once, and an end on the domain
    boundary opens the chain there.
    """
    _seg, _seglen, cum = _arc_length(pts)
    length = cum[-1]
    last = len(pts) - 1
    shift = 2.0 * length + cap * cap_far

    out = [] if cap_near else [(int(index[0]), 0.0)]
    out += [(int(index[k]), cum[k]) for k in range(1, last)]
    out.append((int(index[last]), length + 0.5 * cap if cap_far else length))
    out += [(int(index[k]), shift - cum[k]) for k in range(last - 1, 0, -1)]
    out.append((int(index[0]), shift + 0.5 * cap if cap_near else shift))
    return out


# --------------------------------------------------------------------- the walk

#: A triangle the walk may close has to have some shape to it, not merely a
#: resolvable sign. The case that forces this is a cavity reaching the domain
#: wall: the ring then runs ALONG the wall, and any triangle made of two wall
#: vertices and the surface's end on that same wall is three collinear points.
#: The orientation predicate declines an exactly-collinear triple, but a wall
#: whose vertices differ in the last bit gives a resolvable sign and an area of
#: 1e-15 — which passes every positivity test and is a cell with a zero angle.
#: Refusing it makes the walk advance along the SURFACE instead, which is the
#: triangulation that was wanted. Well below anything a real cell reaches: a
#: cell of a fifth of a degree measures 9e-3 on this scale.
_MIN_QUALITY = 1.0e-6


def _quality(P):
    """Scale-free triangle quality: 1 equilateral, 0 degenerate, <0 inverted."""
    e = np.array([P[2] - P[1], P[0] - P[2], P[1] - P[0]])
    twice_area = ((P[1, 0] - P[0, 0]) * (P[2, 1] - P[0, 1])
                  - (P[1, 1] - P[0, 1]) * (P[2, 0] - P[0, 0]))
    return float(2.0 * np.sqrt(3.0) * twice_area
                 / max(float((e ** 2).sum()), np.finfo(float).tiny))


def _usable(tri, X):
    """Whether the walk may close this triangle: right way round, and not flat."""
    return (reconnect._orient2d(X[tri[0]], X[tri[1]], X[tri[2]]) > 0
            and _quality(X[list(tri)]) > _MIN_QUALITY)


def _encloses(tri, X, points):
    """Whether any of ``points`` lies strictly inside the triangle."""
    A, B, C = X[tri[0]], X[tri[1]], X[tri[2]]
    v0, v1 = B - A, C - A
    det = v0[0] * v1[1] - v0[1] * v1[0]
    for p in points:
        if p in tri:
            continue
        w = X[p] - A
        s = (w[0] * v1[1] - w[1] * v1[0]) / det
        t = (v0[0] * w[1] - v0[1] * w[0]) / det
        if s > 0.0 and t > 0.0 and s + t < 1.0:
            return True
    return False


def _zip(outer, inner, s_out, s_in, X):
    """Triangulate the region between two chains, advancing one side at a time.

    Both chains run in increasing parameter and bound the same region, which
    lies on the left of each. Every step closes one triangle on the chain that
    is further behind, so the two are consumed in step; the other side is tried
    when the preferred one would invert. The chains may SHARE an end vertex —
    where the surface meets the domain boundary — and the triangle that would
    close there is degenerate, so it is skipped rather than emitted.

    Choosing the side by parameter rather than by shape is not a detail. With
    the choice made on shape alone one chain runs away from the other: measured
    on the ribbon prototype, the inner advanced 23 points against the outer's 10
    before the correspondence was 14 surface widths out and every triangle
    inverted.

    There is a third move, and without it the walk wedges more often. A cavity
    ring is not convex — it follows whatever cells were cleared — so a corner of
    it can protrude into the region, and neither cross-triangle at such a corner
    turns the right way. Clipping that corner off the ring as an EAR consumes
    both of its ring edges and lets the walk carry on. It is a last resort
    rather than a preference, because it advances the ring without advancing the
    surface, and doing that by choice would let the two run out of step. The ear
    is refused if it would swallow another vertex of either chain, which would
    overlap the triangles that vertex still has to be given.

    Only the ring may be clipped this way. An ear on the surface's own chain
    would drop one of its points, which is the whole thing being placed.

    Returns ``(triangles, None)``, or ``(None, vertex)`` naming the ring vertex
    the walk could not get past. That vertex is a SPIKE of surviving mesh poking
    into the cavity: the cavity wraps more than half way round it, so the only
    triangle that could fill the notch uses a vertex the walk has already
    consumed, and no forward walk can reach it. The caller's answer is to
    sacrifice the vertex — it is one the cavity should have swallowed in the
    first place — rather than to make the walk able to go backwards.
    """
    outer, s_out = list(outer), list(s_out)
    inner, s_in = list(inner), list(s_in)
    tris = []
    i = j = 0
    while i < len(outer) - 1 or j < len(inner) - 1:
        m, n = len(outer) - 1, len(inner) - 1
        prefer_outer = i < m and (j >= n or s_out[i + 1] <= s_in[j + 1])
        taken = None
        for side in (("o", "i") if prefer_outer else ("i", "o")):
            if side == "o" and i < m:
                tri = (outer[i], outer[i + 1], inner[j])
            elif side == "i" and j < n:
                tri = (outer[i], inner[j + 1], inner[j])
            else:
                continue
            if len(set(tri)) < 3:
                taken = (side, None)      # the chains meet: nothing to close
                break
            if _usable(tri, X):
                taken = (side, tri)
                break

        if taken is None:
            ear = (outer[i], outer[i + 1], outer[i + 2]) if i + 2 <= m else None
            if ear is None or not _usable(ear, X) or _encloses(
                    ear, X, outer[i:] + inner[j:]):
                return None, outer[min(i + 1, m)]
            tris.append(ear)
            del outer[i + 1], s_out[i + 1]
            continue

        side, tri = taken
        if tri is not None:
            tris.append(tri)
        if side == "o":
            i += 1
        else:
            j += 1
    return tris, None


def _opened(chain, s):
    """A closed chain cut open at its lowest parameter, ready for :func:`_zip`.

    The cut is a bridge the walk crosses twice, once at each end, so the edge it
    introduces is an ordinary interior edge with a cell on either side and not a
    seam in the result.
    """
    start = int(np.argmin(s))
    rolled = list(chain[start:]) + list(chain[:start])
    t = np.roll(np.asarray(s, dtype=float), -start) - float(np.min(s))
    return rolled + [rolled[0]], np.concatenate([t, [1.0]])


def _walk_cavity(ring, s_ring, traverse, s_in, X, corners):
    """Fill the cavity, in the one or two pieces the surface leaves of it.

    ``corners`` are the ring vertices where the surface meets the domain
    boundary, in polyline order: none for a fault, one for a fault with one tip
    inside, two for a surface crossing the domain. Returns what :func:`_zip`
    returns — the triangles, or the ring vertex the walk could not get past.
    """
    if not corners:
        outer, s_out = _opened(ring, s_ring)
        inner, t_in = _opened(traverse, s_in)
        return _zip(outer, inner, s_out, t_in, X)

    start = ring.index(corners[0])
    ring = ring[start:] + ring[:start]
    s_ring = np.concatenate([s_ring[start:], s_ring[:start]])
    # A corner sits ON the surface, where the flank test that gives every other
    # ring vertex its parameter has no side to report. Its value is known
    # exactly from the traverse instead.
    s_ring[0] = s_in[0]

    if len(corners) == 1:
        return _zip(ring + [ring[0]], traverse,
                    np.concatenate([s_ring, [s_in[-1]]]), s_in, X)

    q = ring.index(corners[1])
    j = traverse.index(corners[1])
    s_ring[q] = s_in[j]
    out = []
    for chain, s_chain, sub, t_sub in (
            (ring[:q + 1], s_ring[:q + 1], traverse[:j + 1], s_in[:j + 1]),
            (ring[q:] + [ring[0]], np.concatenate([s_ring[q:], [s_in[-1]]]),
             traverse[j:], s_in[j:])):
        piece, stuck = _zip(chain, sub, s_chain, t_sub, X)
        if piece is None:
            return None, stuck
        out += piece
    return out, None


# ------------------------------------------------------------- labels and edges

def _edge_lookup(dm):
    """``{(v0, v1): edge point}`` over the whole mesh, keyed low vertex first."""
    out = {}
    for e in range(*dm.getDepthStratum(1)):
        a, b = (int(v) for v in dm.getCone(e))
        out[(a, b) if a < b else (b, a)] = int(e)
    return out


def _label_placed_edges(dm, chain, name, value):
    """Mark the edges the surface's segments became; return how many.

    Read off the VERTEX NUMBERS the rebuild handed back rather than by geometry:
    the placed points are known exactly, so a geometric test could only weaken
    an identity that is already exact. A segment that is not an edge of the
    result is a defect, and shows up here immediately.
    """
    edge_of = _edge_lookup(dm)
    if not dm.hasLabel(name):
        dm.createLabel(name)
    label = dm.getLabel(name)
    label.setDefaultValue(0)
    for a, b in zip(chain[:-1], chain[1:]):
        e = edge_of.get((a, b) if a < b else (b, a))
        if e is None:
            raise RuntimeError(
                "a segment of the placed surface is not an edge of the result: "
                "the cavity was triangulated across the surface rather than up "
                "to it.")
        label.setValue(e, int(value))
    return len(chain) - 1


def _inherit_boundary_labels(new_dm, dm, splits):
    """Give the halves of a split boundary facet the labels the whole one had.

    Placing a surface's end on the domain boundary replaces one boundary facet
    by two. The rebuild carries labels by point id, and the facet that was split
    has no point id in the result, so its ``Left`` / ``UW_Boundaries`` / named
    boundary values would simply be dropped — leaving a hole in the wall that
    every boundary condition applied there would quietly step over.

    ``splits`` is ``(source edge point, end vertex, placed vertex, end vertex)``
    in the RESULT's numbering for the three vertices.
    """
    edge_of = _edge_lookup(new_dm)
    for source, a, v, b in splits:
        for i in range(dm.getNumLabels()):
            name = dm.getLabelName(i)
            if name in reconnect._TOPOLOGY_LABELS:
                continue
            value = dm.getLabel(name).getValue(source)
            if value < 0:
                continue                  # this label says nothing about it
            if not new_dm.hasLabel(name):
                new_dm.createLabel(name)
            target = new_dm.getLabel(name)
            for x, y in ((a, v), (v, b)):
                target.setValue(edge_of[(x, y) if x < y else (y, x)], int(value))


# -------------------------------------------------------------------- placement

def _place_one(dm, pts, label, label_value, clearance, spacing, end_snap):
    """Place one polyline into ``dm``. Returns the new mesh and its counts."""
    X = _coords(dm)
    cells = _cells_anticlockwise(dm, X)
    wall = [X[[a, b]] for _e, a, b in _boundary_edges(dm)]

    pts = _clip_to_domain(dm, X, cells, pts)
    if spacing is None:
        spacing = _spacing_near(dm, X, cells, pts)
    pts = _resample(pts, spacing)

    # An end that reaches the domain boundary becomes a vertex ON it; an end
    # inside the mesh is a tip and the cavity closes round it. Where exactly one
    # end is on the boundary the polyline is turned so that it is the FIRST,
    # which is what lets the traverse open there and stay a single chain.
    at_wall = list(_distance_to_lines(pts[[0, -1]], wall) < 1.0e-9 * spacing)
    if at_wall[1] and not at_wall[0]:
        pts, at_wall = np.ascontiguousarray(pts[::-1]), at_wall[::-1]

    dm, X, corner, split_at = _settle_ends(dm, X, pts, spacing, end_snap, at_wall)
    vS, _vE = dm.getDepthStratum(0)
    cS, _cE = dm.getHeightStratum(0)
    cells = _cells_anticlockwise(dm, X)
    on_boundary = _boundary_vertices(dm, len(X))

    # Which vertices are in the way. A domain-boundary vertex is never one:
    # deleting it would change the domain. Nor is a vertex of a surface already
    # embedded, which would put a gap in that surface.
    protected = on_boundary | _interface_vertices(dm, len(X))
    victim = ((_distance_to_lines(X, [pts])
               < clearance * _vertex_h(X, _edge_vertices(dm))) & ~protected)

    # One local index space covering the surviving mesh vertices and the placed
    # points, so that the two can appear in one triangle without a special case.
    placed_rows, index = [], np.empty(len(pts), dtype=np.int64)
    for k in range(len(pts)):
        if corner.get(k) is not None:
            index[k] = corner[k]
        else:
            index[k] = len(X) + len(placed_rows)
            placed_rows.append(pts[k])
    placed = np.array(placed_rows).reshape(-1, 2)
    Xall = np.vstack([X, placed])

    cap = spacing
    tip_near, tip_far = 0 not in corner, (len(pts) - 1) not in corner
    length = _arc_length(pts)[2][-1]
    total = 2.0 * length + cap * (tip_near + tip_far)
    traverse = _traverse(index, pts, cap, tip_near, tip_far)
    corners = [int(index[k]) for k in sorted(corner)]

    edge = np.linalg.norm(X[cells[:, 0]] - X[cells[:, 1]], axis=1)
    reachable = np.flatnonzero(
        _distance_to_lines(X[cells].mean(axis=1), [pts]) < edge + spacing)
    crossed = _cells_meeting(X, cells, pts, reachable)

    # The cavity is CLEARED and filled in one loop, because whether it is big
    # enough is not knowable in advance: the walk can only fail at a spike of
    # surviving mesh poking into it, and the answer to a spike is to swallow the
    # vertex at its point. Each round strictly grows the victim set, so this
    # terminates; the cap is against a spike the mesh will not give up — one on
    # the domain wall, or on a surface already embedded.
    for _attempt in range(8):
        drop = np.union1d(np.flatnonzero(victim[cells].any(axis=1)), crossed)
        if not len(drop):
            raise ValueError(
                "the surface meets no cell of this mesh: there is nothing to "
                "place it in.")
        ring = _cavity_ring(cells, drop)
        if ring is None:
            raise RuntimeError(
                "the cells cleared for the surface do not leave one simple "
                "hole. Raise `clearance` so the cavity is wider than the shapes "
                "pinching it.")
        if victim[ring].any():
            raise RuntimeError(
                "a deleted vertex is on the cavity boundary; the cavity is not "
                "the union of the victims' stars.")

        # An end placed part-way along a boundary facet splits it, so it belongs
        # on the cavity ring between that facet's two vertices.
        for k, (_e, a, b) in split_at.items():
            ring = _insert_on_ring(ring, a, b, int(index[k]))

        s_ring, foot = _traverse_parameter(pts, Xall[ring], cap, tip_near,
                                           tip_far)
        # With both ends on the boundary the ring is cut in two and each piece
        # belongs to a known flank, so the side comes from the TOPOLOGY rather
        # than from the sign of a distance — which a ring vertex sitting a hair
        # on the wrong side of the trace would otherwise get wrong, and with it
        # its whole position in the walk.
        if len(corners) == 2:
            s_ring = _flank_parameter(ring, foot, length, corners)

        tris, stuck = _walk_cavity(list(ring), s_ring / total,
                                   [v for v, _s in traverse],
                                   np.array([s for _v, s in traverse]) / total,
                                   Xall, corners)
        if tris is not None:
            break
        if stuck is None or stuck >= len(X) or protected[stuck]:
            raise RuntimeError(
                "the cavity could not be triangulated around the surface, and "
                "the vertex it wedged on may not be deleted — it is on the "
                "domain boundary, on a surface already embedded, or a point of "
                "this one. Move the surface off it, or raise `clearance`.")
        victim[stuck] = True
    else:
        raise RuntimeError(
            "the cavity could not be triangulated around the surface after "
            "growing it 8 times. Raise `clearance`, or place the surface's "
            "points further apart.")

    def mixed(v):
        return int(v) + vS if v < len(X) else -(int(v) - len(X) + 1)

    new_dm, point_map, placed_points = reconnect.rebuild_cavities(
        dm, np.flatnonzero(victim) + vS, drop + cS,
        [tuple(mixed(v) for v in t) for t in tris], placed)

    def new_point(v):
        return (int(placed_points[v - len(X)]) if v >= len(X)
                else int(point_map[v + vS]))

    n_facets = _label_placed_edges(new_dm, [new_point(int(v)) for v in index],
                                   label, label_value)
    _inherit_boundary_labels(
        new_dm, dm, [(e, new_point(a), new_point(int(index[k])), new_point(b))
                     for k, (e, a, b) in split_at.items()])
    return new_dm, {"n_placed": len(placed),
                    "n_on_surface": len(pts) - len(placed),
                    "n_removed": int(victim.sum()),
                    "n_surface_facets": n_facets}


def _wall_facet_at(dm, X, point, spacing):
    """The boundary facet a placed end lands on: ``(edge point, va, vb, u)``.

    ``u`` is the position along the facet, so a caller can tell an end landing
    in the middle of a facet from one landing all but on top of a vertex.
    """
    best, hit = np.inf, None
    for e, a, b in _boundary_edges(dm):
        d = X[b] - X[a]
        u = float(np.clip(((point - X[a]) @ d) / (d @ d), 0.0, 1.0))
        away = float(np.linalg.norm(point - (X[a] + u * d)))
        if away < best:
            best, hit = away, (e, a, b, u)
    if best > 1.0e-9 * spacing:
        raise RuntimeError(
            "the surface's end was clipped to the domain boundary but lies on "
            "no boundary facet; the trace and the mesh geometry disagree.")
    return hit


def _wall_is_straight(dm, X, v, spacing):
    """Whether the domain boundary runs straight through vertex ``v``.

    A vertex on a straight run of wall may be slid ALONG the wall without
    changing the domain at all. A vertex where the wall turns may not: moving it
    would move the corner, so it is left where it is and the surface's end is
    placed beside it instead.
    """
    facets = [(a, b) for _e, a, b in _boundary_edges(dm) if v in (a, b)]
    if len(facets) != 2:
        return False
    directions = []
    for a, b in facets:
        other = b if a == v else a
        d = X[other] - X[v]
        directions.append(d / np.linalg.norm(d))
    cross = abs(directions[0][0] * directions[1][1]
                - directions[0][1] * directions[1][0])
    return bool(cross < 1.0e-9)


def _settle_ends(dm, X, pts, spacing, end_snap, at_wall):
    """Give every end that reaches the wall a mesh vertex, MOVING THE MESH.

    An end on the domain boundary has to be a vertex of the result. Placing one
    a hair from an existing boundary vertex would carve a sliver against the
    wall that no later pass may repair, since a boundary vertex can never be
    deleted — so an end landing within ``end_snap`` of a facet's end slides that
    vertex along the wall onto it instead.

    Moving the mesh rather than the surface is the same choice
    :func:`~underworld3.utilities.line_cut.pull_vertex_onto` makes, for the same
    reason: the surface's position is a design variable in an outer
    optimisation, and it has to end up where it was asked for. The cost is mesh
    displacement of at most half a boundary facet, and it is paid only where the
    wall is locally straight, so the domain itself is never deformed.

    Returns the working mesh (a copy, when a vertex moved), its coordinates, the
    vertex carrying each end, and the facets an end has to be inserted into.
    """
    from underworld3.utilities.line_cut import _set_coordinates

    corner, split_at, moves = {}, {}, {}
    for k, here in ((0, at_wall[0]), (len(pts) - 1, at_wall[1])):
        if not here:
            continue                      # a tip: the cavity closes round it
        e, a, b, u = _wall_facet_at(dm, X, pts[k], spacing)
        # Put the end exactly ON the facet. Clipping solves a 2x2 system, so it
        # lands within rounding of the wall rather than on it, and a vertex a
        # picometre outside the domain is still outside it.
        pts[k] = X[a] + u * (X[b] - X[a])
        near = a if u < 0.5 else b
        if np.linalg.norm(X[near] - pts[k]) < 1.0e-9 * spacing:
            corner[k] = near              # already exactly there
        elif min(u, 1.0 - u) < end_snap and _wall_is_straight(dm, X, near, spacing):
            corner[k] = near
            moves[near] = pts[k]
        else:
            corner[k] = None
            split_at[k] = (e, a, b)

    if not moves:
        return dm, X, corner, split_at
    work = dm.clone()
    X = X.copy()
    for v, target in moves.items():
        X[v] = target
    _set_coordinates(work, np.array(sorted(moves), dtype=np.int64),
                     X[sorted(moves)])
    return work, X, corner, split_at


def _insert_on_ring(ring, a, b, v):
    """Put ``v`` between the consecutive ring vertices ``a`` and ``b``."""
    for i, pair in enumerate(zip(ring, ring[1:] + ring[:1])):
        if set(pair) == {a, b}:
            return ring[:i + 1] + [v] + ring[i + 1:]
    raise RuntimeError(
        "the boundary facet the surface ends on is not on the cavity ring, so "
        "the cell holding it was not cleared. Raise `clearance`.")


def _flank_parameter(ring, foot, length, corners):
    """Re-derive the ring's parameter from which flank each piece of it is on.

    Only for a surface crossing the domain, where the ring genuinely divides:
    the two corners cut it in two, so every vertex of a piece is on one flank
    whatever the sign of its distance to the trace happens to say. Reading the
    side off the topology instead is what stops a ring vertex sitting a hair on
    the wrong side of the trace from being given the far flank's parameter, and
    with it the wrong place in the walk entirely.
    """
    first = np.zeros(len(ring), dtype=bool)
    i, stop = ring.index(corners[0]), ring.index(corners[1])
    while True:
        first[i] = True
        if i == stop:
            break
        i = (i + 1) % len(ring)
    return np.where(first, foot, 2.0 * length - foot)


def place_along_lines(dm, lines, label=CUT_LABEL, label_value=1,
                      clearance=0.55, spacing=None, end_snap=0.25,
                      verbose=False):
    """Embed surfaces in a mesh by placing their points and refilling the hole.

    The surface's own points become mesh vertices, the mesh vertices in the way
    are deleted, and the cavity is retriangulated so that every segment of the
    surface is an element edge carrying ``label``. Nothing is split, so — unlike
    :func:`~underworld3.utilities.line_cut.cut_along_lines` — a surface may end
    INSIDE the mesh, may be finer than the local ``h``, and may run alongside
    another surface at any separation.

    Parameters
    ----------
    dm : PETSc.DMPlex
        A 2-D simplex mesh. **Not modified** — the result is a new mesh, so a
        surface can be moved and re-placed against the same fixed base.
    lines : sequence of array_like
        One or more polylines, each an ``(N, 2)`` array of points. Ends outside
        the mesh are clipped to the boundary, so the usual "specify it a little
        long" convention works. They are placed **one at a time**, each into the
        result of the last, and must not intersect one another.
    label, label_value : str, int
        Name and stratum value of the label put on the surface's edges. The
        default is the one :mod:`~underworld3.utilities.line_cut` uses, so a
        downstream pass — ``relax(pin_bands=...)``, the reconnection passes —
        does not have to know which routine embedded the surface.
    clearance : float
        Delete a mesh vertex within this multiple of its own local ``h`` of the
        surface. This is what buys room for the placed points: too small and the
        cavity is too tight to triangulate around them, too large and more of
        the mesh is rebuilt than needs to be.
    spacing : float or None
        Distance between placed points along the surface. ``None`` takes the
        mean cell diameter near the surface, putting the surface's resolution
        where the mesh's already is. This is the knob the cut does not have: a
        placed surface may be finer than the mesh it is placed in.
    end_snap : float
        An end reaching the domain boundary within this fraction of a boundary
        facet's length from one of its vertices slides that vertex ALONG the
        wall onto the end, rather than placing a second vertex a hair from it. A
        boundary vertex can never be deleted, so that sliver is one no later
        pass could repair. The surface still ends exactly where it was asked to
        — the mesh moves, not the surface — and the slide is refused where the
        wall turns, so the domain is never deformed.
    verbose : bool
        Report the counts and the worst cell of the result.

    Returns
    -------
    placed : PETSc.DMPlex
        A new mesh in which every segment of every surface is an edge carrying
        ``label``.
    info : dict
        ``n_placed`` vertices this created on the surfaces, ``n_on_surface``
        existing vertices it reused, ``n_removed`` vertices it deleted,
        ``n_surface_facets`` edges labelled, and ``min_area`` and ``min_angle``
        of the result.

        One surface is one chain, so its facets number one fewer than the
        vertices along it — the same identity the cut reports, with the placed
        count standing where the split count stood:

        ``n_surface_facets == n_placed + n_on_surface - len(lines)``.

    Raises
    ------
    ValueError
        If a surface does not overlap the mesh, or leaves the domain and
        re-enters it.
    RuntimeError
        If the cleared cells do not leave one simple hole, or the cavity cannot
        be triangulated around the surface without inverting a cell. Both mean
        ``clearance`` is too small for the surface asked for.
    NotImplementedError
        In 3-D, or in parallel.

    Examples
    --------
    A fault that stops inside the mesh — the case the cut refuses outright:

    >>> tip = numpy.array([[0.2, 0.5], [0.6, 0.55]])
    >>> placed, info = place_along_lines(mesh.dm, [tip], label="Fault")
    >>> info["n_surface_facets"] == info["n_placed"] + info["n_on_surface"] - 1
    True

    See Also
    --------
    underworld3.utilities.line_cut.cut_along_lines : the same job by splitting
        the edges the surface crosses.
    underworld3.utilities.reconnect.remove_vertices : the repair pass that
        cleans up what the walk leaves, and which will not touch a labelled edge.
    """
    if dm.getDimension() != 2:
        raise NotImplementedError(
            f"place_along_lines is 2-D; this mesh is {dm.getDimension()}-D. The "
            "cavity of a placed sheet is a polyhedron, and filling one can need "
            "Steiner points that the 2-D walk never has to invent.")
    if uw.mpi.size > 1:
        raise NotImplementedError(
            "place_along_lines is serial. Placement ADDS points, so a surface "
            "crossing a partition seam needs the star-forest's leaf set "
            "extended — the chart-expansion rebuild, which does not exist yet. "
            "Refusing beats returning a mesh whose star-forest is wrong.")

    out = dm
    totals = {"n_placed": 0, "n_on_surface": 0, "n_removed": 0,
              "n_surface_facets": 0}
    for pts in lines:
        out, one = _place_one(out, np.asarray(pts, dtype=float)[:, :2],
                              label, label_value, clearance, spacing, end_snap)
        for key in totals:
            totals[key] += one[key]

    areas = cell_areas(out)
    over = sum(1 for f in range(*out.getHeightStratum(1))
               if len(out.getSupport(f)) > 2)
    if over:
        raise RuntimeError(
            f"{over} facet(s) of the result have more than two cells: the "
            "retriangulated cavity is not conforming.")
    if (areas <= 0.0).any():
        raise RuntimeError(
            f"{int((areas <= 0.0).sum())} cell(s) of the result are inverted.")

    info = dict(totals, min_area=float(areas.min()),
                min_angle=float(min_angles(out).min()))
    if verbose:
        uw.pprint(f"[place {label!r}] placed {info['n_placed']} vertices, "
                  f"reused {info['n_on_surface']}, removed {info['n_removed']}; "
                  f"{info['n_surface_facets']} surface facets, min angle "
                  f"{info['min_angle']:.2f} deg")
    return out, info
