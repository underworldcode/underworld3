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
  happened to put its edges.

Not, however, two surfaces closer together than a cell. That was claimed here
once and it was wrong: the second surface's cavity was consuming the first, and
the identity being asserted summed each placement's own counts and so could not
see it. A cavity is cleared and filled for ONE surface, and the cells carrying
an earlier surface's facets are held back so that surface survives — which
eventually leaves no room. Measured on a 1/16 box: placing one at a time accepts
1.5 h separation and refuses 1.0 h, where the cut accepts 1.0 h and refuses
0.5 h. The two limits are not the same KIND of limit — the cut's is inherent
(converging flanks cross one edge, and an edge splits at one point) while this
one is an implementation limit, lifted by placing both surfaces into a single
cavity, which is the finite-width ribbon and is not built.

The operation is the same on a curve in 2-D and on a sheet in 3-D — place,
delete, retriangulate — which is the other reason to prefer it: cutting
tetrahedra along a surface is an unsolved pattern problem, while filling a
cavity is standard meshing practice. Only the 2-D half exists here.

The fill is delegated to gmsh, and gated
----------------------------------------
Deleting the vertices in the way leaves one hole, bounded by the cavity ring.
The ring goes to gmsh as a discrete curve carrying its existing segmentation;
the surface's chain goes as a second discrete curve **embedded** in the fill,
its end nodes shared with the ring where the surface meets the cavity
boundary — a crossing, or an end at the wall — and free otherwise, which is a
tip. gmsh's constrained boundary recovery must return both curves VERBATIM,
and the call is gated, not trusted: zero moved nodes, every input segment an
edge of the triangulation, nothing inverted — refused rather than accepted
degraded. (Measured basis:
``~/+Simulations/mesh_reconnection_study/gmsh_2d_fill_spike.py`` — boundary,
sub-h hole and free-end embed at once — and ``gmsh_2d_ends_spike.py`` — the
crossing and mixed end-on-boundary cases.)

A hand-rolled walk filled this cavity for one development generation — an
arc-length parameter around the surface's boundary, an angle-interpolated fan
at each tip, an ear-clipping third move. It worked, and it was retired the day
the 3-D sheet proved the fill could be delegated: one fill mechanism for every
dimension beats two, and the tip fan, the walk's hardest case, is gmsh's
ordinary free-end embed.

Scope
-----
Serial. Placement ADDS points, and the chart-expansion rebuild a shared point
would need does not exist yet, so a parallel call is refused rather than
silently returning a mesh whose star-forest is wrong. The parallel route is
gather-first — redistribute so the fault's star is rank-interior, operate
locally, renumber — at which point no placed point is ever shared. See
:func:`~underworld3.utilities.reconnect.rebuild_cavities`.

Both dimensions, one fill. The 2-D curve (:func:`place_along_lines`) and the
3-D sheet (:func:`place_sheet`) both delegate their cavity fill to gmsh —
constrained triangulation with Steiner insertion is exactly what a mesh
generator is for — and both gate the delegation rather than trust it: every
constraint checked bit-identical, every constraint segment or triangle checked
present in the fill, conformity and orientation checked on every call
(measured basis: ``sheet_cavity_spike.py`` 6/6 and the 2-D fill spikes, in
``~/+Simulations/mesh_reconnection_study/``).

Several surfaces are placed **one at a time**, each against the result of the
last. The already-placed segments carry an edge label, and a labelled edge is an
interface, so a later placement will not delete a vertex out of an earlier one.
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

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


def _interface_vertices_and_cells(dm, n_vertices, n_cells):
    """Vertices and cells that an already-embedded surface may not lose.

    Two masks, and the second is the one that is easy to forget. Protecting the
    interface's VERTICES from deletion does not protect its FACETS: a cell is
    not a vertex, and both cells supporting an interface facet can be cleared
    while every one of their corners is protected. The facet then has no support
    left, the refill has no reason to recreate that edge, and the earlier
    surface loses a facet out of the middle of its chain — silently. Measured on
    a T junction: a trunk of 21 facets came back with 20 and nothing raised.

    Read through :func:`reconnect._interface_edges`, which already knows which
    labels mean *interface* and which are PETSc's or UW3's own bookkeeping — a
    distinction that has produced three separate silent failures when guessed at.
    Narrowed here to INTERIOR facets, which is what "a surface already embedded"
    means: the domain's own walls carry edge labels too, and holding their cells
    would forbid clearing anything against a wall — which is exactly what a
    surface crossing the domain has to do.
    """
    pStart, _pEnd = dm.getChart()
    vS, _vE = dm.getDepthStratum(0)
    cS, _cE = dm.getHeightStratum(0)
    locked = reconnect._interface_edges(dm)
    vertices = np.zeros(n_vertices, dtype=bool)
    cells = np.zeros(n_cells, dtype=bool)
    for e in _interior_interface_facets(dm, locked, pStart):
        for v in dm.getCone(e):
            vertices[int(v) - vS] = True
        for c in dm.getSupport(e):
            cells[int(c) - cS] = True
    return vertices, cells


def _interior_interface_facets(dm, locked, pStart):
    """The labelled facets that lie INSIDE the mesh — an embedded surface."""
    return [e for e in range(*dm.getDepthStratum(1))
            if locked[e - pStart] and len(dm.getSupport(e)) == 2]


def _interface_facet_counts(dm):
    """How many INTERIOR facets each label holds, so a breach can be detected.

    Interior only, for the reason :func:`_interface_vertices_and_cells` gives:
    a wall label's count legitimately changes when a surface's end splits a wall
    facet, and comparing that would refuse the ordinary case.
    """
    pStart, _pEnd = dm.getChart()
    locked = reconnect._interface_edges(dm)
    interior = set(_interior_interface_facets(dm, locked, pStart))
    counts = {}
    for i in range(dm.getNumLabels()):
        name = dm.getLabelName(i)
        if name in reconnect._TOPOLOGY_LABELS:
            continue
        label = dm.getLabel(name)
        values = label.getValueIS()
        if values is None:
            continue
        for val in values.getIndices():
            if label.getStratumSize(int(val)) == 0:
                continue
            held = [p for p in label.getStratumIS(int(val)).getIndices()
                    if int(p) in interior]
            if held:
                counts[(name, int(val))] = len(held)
    return counts


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
        # A repeated control point contributes no segment, and a segment of
        # zero length would put two coincident chain nodes into the fill —
        # a degenerate constraint edge gmsh has no valid way to honour.
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
    traverses the hole with the cavity on its left — anticlockwise, the loop
    orientation the fill hands to gmsh, obtained without a single geometric
    test.

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


# --------------------------------------------------------------------- the fill

def _inside_polygon(P, q):
    """Whether ``q`` is STRICTLY inside the polygon ``P`` (on an edge is not).

    Ray crossing, with the strictness the fill's precondition needs: a chain
    point exactly on the cavity boundary is as unplaceable as one beyond it.
    """
    x, y = float(q[0]), float(q[1])
    inside = False
    for (x0, y0), (x1, y1) in zip(P, np.vstack([P[1:], P[:1]])):
        # On-edge: collinear and within the segment's span.
        cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        if (abs(cross) < 1e-14 * max(abs(x1 - x0) + abs(y1 - y0), 1e-300)
                and min(x0, x1) - 1e-14 <= x <= max(x0, x1) + 1e-14
                and min(y0, y1) - 1e-14 <= y <= max(y0, y1) + 1e-14):
            return False
        if (y0 > y) != (y1 > y):
            xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xi:
                inside = not inside
    return inside


def _gmsh_fill_2d(Xall, ring, chain, holes=()):
    """Triangulate the cavity with gmsh: the ring verbatim, the chain embedded.

    The ring — the cavity boundary, anticlockwise — goes in as a discrete
    curve carrying its existing segmentation; the surface's chain as a second
    discrete curve embedded in the plane surface. A chain end that IS a ring
    vertex (a crossing, or an end on the wall) is expressed by having the
    chain's elements reference the ring's own node tag — no duplicate node, no
    snapping — and a free end is a tip, gmsh's ordinary free-end embed.

    ``holes`` are further closed loops of ``Xall`` indices excluded from the
    fill — the 2-D thin volume's skin, meshed elsewhere and sewn on — each a
    discrete curve of its own, verbatim like the ring.

    Everything is gated, because a fill that looks plausible and is not
    conforming is worse than a refusal: zero moved nodes, every input segment
    an edge of the triangulation, triangles present and returned anticlockwise.

    Returns ``(tris, extra)``: triangles indexing ``Xall`` first and then the
    ``extra`` interior points gmsh inserted (rare at cavity sizes, legal
    always — they become ordinary mesh vertices).
    """
    import gmsh

    ring = [int(v) for v in ring]
    chain = [int(v) for v in (chain if chain is not None else [])]
    holes = [[int(v) for v in loop] for loop in holes]
    if len(set(chain)) != len(chain):
        raise RuntimeError("the surface's chain repeats a vertex")
    tag_of = {v: i + 1 for i, v in enumerate(ring)}
    nxt = len(ring) + 1
    hole_nodes = []
    for loop in holes:
        for v in loop:
            if v in tag_of:
                raise RuntimeError("a hole loop shares a vertex with the "
                                   "cavity boundary; the cavity is too tight")
            tag_of[v] = nxt
            hole_nodes.append(v)
            nxt += 1
    interior = [v for v in chain if v not in tag_of]
    for v in interior:
        tag_of[v] = nxt
        nxt += 1
    n_known = nxt - 1

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("cavity2d")

        def discrete_loop(loop_verts):
            tag = gmsh.model.addDiscreteEntity(1)
            gmsh.model.mesh.addNodes(
                1, tag, [tag_of[v] for v in loop_verts],
                np.column_stack([Xall[loop_verts],
                                 np.zeros(len(loop_verts))])
                .reshape(-1).tolist())
            seg = np.array([[tag_of[a], tag_of[b]] for a, b in
                            zip(loop_verts, loop_verts[1:] + loop_verts[:1])],
                           dtype=np.int64)
            gmsh.model.mesh.addElementsByType(tag, 1, [],
                                              seg.reshape(-1).tolist())
            return tag

        ring_tag = discrete_loop(ring)
        hole_tags = [discrete_loop(loop) for loop in holes]

        line_tag = None
        if chain:
            line_tag = gmsh.model.addDiscreteEntity(1)
            if interior:
                gmsh.model.mesh.addNodes(
                    1, line_tag, [tag_of[v] for v in interior],
                    np.column_stack([Xall[interior],
                                     np.zeros(len(interior))])
                    .reshape(-1).tolist())
            cseg = np.array([[tag_of[a], tag_of[b]]
                             for a, b in zip(chain[:-1], chain[1:])],
                            dtype=np.int64)
            gmsh.model.mesh.addElementsByType(line_tag, 1, [],
                                              cseg.reshape(-1).tolist())

        loops = [gmsh.model.geo.addCurveLoop([ring_tag])]
        loops += [gmsh.model.geo.addCurveLoop([t]) for t in hole_tags]
        surf = gmsh.model.geo.addPlaneSurface(loops)
        gmsh.model.geo.synchronize()
        if line_tag is not None:
            gmsh.model.mesh.embed(1, [line_tag], 2, surf)

        # Sizes bracketing what is already there: fine enough to accept the
        # chain's own spacing, coarse enough not to refine the cavity beyond
        # the surviving mesh around it.
        constrained = (ring + ring[:1], chain,
                       *[loop + loop[:1] for loop in holes])
        lengths = np.concatenate(
            [np.linalg.norm(np.diff(Xall[c], axis=0), axis=1)
             for c in constrained if len(c) > 1])
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5 * float(lengths.min()))
        gmsh.option.setNumber("Mesh.MeshSizeMax", 2.0 * float(lengths.max()))
        gmsh.model.mesh.generate(2)

        out_tags, xyz, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(xyz).reshape(-1, 3)
        row = {int(t): i for i, t in enumerate(out_tags)}

        moved = sum(1 for v, t in tag_of.items()
                    if not np.array_equal(xyz[row[t], :2], Xall[v]))
        if moved:
            raise RuntimeError(
                f"gmsh moved {moved} constrained node(s) of the cavity fill; "
                "the fill cannot be sewn back and is refused.")

        back = {t: v for v, t in tag_of.items()}
        extra_tags = sorted(int(t) for t in out_tags if int(t) > n_known)
        for j, t in enumerate(extra_tags):
            back[t] = len(Xall) + j
        extra = (xyz[[row[t] for t in extra_tags]][:, :2]
                 if extra_tags else np.empty((0, 2)))

        tris = None
        for et, nodes in zip(*[gmsh.model.mesh.getElements(2, surf)[i]
                               for i in (0, 2)]):
            if et == 2:
                tris = np.array([back[int(x)] for x in nodes],
                                dtype=np.int64).reshape(-1, 3)
        if tris is None or not len(tris):
            raise RuntimeError("gmsh produced no triangles for the cavity")

        edges = set()
        for a, b, c in tris:
            for e in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
                edges.add((min(e), max(e)))
        wanted = list(zip(ring, ring[1:] + ring[:1]))
        wanted += list(zip(chain, chain[1:]))
        for loop in holes:
            wanted += list(zip(loop, loop[1:] + loop[:1]))
        missing = [(a, b) for a, b in wanted
                   if (min(a, b), max(a, b)) not in edges]
        if missing:
            raise RuntimeError(
                f"{len(missing)} constraint segment(s) are not edges of the "
                "fill: gmsh did not honour the cavity's boundary or the "
                "surface.")

        Xext = np.vstack([Xall, extra])
        P = Xext[tris]
        cw = ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
              - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0])) < 0.0
        tris[cw] = tris[cw][:, [0, 2, 1]]
        return [tuple(int(v) for v in t) for t in tris], extra
    finally:
        gmsh.finalize()


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
    held_v, held_c = _interface_vertices_and_cells(dm, len(X), len(cells))
    held_counts = _interface_facet_counts(dm)
    # A vertex may only be deleted if every cell of its star may be cleared.
    # Held cells stay, so a vertex beside a surface already embedded would end
    # up deleted while still on the cavity's boundary — a cavity that is not
    # the union of its victims' stars, whose ring would name a deleted vertex.
    beside_held = np.zeros(len(X), dtype=bool)
    beside_held[cells[held_c].ravel()] = True
    protected = on_boundary | held_v | beside_held
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

    edge = np.linalg.norm(X[cells[:, 0]] - X[cells[:, 1]], axis=1)
    reachable = np.flatnonzero(
        _distance_to_lines(X[cells].mean(axis=1), [pts]) < edge + spacing)
    crossed = _cells_meeting(X, cells, pts, reachable)

    drop = np.union1d(np.flatnonzero(victim[cells].any(axis=1)), crossed)
    # A cell owning a facet of a surface already embedded is never cleared.
    # Clearing BOTH cells of such a facet destroys it — the facet's support
    # is gone, so the refill has no reason to recreate that edge — and the
    # earlier surface loses a facet out of the middle of its chain without
    # anything raising. Holding the cells instead stops the cavity at the
    # earlier surface, which is also what makes the ligament of an offset
    # junction survive.
    drop = drop[~held_c[drop]]
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

    # Every chain point that is not itself a ring node must lie strictly
    # inside the cavity. One on or beyond the ring means the cavity was
    # stopped short of the surface — cells held for a surface already
    # embedded, or a clearance too small — and gmsh would be handed a
    # constraint it can only satisfy by moving nodes, which the fill gate
    # refuses without saying why. Refuse here, with the cause.
    ring_set = set(int(v) for v in ring)
    outside = [k for k in range(len(pts))
               if int(index[k]) not in ring_set
               and not _inside_polygon(Xall[ring], Xall[int(index[k])])]
    if outside:
        raise RuntimeError(
            f"{len(outside)} point(s) of the surface fall outside the cavity "
            "cleared for it: the cavity was stopped by cells held for a "
            "surface already embedded, or the clearance is too small. "
            "Surfaces must be separated by at least a cell when placed one "
            "at a time.")

    tris, extra = _gmsh_fill_2d(Xall, ring, [int(v) for v in index])
    # Interior points the fill inserted become ordinary mesh vertices; they are
    # counted apart from the surface's own so the chain identity
    # (facets == points - 1) stays readable in the result.
    n_chain_placed = len(placed)
    if len(extra):
        placed = np.vstack([placed, extra])
        Xall = np.vstack([Xall, extra])

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

    # Every surface already embedded must come through with the facets it had.
    # Held cells are what makes that true; this is the check that it IS true,
    # and it is not redundant — losing a facet of an earlier surface produces a
    # mesh that passes conformity, area and orientation, and is wrong only in
    # the one place nothing else looks.
    after = _interface_facet_counts(new_dm)
    for key, before in held_counts.items():
        now = after.get(key, 0)
        # The label being written may GROW — several polylines may share a name,
        # which is how a fault with more than one segment is labelled. Any other
        # label must come through with exactly the facets it had.
        if now < before or (now != before and key != (label, int(label_value))):
            raise RuntimeError(
                f"placing {label!r} would leave the surface {key[0]!r} with "
                f"{now} facets instead of {before}: this surface's cavity "
                "reached one already embedded. Surfaces must be separated by "
                "at least a cell when placed one at a time; two that run closer "
                "than that have to be placed together, into one cavity, which "
                "is not yet implemented.")

    return new_dm, {"n_placed": n_chain_placed,
                    "n_on_surface": len(pts) - n_chain_placed,
                    "n_removed": int(victim.sum()),
                    "n_fill_points": len(extra),
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
        ``n_fill_points`` interior vertices the fill inserted (not on any
        surface), ``n_surface_facets`` edges labelled, and ``min_area`` and
        ``min_angle`` of the result.

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
        If the cleared cells do not leave one simple hole, or the gated gmsh
        fill is refused — a constrained node moved, or an input segment did
        not survive as an edge. Both mean ``clearance`` is too small for the
        surface asked for.
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
        cleans up after a placement, and which will not touch a labelled edge.
    """
    if dm.getDimension() != 2:
        raise NotImplementedError(
            f"place_along_lines takes polylines in a 2-D mesh; this mesh is "
            f"{dm.getDimension()}-D. A surface in a 3-D mesh is a sheet: use "
            "place_sheet.")
    if uw.mpi.size > 1:
        raise NotImplementedError(
            "place_along_lines is serial. Placement ADDS points, so a surface "
            "crossing a partition seam needs the star-forest's leaf set "
            "extended — the chart-expansion rebuild, which does not exist yet. "
            "Refusing beats returning a mesh whose star-forest is wrong.")

    out = dm
    totals = {"n_placed": 0, "n_on_surface": 0, "n_removed": 0,
              "n_fill_points": 0, "n_surface_facets": 0}
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


# ===========================================================================
# The 3-D placed sheet
# ===========================================================================

def _tet_vertices(dm):
    """(n_cells, 4) local vertex indices of every tetrahedron."""
    vS, vE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    return np.array([[int(p) - vS for p in dm.getTransitiveClosure(c)[0]
                      if vS <= p < vE] for c in range(cS, cE)],
                    dtype=np.int64).reshape(cE - cS, 4)


def _face_vertex_triple(dm, f, vS):
    return tuple(sorted(int(p) - vS for p in dm.getTransitiveClosure(f)[0]
                        if int(p) >= vS and dm.getPointDepth(int(p)) == 0))


def _interface_faces_3d(dm):
    """Interior faces carrying a non-topology label — an embedded surface.

    The 3-D analogue of :func:`reconnect._interface_edges`: in 3-D an
    interface is a surface and is identified by its FACES. Cell labels are
    volumes, not interfaces, and are skipped for the same reason as in 2-D;
    exterior faces are the domain's own walls and are handled separately.
    """
    fS, fE = dm.getHeightStratum(1)
    cS, cE = dm.getHeightStratum(0)
    out = set()
    for i in range(dm.getNumLabels()):
        name = dm.getLabelName(i)
        if name in reconnect._TOPOLOGY_LABELS:
            continue
        label = dm.getLabel(name)
        values = label.getValueIS()
        if values is None:
            continue
        for val in values.getIndices():
            if label.getStratumSize(int(val)) == 0:
                continue
            idx = np.asarray(label.getStratumIS(int(val)).getIndices(),
                             dtype=np.int64)
            if ((idx >= cS) & (idx < cE)).any():
                continue                  # a volume label, not an interface
            for p in idx[(idx >= fS) & (idx < fE)]:
                if len(dm.getSupport(int(p))) == 2:
                    out.add(int(p))
    return out


def _interior_face_counts_3d(dm):
    """{(label, value): count of interior faces} — the breach detector."""
    interface = _interface_faces_3d(dm)
    fS, fE = dm.getHeightStratum(1)
    counts = {}
    for i in range(dm.getNumLabels()):
        name = dm.getLabelName(i)
        if name in reconnect._TOPOLOGY_LABELS:
            continue
        label = dm.getLabel(name)
        values = label.getValueIS()
        if values is None:
            continue
        for val in values.getIndices():
            if label.getStratumSize(int(val)) == 0:
                continue
            held = [p for p in label.getStratumIS(int(val)).getIndices()
                    if int(p) in interface]
            if held:
                counts[(name, int(val))] = len(held)
    return counts


def _sheet_distance(X, pts, tris):
    """Distance from each point of ``X`` to a triangulated sheet.

    Exact point-to-triangle distance, looped over the sheet's triangles and
    vectorised over the query points — the sheet is small (hundreds of
    triangles) and the mesh is what is large.
    """
    best = np.full(len(X), np.inf)
    for t in tris:
        A, B, C = pts[t[0]], pts[t[1]], pts[t[2]]
        ab, ac = B - A, C - A
        n = np.cross(ab, ac)
        nn = float(n @ n)
        rel = X - A
        # Barycentric coordinates of the plane projection.
        d00, d01, d11 = float(ab @ ab), float(ab @ ac), float(ac @ ac)
        d20, d21 = rel @ ab, rel @ ac
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        inside = (v >= 0.0) & (w >= 0.0) & (v + w <= 1.0)
        d_plane = np.abs(rel @ n) / np.sqrt(nn)
        # Outside the triangle: distance to the nearest of its three edges.
        d_edge = np.full(len(X), np.inf)
        for P, Q in ((A, B), (B, C), (C, A)):
            e = Q - P
            u = np.clip(((X - P) @ e) / float(e @ e), 0.0, 1.0)
            d_edge = np.minimum(
                d_edge, np.linalg.norm(X - (P + u[:, None] * e), axis=1))
        best = np.minimum(best, np.where(inside, d_plane, d_edge))
    return best


def _propagate_vertex(dm, chart_values, mpi_op, np_combine):
    """Reconcile a chart-length per-point array over the point star-forest.

    Reduce leaf-to-root with ``mpi_op``, broadcast root-to-leaf, combine with
    ``np_combine`` — after which every rank holds the same value for every
    point it can see. The pattern is the contact stream's ``propagate`` (taken
    from feature/fault-split-node c8693579, recorded in the ledger); it is
    what makes marking a pure function of the GLOBAL mesh rather than of the
    partition. COLLECTIVE; a rank sharing nothing still participates.
    """
    if uw.mpi.size == 1:
        return chart_values
    sf = dm.getPointSF()
    try:
        _n, ilocal, _ir = sf.getGraph()
    except (ValueError, TypeError):
        return chart_values
    tmp = chart_values.copy()
    sf.reduceBegin(MPI._typedict[chart_values.dtype.char], tmp,
                   chart_values, mpi_op)
    sf.reduceEnd(MPI._typedict[chart_values.dtype.char], tmp,
                 chart_values, mpi_op)
    out = chart_values.copy()
    sf.bcastBegin(MPI._typedict[chart_values.dtype.char], chart_values, out,
                  MPI.REPLACE)
    sf.bcastEnd(MPI._typedict[chart_values.dtype.char], chart_values, out,
                MPI.REPLACE)
    return np_combine(chart_values, out)


def _shared_point_flags(dm):
    """Chart-length 0/1 flags for points held by more than one rank."""
    return reconnect._shared_points(dm)


def _vertex_h_3d(dm, cells, n_vertices):
    """Per-vertex local h — the min incident cell diameter — SF-reconciled.

    A seam vertex sees only its local cells, so without the reconciliation two
    ranks disagree about its h, and with it about which vertices are victims:
    the marking must be a function of the mesh, not of the partition.
    """
    from underworld3.utilities.edge_split import cell_diameters

    # The gather can leave a rank with NO cells, and `cell_diameters` raises
    # on one (empty `ends` is 1-D). An empty rank contributes the identity of
    # every reduction, never a raise — the 2-D module's parallel discipline.
    h_cell = cell_diameters(dm) if len(cells) else np.zeros(0)
    vS, _vE = dm.getDepthStratum(0)
    pStart, pEnd = dm.getChart()
    work = np.full(pEnd - pStart, np.inf)
    for c, tet in enumerate(cells):
        idx = tet + vS - pStart
        work[idx] = np.minimum(work[idx], h_cell[c])
    work = _propagate_vertex(dm, work, MPI.MIN, np.minimum)
    return work[vS - pStart: vS - pStart + n_vertices], h_cell


def _true_wall_vertex_mask(dm, n_vertices):
    """Vertices of the DOMAIN boundary — not of a partition seam.

    On a distributed mesh a seam face also has local support 1, so "support
    == 1" alone misclassifies the seam as wall and would protect (and worse,
    trust) the wrong vertices. A wall face is support 1 AND unshared; the
    vertex mask is then OR-reconciled, because a wall vertex can be shared
    with a rank that owns no wall face touching it.
    """
    vS, _vE = dm.getDepthStratum(0)
    pStart, pEnd = dm.getChart()
    shared = _shared_point_flags(dm)
    mark = np.zeros(pEnd - pStart, dtype=np.int32)
    for f in range(*dm.getHeightStratum(1)):
        if len(dm.getSupport(f)) == 1 and not shared[f - pStart]:
            for q in dm.getTransitiveClosure(f)[0]:
                if dm.getPointDepth(int(q)) == 0:
                    mark[int(q) - pStart] = 1
    mark = _propagate_vertex(dm, mark, MPI.MAX, np.maximum)
    return mark[vS - pStart: vS - pStart + n_vertices] == 1


def _gather_region(dm, vertex_mark_chart, verbose=False):
    """Redistribute so every marked vertex's cell star (+1 layer) is one rank's.

    A mask-driven port of the contact stream's ``_redistribute_fault_interior``
    (feature/fault-split-node c8693579 / 1d487319; the label-driven original
    is measured at np=2..8 with serial-identical topology). Only the marked
    star moves — everything else keeps its load-balanced home — via a shell
    partitioner. Returns ``(new_dm, moved)``; the input is untouched.
    """
    comm = dm.getComm().tompi4py()
    if comm.size == 1:
        return dm, False

    work = dm.clone()
    cS, cE = work.getHeightStratum(0)
    vS, vE = work.getDepthStratum(0)
    pStart, pEnd = work.getChart()

    mark = vertex_mark_chart.astype(np.int32).copy()
    mark = _propagate_vertex(work, mark, MPI.MAX, np.maximum)

    def star_of_marked(m):
        out = set()
        for v in range(vS, vE):
            if m[v - pStart]:
                for q in work.getTransitiveClosure(v, useCone=False)[0]:
                    if cS <= int(q) < cE:
                        out.add(int(q))
        return out

    star = star_of_marked(mark)
    # One growth layer: the surgery needs every point in the closure of a
    # region cell unshared, and a point is unshared exactly when all its
    # incident cells are co-resident.
    mark2 = np.zeros(pEnd - pStart, dtype=np.int32)
    for c in star:
        for q in work.getTransitiveClosure(c)[0]:
            if vS <= int(q) < vE:
                mark2[int(q) - pStart] = 1
    mark2 = _propagate_vertex(work, mark2, MPI.MAX, np.maximum)
    star |= star_of_marked(mark2)

    counts = np.asarray(comm.allgather(len(star)))
    if counts.sum() == 0:
        raise ValueError("place_sheet: the sheet meets no cell on any rank")
    target = int(np.argmax(counts))

    assign = np.full(cE - cS, comm.rank, dtype=np.int32)
    for c in star:
        assign[c - cS] = target
    order = np.argsort(assign, kind="stable").astype(np.int32)
    sizes = np.bincount(assign, minlength=comm.size).astype(np.int32)

    part = work.getPartitioner()
    part.setType(PETSc.Partitioner.Type.SHELL)
    part.setShellPartition(comm.size, sizes=sizes, points=order)
    work.distribute()
    if verbose:
        uw.pprint(f"[place_sheet] gathered a {int(counts.sum())}-cell region "
                  f"onto rank {target}")
    return work, True


def _carve_cavity_3d(dm, X, cells, sheet_pts, sheet_tris, clearance,
                     held_cells, h_vertex, on_wall, shared_chart):
    """Victims, dropped tets and the closed cavity shell around the sheet.

    The same two-part rule as 2-D — vertices within the clearance go, and any
    tet the sheet passes through goes (all four corners can sit outside the
    clearance while the sheet crosses the interior) — with the same guards:
    wall vertices are never victims, cells of an embedded surface are held,
    a victim's whole star must be dropped, and the shell must be a closed
    manifold that never touches the domain wall. Rank-local: the caller
    guarantees (and this function asserts) that the whole region is interior
    to this rank — the gather's contract.
    """
    from underworld3.utilities.edge_split import cell_diameters

    h_cell = cell_diameters(dm)
    d_sheet = _sheet_distance(X, sheet_pts, sheet_tris)

    held_vertex = np.zeros(len(X), dtype=bool)
    if held_cells:
        for c in held_cells:
            held_vertex[cells[c]] = True
    victim = (d_sheet < clearance * h_vertex) & ~on_wall & ~held_vertex

    drop = victim[cells].any(axis=1)
    cen = X[cells].mean(axis=1)
    for t in sheet_tris:
        A, B, C = sheet_pts[t[0]], sheet_pts[t[1]], sheet_pts[t[2]]
        n = np.cross(B - A, C - A)
        n = n / np.linalg.norm(n)
        diam = max(np.linalg.norm(B - A), np.linalg.norm(C - A),
                   np.linalg.norm(C - B))
        near = np.linalg.norm(cen - (A + B + C) / 3.0, axis=1) < diam + h_cell
        if not near.any():
            continue
        s = (X[cells[near]] - A) @ n
        straddle = (s.max(axis=1) > 1e-12) & (s.min(axis=1) < -1e-12)
        sub = np.flatnonzero(near)[straddle]
        if not len(sub):
            continue
        rel = cen[sub] - A
        d00 = float((B - A) @ (B - A)); d01 = float((B - A) @ (C - A))
        d11 = float((C - A) @ (C - A))
        d20, d21 = rel @ (B - A), rel @ (C - A)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        inside = (v > -0.2) & (w > -0.2) & (v + w < 1.2)   # conservative
        drop[sub[inside]] = True
    if held_cells:
        drop[list(held_cells)] = False

    for c in np.flatnonzero(~drop):
        if victim[cells[c]].any():
            drop[c] = True
    if held_cells and drop[list(held_cells)].any():
        raise RuntimeError(
            "the sheet's cavity needs a cell that belongs to a surface "
            "already embedded. Surfaces must be separated by at least a "
            "cell; place close pairs as one thin volume instead.")

    drop_ids = np.flatnonzero(drop)
    if not len(drop_ids):
        raise ValueError("the sheet meets no cell of this mesh")

    cS, _cE = dm.getHeightStratum(0)
    vS, _vE = dm.getDepthStratum(0)
    fS, fE = dm.getHeightStratum(1)
    pStart, _pEnd = dm.getChart()
    dropped = set(int(c) + cS for c in drop_ids)
    shell = []
    for f in range(fS, fE):
        support = [int(c) for c in dm.getSupport(f)]
        n_in = sum(1 for c in support if c in dropped)
        if n_in == 0:
            continue
        if len(support) == 1:
            if shared_chart[f - pStart]:
                raise RuntimeError(
                    "the sheet's cavity touches a partition seam after the "
                    "gather; the region marking under-reached. Raise "
                    "`clearance` margin in the gather mask — this is a "
                    "defect, not a configuration error.")
            raise RuntimeError(
                "the sheet's cavity reached the domain wall; the sheet must "
                "be interior, with clearance to spare")
        if n_in == 1:
            shell.append((f, [int(p) - vS
                              for p in dm.getTransitiveClosure(f)[0]
                              if vS <= int(p) < vS + len(X)]))
    shell_verts = sorted({v for _f, verts in shell for v in verts})
    if victim[shell_verts].any():
        raise RuntimeError("a deleted vertex is on the cavity shell")

    from collections import Counter
    edge_count = Counter()
    for _f, verts in shell:
        a, b, c = sorted(verts)
        for e in ((a, b), (a, c), (b, c)):
            edge_count[e] += 1
    if any(k != 2 for k in edge_count.values()):
        raise RuntimeError(
            "the cavity shell is not a closed manifold; raise `clearance` so "
            "the cavity is wider than the shapes pinching it")

    return np.flatnonzero(victim), drop_ids, shell


def _gmsh_fill_3d(shell_xyz, shell_tris, sheet_pts, sheet_tris, h):
    """Tetrahedralise inside the shell with the sheet embedded, via gmsh.

    Gated, not trusted: the caller checks that no constrained node moved and
    that every sheet triangle survives; here the fill only has to exist.
    Runs as a serial library call on the rank that owns the gathered region.
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("uw_sheet_cavity")
        shell_tag = gmsh.model.addDiscreteEntity(2)
        n_shell = len(shell_xyz)
        gmsh.model.mesh.addNodes(2, shell_tag, list(range(1, n_shell + 1)),
                                 shell_xyz.reshape(-1).tolist())
        gmsh.model.mesh.addElementsByType(
            shell_tag, 2, [], (shell_tris + 1).reshape(-1).tolist())

        sheet_tag = gmsh.model.addDiscreteEntity(2)
        n_sheet = len(sheet_pts)
        gmsh.model.mesh.addNodes(2, sheet_tag,
                                 list(range(n_shell + 1,
                                            n_shell + n_sheet + 1)),
                                 sheet_pts.reshape(-1).tolist())
        gmsh.model.mesh.addElementsByType(
            sheet_tag, 2, [], (sheet_tris + n_shell + 1).reshape(-1).tolist())

        loop = gmsh.model.geo.addSurfaceLoop([shell_tag])
        vol = gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(2, [sheet_tag], 3, vol)

        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.3 * h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1.2 * h)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.model.mesh.generate(3)

        tags, xyz, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(xyz).reshape(-1, 3)
        row_of = {int(t): i for i, t in enumerate(np.asarray(tags))}
        ordered = sorted(row_of)
        points = xyz[[row_of[t] for t in ordered]]
        renum = {t: i for i, t in enumerate(ordered)}

        etypes, _eids, enodes = gmsh.model.mesh.getElements(3, vol)
        tets = None
        for et, nodes in zip(etypes, enodes):
            if et == 4:
                tets = np.array([renum[int(t)] for t in nodes],
                                dtype=np.int64).reshape(-1, 4)
        if tets is None:
            raise RuntimeError("gmsh produced no tetrahedra for the cavity")

        moved = sum(1 for t in range(1, n_shell + 1)
                    if not np.array_equal(points[renum[t]], shell_xyz[t - 1]))
        moved += sum(1 for t in range(n_shell + 1, n_shell + n_sheet + 1)
                     if not np.array_equal(points[renum[t]],
                                           sheet_pts[t - n_shell - 1]))

        setypes, _sids, senodes = gmsh.model.mesh.getElements(2, sheet_tag)
        sheet_out = None
        for et, nodes in zip(setypes, senodes):
            if et == 2:
                sheet_out = np.array([renum[int(t)] for t in nodes],
                                     dtype=np.int64).reshape(-1, 3)
        return points, tets, sheet_out, moved, n_shell
    finally:
        gmsh.finalize()


def _rebuild_sewn_3d(dm, drop_cell_ids, victim_ids, made_cells, placed):
    """Rebuild the local chart with cells replaced; every rank, collectively.

    The uninterpolated-cells + ``DMPlexInterpolate`` pattern (taken from
    fault_split.split_along_label_3d on feature/fault-split-node, recorded in
    the ledger): only cell-to-vertex cones are wired by hand — trivially
    orientation-free — and PETSc derives the faces, edges and every cone
    orientation. A rank with no surgery rebuilds its chart unchanged, because
    ``interpolate`` is collective and the star-forest of the interpolated
    mesh must be built by every rank together.

    ``made_cells`` entries are mixed: a non-negative value is an OLD vertex
    point; ``-(k+1)`` is row ``k`` of ``placed``. Returns the interpolated
    mesh, the chart point map (old -> new, -1 for deleted), and the new
    vertex ids of the placed rows.
    """
    pStart, pEnd = dm.getChart()
    cS, cE = dm.getHeightStratum(0)
    vS, vE = dm.getDepthStratum(0)
    fS, fE = dm.getHeightStratum(1)
    eS, eE = dm.getDepthStratum(1)
    nv_old = vE - vS

    keep_cell = np.ones(cE - cS, dtype=bool)
    keep_cell[np.asarray(drop_cell_ids, dtype=np.int64)] = False
    keep_vertex = np.ones(nv_old, dtype=bool)
    keep_vertex[np.asarray(victim_ids, dtype=np.int64)] = False
    v_old_to_compact = -np.ones(nv_old, dtype=np.int64)
    v_old_to_compact[keep_vertex] = np.arange(int(keep_vertex.sum()))
    n_surv = int(keep_vertex.sum())
    placed = np.asarray(placed, dtype=float).reshape(-1, 3)

    cell_verts = _tet_vertices(dm)
    kept = cell_verts[keep_cell]
    nc_new = int(keep_cell.sum()) + len(made_cells)
    nv_new = n_surv + len(placed)

    def v_uninterp(x):
        if x < 0:
            return nc_new + n_surv + (-int(x) - 1)
        return nc_new + int(v_old_to_compact[int(x)])

    new = PETSc.DMPlex().create(comm=dm.comm)
    new.setDimension(3)
    new.setChart(0, nc_new + nv_new)
    for i in range(nc_new):
        new.setConeSize(i, 4)
    new.setUp()
    for i, tet in enumerate(kept):
        new.setCone(i, [nc_new + int(v_old_to_compact[v]) for v in tet])
    for j, tet in enumerate(made_cells):
        new.setCone(int(keep_cell.sum()) + j,
                    [v_uninterp(v) for v in tet])
    new.symmetrize()
    new.stratify()
    new.interpolate()

    vS2, vE2 = new.getDepthStratum(0)
    if (new.getHeightStratum(0) != (0, nc_new)
            or (vS2, vE2) != (nc_new, nc_new + nv_new)):
        raise RuntimeError(
            "place_sheet internal: DMPlexInterpolate moved the cell or "
            "vertex numbering the point-map arithmetic relies on.")

    X = _coords(dm)[:nv_old]
    coords_new = np.vstack([X[keep_vertex], placed]) if len(placed) \
        else X[keep_vertex]
    reconnect._write_coordinates(new, dm.getCoordinateDim(), (vS2, vE2),
                                 coords_new)

    point_map = np.full(pEnd - pStart, -1, dtype=np.int64)
    surv_cells = np.flatnonzero(keep_cell)
    point_map[surv_cells + cS - pStart] = np.arange(len(surv_cells))
    surv_verts = np.flatnonzero(keep_vertex)
    point_map[surv_verts + vS - pStart] = vS2 + np.arange(n_surv)
    placed_new = vS2 + n_surv + np.arange(len(placed))

    # Old faces and edges are recovered by JOINING their surviving vertex
    # tuples in the new chart (the contact stream's recovery move). One that
    # does not join back was interior to the cavity and is legitimately gone;
    # the breach detector downstream is what confirms nothing LABELLED went
    # with it.
    for lo, hi in ((fS, fE), (eS, eE)):
        for q in range(lo, hi):
            verts = [int(x) - vS for x in dm.getTransitiveClosure(q)[0]
                     if dm.getPointDepth(int(x)) == 0]
            ms = [v_old_to_compact[v] for v in verts]
            if any(m < 0 for m in ms):
                continue
            joined = new.getFullJoin([int(vS2 + m) for m in ms])
            if len(joined) == 1:
                point_map[q - pStart] = int(joined[0])

    reconnect._copy_labels(new, dm, point_map)
    if uw.mpi.size > 1:
        reconnect._rebuild_point_sf(new, dm, point_map,
                                    int(new.getChart()[1]))
    return new, point_map, placed_new


def _owned_stratum_counts(dm):
    """Owned (root) point counts per stratum: (vertices, edges, faces, cells).

    A shared point is counted by its owner alone, so the sums allreduce to
    the global stratum sizes and the global Euler number is computable.
    """
    pStart, _pEnd = dm.getChart()
    shared_leaf = np.zeros(dm.getChart()[1] - pStart, dtype=bool)
    if uw.mpi.size > 1:
        try:
            _n, ilocal, _ir = dm.getPointSF().getGraph()
            if ilocal is not None and len(ilocal):
                shared_leaf[np.asarray(ilocal, dtype=np.int64) - pStart] = True
        except (ValueError, TypeError):
            # An unpopulated star-forest reports a root count petsc4py cannot
            # shape an array from (the same sanctioned mode reconnect's
            # _shared_points documents). Nothing is shared, so every local
            # point is owned and the zero mask is already right.
            pass
    out = []
    for lo, hi in (dm.getDepthStratum(0), dm.getDepthStratum(1),
                   dm.getHeightStratum(1), dm.getHeightStratum(0)):
        out.append(int(hi - lo) - int(shared_leaf[lo - pStart:
                                                  hi - pStart].sum()))
    return out


def place_sheet(dm, points, triangles, label=CUT_LABEL, label_value=1,
                clearance=0.6, verbose=False):
    """Embed a triangulated sheet in a 3-D mesh by placing its points.

    The 3-D form of :func:`place_along_lines`: the sheet's points become mesh
    vertices and every sheet triangle a labelled interior face, with the RIM
    free inside the mesh, on a mesh that already exists — so the fault's
    position is a design variable, not a property of mesh generation.

    Works in serial and in parallel, through ONE mechanism. In parallel the
    sheet's region is gathered onto a single rank first (the contact stream's
    measured policy — the star is thin, so the imbalance is bounded by the
    fault region, not the refined band), the serial carve-and-fill runs there
    as the rank-local step, and every rank rebuilds its chart collectively
    through the uninterpolate-then-``DMPlexInterpolate`` pattern. Because the
    gathered region is rank-interior, every point the surgery deletes or adds
    is unshared: the star-forest's leaf set is provably unchanged and only
    renumbers. The result is partition-independent by construction — the fill
    sees the identical cavity whatever the incoming partition was.

    The cavity fill is delegated to gmsh and GATED per call: both constraint
    surfaces bit-identical, every sheet triangle an interior face, conformity,
    global Euler number, volume conservation, and every previously embedded
    surface's interior-face count re-read off the result.

    Parameters
    ----------
    dm : PETSc.DMPlex
        A 3-D simplex mesh, serial or distributed. **Not modified.**
    points, triangles : array_like
        The sheet: ``(N, 3)`` vertices and ``(M, 3)`` triangle indices — the
        form :class:`~underworld3.meshing.FaultSurface` carries. Interior to
        the domain, non-self-intersecting, at least a cell from any embedded
        surface.
    label, label_value : str, int
        Label put on the sheet's faces in the result.
    clearance : float
        Delete a mesh vertex within this multiple of its local ``h`` of the
        sheet.
    verbose : bool
        Report the counts.

    Returns
    -------
    placed : PETSc.DMPlex
        A new mesh (distributed as the input was, with the sheet's region
        resident on one rank) in which every sheet triangle is a face
        carrying ``label``.
    info : dict
        Global counts: ``n_placed``, ``n_on_surface`` (always 0 in 3-D),
        ``n_removed``, ``n_surface_facets``, ``min_volume``.

    Raises
    ------
    NotImplementedError
        In 2-D — use :func:`place_along_lines`.
    RuntimeError, ValueError
        The carve/fill refusals; ALWAYS raised collectively — every rank
        raises the same error, or none does (the parallel discipline the
        2-D cut established).
    """
    if dm.getDimension() != 3:
        raise NotImplementedError(
            f"place_sheet is 3-D; this mesh is {dm.getDimension()}-D. Use "
            "place_along_lines for a curve in 2-D.")

    comm = uw.mpi.comm
    sheet_pts = np.asarray(points, dtype=float).reshape(-1, 3)
    sheet_tris = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)

    # -------------------------------------------------- mark, then gather
    vS, vE = dm.getDepthStratum(0)
    pStart, pEnd = dm.getChart()
    X = _coords(dm)[: vE - vS]
    cells = _tet_vertices(dm)
    h_vertex, _h_cell = _vertex_h_3d(dm, cells, len(X))
    d_sheet = _sheet_distance(X, sheet_pts, sheet_tris)
    # The gather mask is a SUPERSET of everything the carve may touch: the
    # victims (clearance) plus the crossed cells' vertices, which sit within
    # a cell diameter of the sheet. The +2 margin covers grading between
    # neighbouring cells; the carve asserts nothing shared afterwards, so an
    # under-reach is loud, never silent.
    mark = np.zeros(pEnd - pStart, dtype=np.int32)
    mark[np.flatnonzero(d_sheet < (clearance + 2.0) * h_vertex)
         + vS - pStart] = 1

    volume_before = np.array(
        [_owned_cell_volume(dm)], dtype=float)
    comm.Allreduce(MPI.IN_PLACE, volume_before, op=MPI.SUM)

    dm_work, moved = _gather_region(dm, mark, verbose=verbose)
    if moved:
        vS, vE = dm_work.getDepthStratum(0)
        pStart, pEnd = dm_work.getChart()
        X = _coords(dm_work)[: vE - vS]
        cells = _tet_vertices(dm_work)
        h_vertex, _h_cell = _vertex_h_3d(dm_work, cells, len(X))
        d_sheet = _sheet_distance(X, sheet_pts, sheet_tris)

    on_wall = _true_wall_vertex_mask(dm_work, len(X))
    shared = _shared_point_flags(dm_work).astype(bool)

    cS, _cE = dm_work.getHeightStratum(0)
    interface = _interface_faces_3d(dm_work)
    held_cells = set()
    for f in interface:
        for c in dm_work.getSupport(f):
            held_cells.add(int(c) - cS)
    held_counts = _interior_face_counts_3d(dm_work)

    from underworld3.utilities.edge_split import cell_diameters
    h_mean = np.array([float(cell_diameters(dm_work).sum()) if len(cells)
                       else 0.0, float(len(cells))])
    comm.Allreduce(MPI.IN_PLACE, h_mean, op=MPI.SUM)
    h = float(h_mean[0] / h_mean[1])

    # The surgery rank: the one holding the region. Every carve refusal is
    # REDUCED before anyone raises — a rank-local raise in parallel is a hang.
    n_region = int((d_sheet < clearance * h_vertex).sum())
    owners = np.asarray(comm.allgather(n_region))
    if owners.sum() == 0:
        raise ValueError("the sheet meets no cell of this mesh")
    target = int(np.argmax(owners))

    failure = None
    victims = drop_ids = None
    fill = None
    if comm.rank == target:
        try:
            victims, drop_ids, shell = _carve_cavity_3d(
                dm_work, X, cells, sheet_pts, sheet_tris, clearance,
                held_cells, h_vertex, on_wall, shared)
            # The gather's contract, asserted: nothing the surgery touches is
            # shared. A violation is a marking defect and must be loud.
            touched = set()
            for c in drop_ids:
                for q in dm_work.getTransitiveClosure(int(c) + cS)[0]:
                    touched.add(int(q))
            if any(shared[q - pStart] for q in touched):
                raise RuntimeError(
                    "place_sheet internal: the gathered region touches a "
                    "shared point; the gather mask under-reached.")

            shell_vert_ids = sorted({v for _f, verts in shell
                                     for v in verts})
            local = {v: i for i, v in enumerate(shell_vert_ids)}
            shell_xyz = X[shell_vert_ids]
            shell_tris = np.array([[local[v] for v in verts]
                                   for _f, verts in shell], dtype=np.int64)
            fill = _gmsh_fill_3d(shell_xyz, shell_tris, sheet_pts,
                                 sheet_tris, h)
            fill_pts, fill_tets, sheet_out, moved_nodes, n_shell = fill
            if moved_nodes:
                raise RuntimeError(
                    f"the fill moved {moved_nodes} constrained node(s); the "
                    "cavity cannot be sewn back. A defect, not a tolerance.")
            if sheet_out is None or len(sheet_out) != len(sheet_tris):
                raise RuntimeError(
                    "the fill remeshed the sheet "
                    f"({0 if sheet_out is None else len(sheet_out)} "
                    f"triangles for {len(sheet_tris)} given).")
        except (RuntimeError, ValueError) as exc:
            failure = f"{type(exc).__name__}: {exc}"

    failures = comm.allgather(failure)
    real = [f for f in failures if f]
    if real:
        raise RuntimeError(f"place_sheet failed on the surgery rank: "
                           f"{real[0]}")

    # ------------------------------------------------ rebuild, every rank
    if comm.rank == target:
        fill_pts, fill_tets, sheet_out, _moved, n_shell = fill
        made = np.where(
            fill_tets < n_shell,
            np.asarray(shell_vert_ids, dtype=np.int64)[
                np.clip(fill_tets, 0, n_shell - 1)],
            -(fill_tets - n_shell) - 1)
        placed = fill_pts[n_shell:]
        victims_arr = np.asarray(victims, dtype=np.int64)
        drop_arr = np.asarray(drop_ids, dtype=np.int64)
    else:
        made = np.empty((0, 4), dtype=np.int64)
        placed = np.empty((0, 3), dtype=float)
        victims_arr = np.empty(0, dtype=np.int64)
        drop_arr = np.empty(0, dtype=np.int64)

    new, point_map, placed_new = _rebuild_sewn_3d(
        dm_work, drop_arr, victims_arr, made, placed)

    # The sheet's faces, labelled by joining the fill's vertex tuples. The
    # label object must exist on every rank even though only the surgery rank
    # holds faces to mark.
    if not new.hasLabel(label):
        new.createLabel(label)
    n_facets_local = 0
    if comm.rank == target:
        out_label = new.getLabel(label)
        n_shell_ids = np.asarray(shell_vert_ids, dtype=np.int64)
        for t in sheet_out:
            ids = []
            for v in t:
                if v < n_shell:
                    old_pt = int(n_shell_ids[v]) + dm_work.getDepthStratum(0)[0]
                    ids.append(int(point_map[old_pt - pStart]))
                else:
                    ids.append(int(placed_new[v - n_shell]))
            joined = new.getFullJoin(ids)
            if len(joined) != 1:
                failure = ("a sheet triangle is not a face of the sewn mesh; "
                           "the fill was not sewn where it was cut.")
                break
            out_label.setValue(int(joined[0]), int(label_value))
            n_facets_local += 1
    failures = comm.allgather(failure)
    real = [f for f in failures if f]
    if real:
        raise RuntimeError(real[0])

    # ------------------------------------------------------- global gates
    counts = np.array([n_facets_local, len(victims_arr),
                       len(placed)], dtype=np.int64)
    comm.Allreduce(MPI.IN_PLACE, counts, op=MPI.SUM)
    n_facets, n_removed, n_placed = (int(x) for x in counts)
    if n_facets != len(sheet_tris):
        raise RuntimeError(
            f"{n_facets} sheet faces labelled for {len(sheet_tris)} "
            "triangles given.")

    volume_after = np.array([_owned_cell_volume(new)], dtype=float)
    comm.Allreduce(MPI.IN_PLACE, volume_after, op=MPI.SUM)
    if abs(volume_after[0] - volume_before[0]) > 1e-9 * volume_before[0]:
        raise RuntimeError(
            f"the placement changed the domain volume: "
            f"{volume_before[0]:.12f} -> {volume_after[0]:.12f}")

    owned = np.asarray(_owned_stratum_counts(new), dtype=np.int64)
    comm.Allreduce(MPI.IN_PLACE, owned, op=MPI.SUM)
    nv_g, ne_g, nf_g, nc_g = (int(x) for x in owned)
    if nv_g - ne_g + nf_g - nc_g != 1:
        raise RuntimeError(
            f"the sewn mesh has global Euler number "
            f"{nv_g - ne_g + nf_g - nc_g}, not 1")

    after = _interior_face_counts_3d(new)
    for key, before in held_counts.items():
        now = after.get(key, 0)
        if now < before or (now != before
                            and key != (label, int(label_value))):
            raise RuntimeError(
                f"placing {label!r} would leave the surface {key[0]!r} with "
                f"{now} interior faces instead of {before}.")

    min_vol = np.array([_owned_min_cell_volume(new)], dtype=float)
    comm.Allreduce(MPI.IN_PLACE, min_vol, op=MPI.MIN)

    info = {"n_placed": n_placed, "n_on_surface": 0,
            "n_removed": n_removed, "n_surface_facets": n_facets,
            "min_volume": float(min_vol[0])}
    if verbose:
        uw.pprint(f"[place_sheet {label!r}] placed {info['n_placed']} "
                  f"vertices, removed {info['n_removed']}; "
                  f"{info['n_surface_facets']} sheet faces")
    return new, info


# ===========================================================================
# The embedded thin volume — mesh the whole assembly, then embed it
# ===========================================================================
#
# The finite-width fault representation: each fault surface is thickened by
# ±width/2 into a THIN VOLUME, the volumes of a network are resolved against
# one another in gmsh's OCC kernel (``fragment`` — the only junction
# resolver, and CAD-only), the assembly is meshed standalone at layer scale,
# and the meshed assembly is embedded into the existing mesh by carving a
# cavity and filling the ANNULAR GAP between the cavity shell and the
# assembly's boundary skin — the skin a HOLE in the fill volume, both
# constraint surfaces discrete and verbatim. Junctions need no geometric
# treatment: two volumes that meet become ordinary cells of the union, and
# the rheology decides what happens there. Measured basis:
# ``~/+Simulations/mesh_reconnection_study/thin_volume_spike.py`` — widths
# h, h/2, h/4 and junction angles down to 10 degrees, all gated.

def _patch_frame(patch):
    """Unit normal of a planar patch, with planarity asserted."""
    P = np.asarray(patch, dtype=float)
    n = np.cross(P[1] - P[0], P[2] - P[0])
    norm = float(np.linalg.norm(n))
    if norm == 0.0:
        raise ValueError("a patch's first three corners are collinear")
    n = n / norm
    off = (P - P[0]) @ n
    span = float(np.linalg.norm(P - P[0], axis=1).max())
    if np.abs(off).max() > 1e-9 * max(span, 1.0):
        raise ValueError("a thin-volume patch must be planar; corner "
                         f"off-plane by {np.abs(off).max():.2e}")
    return n


def _occ_assembly_3d(patches, width, size):
    """Thicken each planar patch by ±width/2, fragment together, mesh.

    Returns ``(points, tets, cad_volume)`` — the assembly mesh in its own
    numbering, and the CAD volume of the fragment pieces, against which the
    meshed volume is gated (planar-faced solids mesh to their exact volume).
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("uw_thin_volume_assembly")
        occ = gmsh.model.occ
        solids = []
        for patch in patches:
            P = np.asarray(patch, dtype=float)
            n = _patch_frame(P)
            base = P - 0.5 * width * n
            pts = [occ.addPoint(*q) for q in base]
            lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)])
                     for i in range(len(pts))]
            loop = occ.addCurveLoop(lines)
            surf = occ.addPlaneSurface([loop])
            out = occ.extrude([(2, surf)], *(width * n))
            solids += [t for d, t in out if d == 3]
        if len(solids) > 1:
            occ.fragment([(3, solids[0])], [(3, t) for t in solids[1:]])
        occ.synchronize()

        vols = gmsh.model.getEntities(3)
        cad_volume = sum(occ.getMass(3, t) for _d, t in vols)

        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.7 * size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1.3 * size)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.model.mesh.generate(3)

        tags, xyz, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(xyz).reshape(-1, 3)
        renum = {int(t): i for i, t in enumerate(tags)}
        tets = []
        for v in vols:
            et, _ei, en = gmsh.model.mesh.getElements(3, v[1])
            for t, nodes in zip(et, en):
                if t == 4:
                    tets.append(np.array([renum[int(x)] for x in nodes],
                                         dtype=np.int64).reshape(-1, 4))
        if not tets:
            raise RuntimeError("the assembly meshed to no tetrahedra")
        return xyz, np.vstack(tets), float(cad_volume)
    finally:
        gmsh.finalize()


def _assembly_skin(points, cells):
    """Boundary facets (support 1) of a standalone assembly mesh.

    Works for tets (skin = triangles) and triangles (skin = edges). Returns
    ``(skin_xyz, skin_local, node_ids)`` — the skin in its own compact
    numbering plus the assembly node each skin node is.
    """
    from collections import Counter
    from itertools import combinations

    nv = cells.shape[1]
    faces = Counter()
    for cell in cells:
        for tri in combinations(sorted(int(v) for v in cell), nv - 1):
            faces[tri] += 1
    skin = np.array([k for k, n in faces.items() if n == 1], dtype=np.int64)
    node_ids = np.unique(skin)
    local = {int(v): i for i, v in enumerate(node_ids)}
    skin_local = np.array([[local[int(v)] for v in f] for f in skin],
                          dtype=np.int64)
    return points[node_ids], skin_local, node_ids


def _carve_around_volume_3d(dm, X, cells, skin_pts, skin_tris, reach_vertex,
                            reach_cell, held_cells, on_wall, shared_chart):
    """Victims, dropped tets and the closed shell around a FAT object.

    Differs from the sheet's carve in two measured ways. The reach is a
    LENGTH per vertex/cell (``max(clearance*h, 0.6*width)``), not a bare
    multiple of h — it must cover the layer's own half-width however sub-h
    the layer is. And the union of victim stars around a volume can PINCH —
    a shell edge whose surrounding cells are part-dropped in two wedges — so
    the drop set is GROWN at every non-manifold shell edge until the shell
    closes; dropping more cells only enlarges the fill (thin_volume_spike:
    converges in a few rounds).
    """
    d_skin = _sheet_distance(X, skin_pts, skin_tris)

    held_vertex = np.zeros(len(X), dtype=bool)
    if held_cells:
        for c in held_cells:
            held_vertex[cells[c]] = True
    victim = (d_skin < reach_vertex) & ~on_wall & ~held_vertex

    drop = victim[cells].any(axis=1)
    # A background cell can straddle the layer's rim with every corner
    # outside the reach; its centroid cannot be far from the skin.
    cen_d = _sheet_distance(X[cells].mean(axis=1), skin_pts, skin_tris)
    drop |= cen_d < reach_cell
    if held_cells:
        drop[list(held_cells)] = False
    for c in np.flatnonzero(~drop):
        if victim[cells[c]].any():
            drop[c] = True
    if held_cells and drop[list(held_cells)].any():
        raise RuntimeError(
            "the thin volume's cavity needs a cell that belongs to a surface "
            "already embedded. Zones and surfaces must be separated by at "
            "least a cell.")
    if not drop.any():
        raise ValueError("the thin volume meets no cell of this mesh")

    cS, _cE = dm.getHeightStratum(0)
    vS, _vE = dm.getDepthStratum(0)
    fS, fE = dm.getHeightStratum(1)
    pStart, _pEnd = dm.getChart()

    face_verts = {}
    face_support = {}
    for f in range(fS, fE):
        face_support[f] = [int(c) - cS for c in dm.getSupport(f)]
        face_verts[f] = [int(p) - vS for p in dm.getTransitiveClosure(f)[0]
                         if vS <= int(p) < vS + len(X)]

    from collections import Counter
    for _round in range(20):
        shell = []
        for f in range(fS, fE):
            support = face_support[f]
            n_in = sum(1 for c in support if drop[c])
            if n_in == 0:
                continue
            if len(support) == 1:
                if shared_chart[f - pStart]:
                    raise RuntimeError(
                        "the thin volume's cavity touches a partition seam "
                        "after the gather; the region marking under-reached. "
                        "A defect, not a configuration error.")
                raise RuntimeError(
                    "the thin volume's cavity reached the domain wall; the "
                    "volume must be interior, with clearance to spare")
            if n_in == 1:
                shell.append((f, face_verts[f]))
        edge_count = Counter()
        for _f, verts in shell:
            a, b, c = sorted(verts)
            for e in ((a, b), (a, c), (b, c)):
                edge_count[e] += 1
        bad = [e for e, k in edge_count.items() if k != 2]
        if not bad:
            break
        for a, b in bad:
            grow = (cells == a).any(axis=1) & (cells == b).any(axis=1)
            if held_cells:
                held = np.zeros(len(cells), dtype=bool)
                held[list(held_cells)] = True
                if (grow & held).any():
                    raise RuntimeError(
                        "closing the cavity shell needs a cell held for a "
                        "surface already embedded; move the zone away or "
                        "raise `clearance`.")
            drop |= grow
    else:
        raise RuntimeError(
            "the cavity shell did not close in 20 growth rounds; raise "
            "`clearance`.")

    shell_verts = sorted({v for _f, verts in shell for v in verts})
    if victim[shell_verts].any():
        raise RuntimeError("a deleted vertex is on the cavity shell")
    return np.flatnonzero(victim), np.flatnonzero(drop), shell


def _gmsh_fill_annulus_3d(shell_xyz, shell_tris, skin_xyz, skin_tris,
                          size_out, size_in):
    """Tetrahedralise BETWEEN the cavity shell and the assembly skin.

    The skin is a HOLE in the fill volume: outer surface loop the shell,
    inner surface loop the skin, both discrete entities carrying their
    triangulations verbatim (the mechanism thin_volume_spike measured; the
    embedded-sheet fill cannot express an interior boundary).

    Returns ``(points, tets, moved, skin_out, n_shell)`` with the fill's
    nodes ordered shell first, skin second, new points after.
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("uw_thin_volume_gap")
        n_shell, n_skin = len(shell_xyz), len(skin_xyz)

        shell_tag = gmsh.model.addDiscreteEntity(2)
        gmsh.model.mesh.addNodes(2, shell_tag, list(range(1, n_shell + 1)),
                                 shell_xyz.reshape(-1).tolist())
        gmsh.model.mesh.addElementsByType(
            shell_tag, 2, [], (shell_tris + 1).reshape(-1).tolist())

        skin_tag = gmsh.model.addDiscreteEntity(2)
        gmsh.model.mesh.addNodes(2, skin_tag,
                                 list(range(n_shell + 1,
                                            n_shell + n_skin + 1)),
                                 skin_xyz.reshape(-1).tolist())
        gmsh.model.mesh.addElementsByType(
            skin_tag, 2, [], (skin_tris + n_shell + 1).reshape(-1).tolist())

        outer = gmsh.model.geo.addSurfaceLoop([shell_tag])
        inner = gmsh.model.geo.addSurfaceLoop([skin_tag])
        vol = gmsh.model.geo.addVolume([outer, inner])
        gmsh.model.geo.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.7 * size_in)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1.3 * size_out)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.model.mesh.generate(3)

        tags, xyz, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(xyz).reshape(-1, 3)
        row_of = {int(t): i for i, t in enumerate(np.asarray(tags))}
        ordered = sorted(row_of)
        points = xyz[[row_of[t] for t in ordered]]
        renum = {t: i for i, t in enumerate(ordered)}

        et, _ei, en = gmsh.model.mesh.getElements(3, vol)
        tets = None
        for t, nodes in zip(et, en):
            if t == 4:
                tets = np.array([renum[int(x)] for x in nodes],
                                dtype=np.int64).reshape(-1, 4)
        if tets is None:
            raise RuntimeError("gmsh produced no tetrahedra in the gap")

        moved = sum(1 for t in range(1, n_shell + 1)
                    if not np.array_equal(points[renum[t]],
                                          shell_xyz[t - 1]))
        moved += sum(1 for t in range(n_shell + 1, n_shell + n_skin + 1)
                     if not np.array_equal(points[renum[t]],
                                           skin_xyz[t - n_shell - 1]))

        set_, _sids, sen = gmsh.model.mesh.getElements(2, skin_tag)
        skin_out = 0
        for t, nodes in zip(set_, sen):
            if t == 2:
                skin_out = len(nodes) // 3
        return points, tets, moved, skin_out, n_shell
    finally:
        gmsh.finalize()


def _occ_assembly_2d(polylines, width, size):
    """Thicken each polyline segment into a quad, fragment together, mesh.

    The 2-D thin volume: a ribbon is the union of one quad per polyline
    segment, kinks and crossings resolved by ``fragment`` exactly as the 3-D
    junctions are. Returns ``(points, triangles, cad_area)``.
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("uw_ribbon_assembly")
        occ = gmsh.model.occ

        # ONE mitre-joined outline polygon per polyline. Per-segment quads
        # fragmented together were measured to sliver at every kink — the
        # overlap of two quads on the kink's inner side becomes a separate
        # thin lens face whose tip meshes at ~2 degrees — while a single
        # outline has no internal seam at all. Fragment then resolves only
        # the junctions BETWEEN polylines, which is its job.
        def outline(P):
            P = np.asarray(P, dtype=float)[:, :2]
            keep = np.concatenate(
                [[True], np.linalg.norm(np.diff(P, axis=0), axis=1) > 0.0])
            P = P[keep]
            if len(P) < 2:
                raise ValueError("a polyline needs two distinct points")
            t = np.diff(P, axis=0)
            t = t / np.linalg.norm(t, axis=1)[:, None]
            n = np.column_stack([-t[:, 1], t[:, 0]])
            left = [P[0] + 0.5 * width * n[0]]
            right = [P[0] - 0.5 * width * n[0]]
            for k in range(1, len(P) - 1):
                m = n[k - 1] + n[k]
                nm = float(np.linalg.norm(m))
                half_cos = 0.5 * nm         # cos(theta/2) of the turn
                if half_cos < 0.25:
                    raise ValueError(
                        "a polyline turns too sharply to buffer with a "
                        "mitre join (interior angle under ~30 degrees); "
                        "smooth the trace or split it into two polylines.")
                m = m / nm
                reach = 0.5 * width / half_cos
                left.append(P[k] + reach * m)
                right.append(P[k] - reach * m)
            left.append(P[-1] + 0.5 * width * n[-1])
            right.append(P[-1] - 0.5 * width * n[-1])
            return np.array(left + right[::-1])

        surfs = []
        for line in polylines:
            ring = outline(line)
            pts = [occ.addPoint(q[0], q[1], 0.0) for q in ring]
            lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)])
                     for i in range(len(pts))]
            surfs.append(occ.addPlaneSurface([occ.addCurveLoop(lines)]))
        if not surfs:
            raise ValueError("the polylines contain no segment to thicken")
        if len(surfs) > 1:
            occ.fragment([(2, surfs[0])], [(2, t) for t in surfs[1:]])
        occ.synchronize()

        faces = gmsh.model.getEntities(2)
        cad_area = sum(occ.getMass(2, t) for _d, t in faces)

        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.7 * size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1.3 * size)
        gmsh.model.mesh.generate(2)

        tags, xyz, _ = gmsh.model.mesh.getNodes()
        xy = np.asarray(xyz).reshape(-1, 3)[:, :2]
        renum = {int(t): i for i, t in enumerate(tags)}
        tris = []
        for f in faces:
            et, _ei, en = gmsh.model.mesh.getElements(2, f[1])
            for t, nodes in zip(et, en):
                if t == 2:
                    tris.append(np.array([renum[int(x)] for x in nodes],
                                         dtype=np.int64).reshape(-1, 3))
        if not tris:
            raise RuntimeError("the ribbon assembly meshed to no triangles")
        return xy, np.vstack(tris), float(cad_area)
    finally:
        gmsh.finalize()


def _segments_distance(X, pts, edges):
    """Distance from each point of ``X`` to a set of segments (2-D skin)."""
    best = np.full(len(X), np.inf)
    for a, b in edges:
        A, B = pts[int(a)], pts[int(b)]
        e = B - A
        u = np.clip(((X - A) @ e) / float(e @ e), 0.0, 1.0)
        best = np.minimum(
            best, np.linalg.norm(X - (A + u[:, None] * e), axis=1))
    return best


def _skin_loops(skin_edges):
    """Order a 2-D skin's edges into closed loops of vertex ids.

    A manifold skin gives every vertex exactly two incident edges; anything
    else is a defect of the assembly mesh and is refused.
    """
    adj = {}
    for a, b in skin_edges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    if any(len(v) != 2 for v in adj.values()):
        raise RuntimeError("the assembly's skin is not a set of closed "
                           "loops; the layer mesh is defective")
    loops, seen = [], set()
    for start in sorted(adj):
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        while True:
            seen.add(cur)
            a, b = adj[cur]
            nxt = b if a == prev else a
            if nxt == start:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        loops.append(loop)
    return loops


def _ring_growing(cells, drop, held_mask):
    """The cavity ring, growing the drop set at pinch vertices until simple.

    The 2-D form of the 3-D shell growth: a fat object's victim-star union
    can leave a vertex where the cavity boundary touches itself. Growing the
    drop at that vertex merges the wedges; dropping more only enlarges the
    fill.
    """
    for _round in range(20):
        directed = {}
        for ci in np.flatnonzero(drop):
            v0, v1, v2 = cells[ci]
            for a, b in ((v0, v1), (v1, v2), (v2, v0)):
                directed[(int(a), int(b))] = int(ci)
        ring_edges = [(a, b) for (a, b) in directed
                      if (b, a) not in directed]
        starts = {}
        pinch = set()
        for a, b in ring_edges:
            if a in starts:
                pinch.add(a)
            starts[a] = b
        if not pinch:
            ring = _cavity_ring(cells, np.flatnonzero(drop))
            if ring is None:
                raise RuntimeError(
                    "the cells cleared for the thin volume do not leave one "
                    "simple hole. Raise `clearance`.")
            return ring, drop
        for v in pinch:
            grow = (cells == v).any(axis=1)
            if (grow & held_mask).any():
                raise RuntimeError(
                    "closing the cavity needs a cell held for a surface "
                    "already embedded; move the zone away or raise "
                    "`clearance`.")
            drop |= grow
    raise RuntimeError(
        "the cavity did not become simple in 20 growth rounds; raise "
        "`clearance`.")


ZONE_LABEL = "uw_zone"


def _place_thin_volume_2d(dm, polylines, width, label, label_value,
                          clearance, size, verbose):
    """The ribbon: the identical construction one dimension down.

    Serial, sharing :func:`place_along_lines`' refusal — the 2-D placement
    family adds points without the chart-expansion rebuild, and the 3-D form
    is the parallel one.
    """
    if uw.mpi.size > 1:
        raise NotImplementedError(
            "place_thin_volume in 2-D is serial, like place_along_lines. "
            "The 3-D form is parallel (gather-first).")

    asm_pts, asm_tris, cad_area = _occ_assembly_2d(polylines, width, size)
    P = asm_pts[asm_tris]
    twice = ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
             - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
    mesh_area = float(np.abs(twice).sum() / 2.0)
    if abs(mesh_area - cad_area) > 1e-9 * cad_area:
        raise RuntimeError(
            f"the ribbon assembly meshed to area {mesh_area:.12e} against "
            f"CAD {cad_area:.12e}; the layer mesh does not fill its own "
            "quads.")

    _skin_xyz, skin_local, skin_node_ids = _assembly_skin(asm_pts, asm_tris)
    skin_edges = [(int(skin_node_ids[a]), int(skin_node_ids[b]))
                  for a, b in skin_local]
    loops_asm = _skin_loops(skin_edges)

    X = _coords(dm)
    vS, vE = dm.getDepthStratum(0)
    cS, _cE = dm.getHeightStratum(0)
    X = X[: vE - vS]
    cells = _cells_anticlockwise(dm, X)
    on_boundary = _boundary_vertices(dm, len(X))
    held_v, held_c = _interface_vertices_and_cells(dm, len(X), len(cells))
    held_counts = _interface_facet_counts(dm)
    beside_held = np.zeros(len(X), dtype=bool)
    beside_held[cells[held_c].ravel()] = True
    protected = on_boundary | held_v | beside_held

    area_before = float(cell_areas(dm).sum())

    h_v = _vertex_h(X, _edge_vertices(dm))
    d = _segments_distance(X, asm_pts, skin_edges)
    reach_v = np.maximum(clearance * h_v, 0.6 * width)
    victim = (d < reach_v) & ~protected

    drop = victim[cells].any(axis=1)
    cen = X[cells].mean(axis=1)
    reach_c = np.maximum(clearance * h_v[cells].min(axis=1), 0.6 * width)
    drop |= _segments_distance(cen, asm_pts, skin_edges) < reach_c
    drop &= ~held_c
    need = victim[cells].any(axis=1)
    if (need & held_c).any():
        raise RuntimeError(
            "the ribbon's cavity needs a cell that belongs to a surface "
            "already embedded. Zones and surfaces must be separated by at "
            "least a cell.")
    drop |= need
    if not drop.any():
        raise ValueError("the thin volume meets no cell of this mesh")

    ring, drop = _ring_growing(cells, drop, held_c)
    if on_boundary[np.asarray(ring)].any():
        raise RuntimeError(
            "the ribbon's cavity reached the domain wall; the volume must "
            "be interior, with clearance to spare")
    if victim[np.asarray(ring)].any():
        raise RuntimeError("a deleted vertex is on the cavity boundary")

    Xall = np.vstack([X, asm_pts])
    holes = [[len(X) + int(v) for v in loop] for loop in loops_asm]
    gap_tris, extra = _gmsh_fill_2d(Xall, ring, None, holes=holes)
    placed = np.vstack([asm_pts, extra]) if len(extra) else asm_pts

    def mixed(v):
        return int(v) + vS if v < len(X) else -(int(v) - len(X) + 1)

    made = [tuple(mixed(v) for v in t) for t in gap_tris]
    made += [tuple(-(int(v) + 1) for v in t) for t in asm_tris]

    new_dm, _point_map, placed_points = reconnect.rebuild_cavities(
        dm, np.flatnonzero(victim) + vS, np.flatnonzero(drop) + cS,
        made, placed)

    skin_label = label + "_skin"
    for name in (label, skin_label):
        if not new_dm.hasLabel(name):
            new_dm.createLabel(name)
    out_label = new_dm.getLabel(label)
    out_skin = new_dm.getLabel(skin_label)
    n_zone = 0
    for t in asm_tris:
        joined = new_dm.getFullJoin([int(placed_points[int(v)]) for v in t])
        if len(joined) != 1:
            raise RuntimeError(
                "an assembly cell is not a cell of the sewn mesh; the embed "
                "lost the layer.")
        out_label.setValue(int(joined[0]), int(label_value))
        n_zone += 1
    n_skin = 0
    for a, b in skin_edges:
        joined = new_dm.getFullJoin([int(placed_points[a]),
                                     int(placed_points[b])])
        if len(joined) != 1:
            raise RuntimeError(
                "a skin edge is not an edge of the sewn mesh; the gap was "
                "not sewn onto the layer.")
        out_skin.setValue(int(joined[0]), int(label_value))
        n_skin += 1

    areas = cell_areas(new_dm)
    over = sum(1 for f in range(*new_dm.getHeightStratum(1))
               if len(new_dm.getSupport(f)) > 2)
    if over:
        raise RuntimeError(
            f"{over} facet(s) of the result have more than two cells")
    if (areas <= 0.0).any():
        raise RuntimeError(
            f"{int((areas <= 0.0).sum())} cell(s) of the result are inverted")
    if abs(float(areas.sum()) - area_before) > 1e-9 * area_before:
        raise RuntimeError(
            f"the placement changed the domain area: {area_before:.12f} -> "
            f"{float(areas.sum()):.12f}")

    after = _interface_facet_counts(new_dm)
    for key, before in held_counts.items():
        now = after.get(key, 0)
        if now < before or (now != before
                            and key != (skin_label, int(label_value))):
            raise RuntimeError(
                f"placing {label!r} would leave the surface {key[0]!r} with "
                f"{now} facets instead of {before}.")

    info = {"n_zone_cells": n_zone, "n_skin_faces": n_skin,
            "n_placed": len(placed), "n_removed": int(victim.sum()),
            "min_area": float(areas.min()),
            "min_angle": float(min_angles(new_dm).min())}
    if verbose:
        uw.pprint(f"[place_thin_volume {label!r}] {n_zone} zone cells, "
                  f"{n_skin} skin edges; placed {info['n_placed']} vertices, "
                  f"removed {info['n_removed']}; min angle "
                  f"{info['min_angle']:.2f} deg")
    return new_dm, info


def place_thin_volume(dm, patches, width, label=ZONE_LABEL, label_value=1,
                      clearance=0.7, size=None, verbose=False):
    """Embed a THIN VOLUME of the given width around each patch, junctions free.

    The finite-width fault representation: each planar patch is thickened by
    ``±width/2``, the thickened volumes of the whole network are resolved
    against one another with OCC ``fragment`` — a junction becomes ordinary
    cells of the union, no geometric treatment, the rheology decides — the
    assembly is meshed standalone at layer scale (sub-``h`` widths are the
    point: ``V = 2 ε̇ w`` makes the width constitutive), and the meshed
    assembly is embedded whole into the existing mesh: a cavity is carved
    around it and gmsh fills the annular gap with the assembly's boundary
    skin as an interior HOLE, both constraint surfaces verbatim.

    In the result the layer's CELLS carry ``(label, label_value)`` — the
    volume representation exists to give the zone cells to the rheology —
    and the skin's faces carry ``(label + "_skin", label_value)``. The two
    must be separate labels: an interface is identified by a label stratum
    holding only faces (:func:`_interface_faces_3d` skips any stratum that
    contains cells), and it is the skin label that makes a later placement
    hold its cavity clear of this zone.

    Serial and parallel through the same gather-first mechanism as
    :func:`place_sheet`; the assembly is meshed once (rank 0) and
    broadcast, so every rank marks against the identical skin.

    Parameters
    ----------
    dm : PETSc.DMPlex
        A 3-D simplex mesh, serial or distributed. **Not modified.**
    patches : sequence of array_like
        In 3-D: one or more PLANAR polygons, ``(N, 3)`` corners each. In
        2-D: one or more polylines, ``(N, 2)`` points each, thickened into
        ribbons. Interior to the domain with clearance to spare; patches may
        cross — that is the point — but must not touch a surface already
        embedded.
    width : float
        The layer thickness, a real mesh parameter; ``width < h`` is
        supported and measured.
    label, label_value : str, int
        Label carried by the layer's cells AND the skin's faces.
    clearance : float
        Delete a mesh vertex within ``max(clearance*h, 0.6*width)`` of the
        skin.
    size : float or None
        The layer's own mesh size; ``None`` takes ``0.9 * width``.
    verbose : bool
        Report the counts.

    Returns
    -------
    placed : PETSc.DMPlex
        A new mesh with the assembly's cells embedded verbatim.
    info : dict
        Global counts: ``n_zone_cells``, ``n_skin_faces``, ``n_placed``
        (vertices added), ``n_removed`` (vertices deleted), ``min_volume``.

    Raises
    ------
    NotImplementedError
        In 2-D in parallel (the ribbon shares :func:`place_along_lines`'
        serial scope; the 3-D form is the parallel one).
    RuntimeError, ValueError
        Carve/fill refusals, always collective.
    """
    width = float(width)
    if width <= 0.0:
        raise ValueError("width must be positive")
    size = 0.9 * width if size is None else float(size)

    if dm.getDimension() == 2:
        return _place_thin_volume_2d(dm, patches, width, label, label_value,
                                     clearance, size, verbose)
    if dm.getDimension() != 3:
        raise NotImplementedError(
            f"place_thin_volume takes a 2-D or 3-D simplex mesh; this mesh "
            f"is {dm.getDimension()}-D.")

    comm = uw.mpi.comm

    # ------------------------------------------ the assembly, once, shared
    failure = None
    payload = None
    if comm.rank == 0:
        try:
            asm_pts, asm_tets, cad_vol = _occ_assembly_3d(
                patches, width, size)
            v6 = np.einsum(
                "ij,ij->i",
                np.cross(asm_pts[asm_tets][:, 1] - asm_pts[asm_tets][:, 0],
                         asm_pts[asm_tets][:, 2] - asm_pts[asm_tets][:, 0]),
                asm_pts[asm_tets][:, 3] - asm_pts[asm_tets][:, 0])
            mesh_vol = float(np.abs(v6).sum() / 6.0)
            if abs(mesh_vol - cad_vol) > 1e-9 * cad_vol:
                raise RuntimeError(
                    f"the assembly meshed to volume {mesh_vol:.12e} against "
                    f"CAD {cad_vol:.12e}; the layer mesh does not fill its "
                    "own solids.")
            payload = (asm_pts, asm_tets)
        except (RuntimeError, ValueError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
    failures = comm.allgather(failure)
    real = [f for f in failures if f]
    if real:
        raise RuntimeError(f"place_thin_volume assembly failed: {real[0]}")
    asm_pts, asm_tets = comm.bcast(payload, root=0)
    skin_xyz, skin_tris, skin_node_ids = _assembly_skin(asm_pts, asm_tets)

    # -------------------------------------------------- mark, then gather
    vS, vE = dm.getDepthStratum(0)
    pStart, pEnd = dm.getChart()
    X = _coords(dm)[: vE - vS]
    cells = _tet_vertices(dm)
    h_vertex, _h_cell = _vertex_h_3d(dm, cells, len(X))
    d_skin = _sheet_distance(X, skin_xyz, skin_tris)
    reach_v = np.maximum(clearance * h_vertex, 0.6 * width)
    mark = np.zeros(pEnd - pStart, dtype=np.int32)
    mark[np.flatnonzero(d_skin < reach_v + 2.0 * h_vertex)
         + vS - pStart] = 1

    volume_before = np.array([_owned_cell_volume(dm)], dtype=float)
    comm.Allreduce(MPI.IN_PLACE, volume_before, op=MPI.SUM)

    dm_work, moved = _gather_region(dm, mark, verbose=verbose)
    if moved:
        vS, vE = dm_work.getDepthStratum(0)
        pStart, pEnd = dm_work.getChart()
        X = _coords(dm_work)[: vE - vS]
        cells = _tet_vertices(dm_work)
        h_vertex, _h_cell = _vertex_h_3d(dm_work, cells, len(X))
        d_skin = _sheet_distance(X, skin_xyz, skin_tris)
        reach_v = np.maximum(clearance * h_vertex, 0.6 * width)

    on_wall = _true_wall_vertex_mask(dm_work, len(X))
    shared = _shared_point_flags(dm_work).astype(bool)

    cS, _cE = dm_work.getHeightStratum(0)
    interface = _interface_faces_3d(dm_work)
    held_cells = set()
    for f in interface:
        for c in dm_work.getSupport(f):
            held_cells.add(int(c) - cS)
    held_counts = _interior_face_counts_3d(dm_work)

    from underworld3.utilities.edge_split import cell_diameters
    h_cell_local = cell_diameters(dm_work) if len(cells) else np.zeros(0)
    h_mean = np.array([float(h_cell_local.sum()), float(len(cells))])
    comm.Allreduce(MPI.IN_PLACE, h_mean, op=MPI.SUM)
    h = float(h_mean[0] / h_mean[1])
    reach_c = (np.maximum(clearance * h_cell_local, 0.6 * width)
               if len(cells) else np.zeros(0))

    n_region = int((d_skin < reach_v).sum())
    owners = np.asarray(comm.allgather(n_region))
    if owners.sum() == 0:
        raise ValueError("the thin volume meets no cell of this mesh")
    target = int(np.argmax(owners))

    failure = None
    victims = drop_ids = None
    fill = shell_vert_ids = None
    if comm.rank == target:
        try:
            victims, drop_ids, shell = _carve_around_volume_3d(
                dm_work, X, cells, skin_xyz, skin_tris, reach_v, reach_c,
                held_cells, on_wall, shared)
            touched = set()
            for c in drop_ids:
                for q in dm_work.getTransitiveClosure(int(c) + cS)[0]:
                    touched.add(int(q))
            if any(shared[q - pStart] for q in touched):
                raise RuntimeError(
                    "place_thin_volume internal: the gathered region touches "
                    "a shared point; the gather mask under-reached.")

            shell_vert_ids = sorted({v for _f, verts in shell
                                     for v in verts})
            local = {v: i for i, v in enumerate(shell_vert_ids)}
            shell_xyz = X[shell_vert_ids]
            shell_tris = np.array([[local[v] for v in verts]
                                   for _f, verts in shell], dtype=np.int64)
            fill = _gmsh_fill_annulus_3d(shell_xyz, shell_tris, skin_xyz,
                                         skin_tris, size_out=h, size_in=size)
            _pts, _tets, moved_nodes, skin_out, _n_shell = fill
            if moved_nodes:
                raise RuntimeError(
                    f"the gap fill moved {moved_nodes} constrained node(s); "
                    "the cavity cannot be sewn back.")
            if skin_out != len(skin_tris):
                raise RuntimeError(
                    f"the gap fill remeshed the skin ({skin_out} triangles "
                    f"for {len(skin_tris)} given).")
        except (RuntimeError, ValueError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
    failures = comm.allgather(failure)
    real = [f for f in failures if f]
    if real:
        raise RuntimeError(
            f"place_thin_volume failed on the surgery rank: {real[0]}")

    # ------------------------------------------------ rebuild, every rank
    # Placed rows: the assembly's nodes first, the gap fill's new points
    # after. A gap-fill node is a shell node (an OLD vertex), a skin node
    # (assembly row) or new; assembly tets reference assembly rows only.
    if comm.rank == target:
        fill_pts, fill_tets, _m, _s, n_shell = fill
        n_skin = len(skin_xyz)
        skin_row = np.asarray(skin_node_ids, dtype=np.int64)
        gap_new = fill_pts[n_shell + n_skin:]

        def gap_code(v):
            if v < n_shell:
                return int(shell_vert_ids[v])
            if v < n_shell + n_skin:
                return -(int(skin_row[v - n_shell]) + 1)
            return -(len(asm_pts) + (int(v) - n_shell - n_skin) + 1)

        made = np.array(
            [[gap_code(int(v)) for v in tet] for tet in fill_tets]
            + [[-(int(v) + 1) for v in tet] for tet in asm_tets],
            dtype=np.int64)
        placed = np.vstack([asm_pts, gap_new])
        victims_arr = np.asarray(victims, dtype=np.int64)
        drop_arr = np.asarray(drop_ids, dtype=np.int64)
    else:
        made = np.empty((0, 4), dtype=np.int64)
        placed = np.empty((0, 3), dtype=float)
        victims_arr = np.empty(0, dtype=np.int64)
        drop_arr = np.empty(0, dtype=np.int64)

    new, point_map, placed_new = _rebuild_sewn_3d(
        dm_work, drop_arr, victims_arr, made, placed)

    # Label the zone's cells and the skin's faces, by joining vertex tuples.
    # Two labels, deliberately: a stratum holding cells is a volume label and
    # is invisible to the interface machinery, so the skin faces — which a
    # later placement must hold clear of — go under their own name.
    skin_label = label + "_skin"
    for name in (label, skin_label):
        if not new.hasLabel(name):
            new.createLabel(name)
    n_cells_local = 0
    n_skin_local = 0
    if comm.rank == target:
        out_label = new.getLabel(label)
        out_skin = new.getLabel(skin_label)
        for tet in asm_tets:
            joined = new.getFullJoin([int(placed_new[int(v)]) for v in tet])
            if len(joined) != 1:
                failure = ("an assembly cell is not a cell of the sewn mesh; "
                           "the embed lost the layer.")
                break
            out_label.setValue(int(joined[0]), int(label_value))
            n_cells_local += 1
        else:
            for tri in skin_tris:
                joined = new.getFullJoin(
                    [int(placed_new[int(skin_row[int(v)])]) for v in tri])
                if len(joined) != 1:
                    failure = ("a skin triangle is not a face of the sewn "
                               "mesh; the gap was not sewn onto the layer.")
                    break
                out_skin.setValue(int(joined[0]), int(label_value))
                n_skin_local += 1
    failures = comm.allgather(failure)
    real = [f for f in failures if f]
    if real:
        raise RuntimeError(real[0])

    # ------------------------------------------------------- global gates
    counts = np.array([n_cells_local, n_skin_local, len(victims_arr),
                       len(placed)], dtype=np.int64)
    comm.Allreduce(MPI.IN_PLACE, counts, op=MPI.SUM)
    n_zone, n_skin_faces, n_removed, n_placed = (int(x) for x in counts)
    if n_zone != len(asm_tets):
        raise RuntimeError(
            f"{n_zone} zone cells labelled for {len(asm_tets)} assembly "
            "cells given.")
    if n_skin_faces != len(skin_tris):
        raise RuntimeError(
            f"{n_skin_faces} skin faces labelled for {len(skin_tris)} "
            "given.")

    volume_after = np.array([_owned_cell_volume(new)], dtype=float)
    comm.Allreduce(MPI.IN_PLACE, volume_after, op=MPI.SUM)
    if abs(volume_after[0] - volume_before[0]) > 1e-9 * volume_before[0]:
        raise RuntimeError(
            f"the placement changed the domain volume: "
            f"{volume_before[0]:.12f} -> {volume_after[0]:.12f}")

    owned = np.asarray(_owned_stratum_counts(new), dtype=np.int64)
    comm.Allreduce(MPI.IN_PLACE, owned, op=MPI.SUM)
    nv_g, ne_g, nf_g, nc_g = (int(x) for x in owned)
    if nv_g - ne_g + nf_g - nc_g != 1:
        raise RuntimeError(
            f"the sewn mesh has global Euler number "
            f"{nv_g - ne_g + nf_g - nc_g}, not 1")

    after = _interior_face_counts_3d(new)
    for key, before in held_counts.items():
        now = after.get(key, 0)
        # A second zone under the same name may GROW the skin stratum; any
        # other label must come through with exactly the faces it had.
        if now < before or (now != before
                            and key != (skin_label, int(label_value))):
            raise RuntimeError(
                f"placing {label!r} would leave the surface {key[0]!r} with "
                f"{now} interior faces instead of {before}.")

    min_vol = np.array([_owned_min_cell_volume(new)], dtype=float)
    comm.Allreduce(MPI.IN_PLACE, min_vol, op=MPI.MIN)

    info = {"n_zone_cells": n_zone, "n_skin_faces": n_skin_faces,
            "n_placed": n_placed, "n_removed": n_removed,
            "min_volume": float(min_vol[0])}
    if verbose:
        uw.pprint(f"[place_thin_volume {label!r}] {info['n_zone_cells']} "
                  f"zone cells, {info['n_skin_faces']} skin faces; placed "
                  f"{info['n_placed']} vertices, removed "
                  f"{info['n_removed']}")
    return new, info


def _cell_volumes_signed6(dm):
    X = _coords(dm)
    vS, _vE = dm.getDepthStratum(0)
    cells = _tet_vertices(dm)
    P = X[cells + 0]
    return np.einsum("ij,ij->i",
                     np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]),
                     P[:, 3] - P[:, 0])


def _owned_cell_volume(dm):
    """Total volume of this rank's cells. Cells are never shared, so the
    allreduced sum is the domain volume."""
    return float(np.abs(_cell_volumes_signed6(dm)).sum() / 6.0)


def _owned_min_cell_volume(dm):
    v = np.abs(_cell_volumes_signed6(dm)) / 6.0
    return float(v.min()) if len(v) else np.inf
