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
Serial and parallel, gather-first (the fault's star is redistributed onto
one rank, the surgery runs there, every rank rebuilds collectively; no
placed point is ever shared). One parallel restriction in 2-D: a surface
whose END reaches the domain wall carries the end-settling machinery
(wall-vertex slides, facet splits) whose collective form is not built —
refused with the reason; interior surfaces, the fault case, run at any
rank count.

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


def _gmsh_fill_2d(Xall, ring, chain, holes=(), size_of=None):
    """Triangulate the cavity with gmsh: the ring verbatim, the chain embedded.

    The ring — the cavity boundary, anticlockwise — goes in as a discrete
    curve carrying its existing segmentation; the surface's chain as a second
    discrete curve embedded in the plane surface. ``chain`` is one polyline
    of ``Xall`` indices or a sequence of polylines (a trace crossing a
    collar piece in several runs), each its own discrete curve. A chain end
    that IS a ring vertex (a crossing, or an end on the wall) is expressed
    by having the chain's elements reference the ring's own node tag — no
    duplicate node, no snapping — and a free end is a tip, gmsh's ordinary
    free-end embed.

    ``holes`` are further closed loops of ``Xall`` indices excluded from the
    fill — the 2-D thin volume's skin, meshed elsewhere and sewn on — each a
    discrete curve of its own, verbatim like the ring.

    ``size_of`` is an optional callable ``(x, y) -> h`` giving the target
    mesh size at a point of the plane — a VARIABLE-resolution fill (a
    sheet fine near its shallow tip, coarse at depth). Without it the
    fill takes its size from the constrained curves' own segmentation.

    Everything is gated, because a fill that looks plausible and is not
    conforming is worse than a refusal: zero moved nodes, every input segment
    an edge of the triangulation, triangles present and returned anticlockwise.

    Returns ``(tris, extra)``: triangles indexing ``Xall`` first and then the
    ``extra`` interior points gmsh inserted (rare at cavity sizes, legal
    always — they become ordinary mesh vertices).
    """
    import gmsh

    ring = [int(v) for v in ring]
    if chain is None or not len(chain):
        chains = []
    elif isinstance(chain[0], (list, tuple, np.ndarray)):
        chains = [[int(v) for v in c] for c in chain]
    else:
        chains = [[int(v) for v in chain]]
    flat_chain = [v for c in chains for v in c]
    holes = [[int(v) for v in loop] for loop in holes]
    if len(set(flat_chain)) != len(flat_chain):
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
    interior = [v for v in flat_chain if v not in tag_of]
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

        chain_tags = []
        added = set(ring) | set(hole_nodes)
        for c in chains:
            line_tag = gmsh.model.addDiscreteEntity(1)
            own = [v for v in c if v not in added]
            added.update(own)
            if own:
                gmsh.model.mesh.addNodes(
                    1, line_tag, [tag_of[v] for v in own],
                    np.column_stack([Xall[own], np.zeros(len(own))])
                    .reshape(-1).tolist())
            cseg = np.array([[tag_of[a], tag_of[b]]
                             for a, b in zip(c[:-1], c[1:])],
                            dtype=np.int64)
            gmsh.model.mesh.addElementsByType(line_tag, 1, [],
                                              cseg.reshape(-1).tolist())
            chain_tags.append(line_tag)

        loops = [gmsh.model.geo.addCurveLoop([ring_tag])]
        loops += [gmsh.model.geo.addCurveLoop([t]) for t in hole_tags]
        surf = gmsh.model.geo.addPlaneSurface(loops)
        gmsh.model.geo.synchronize()
        if chain_tags:
            gmsh.model.mesh.embed(1, chain_tags, 2, surf)

        # Sizes bracketing what is already there: fine enough to accept the
        # chain's own spacing, coarse enough not to refine the cavity beyond
        # the surviving mesh around it. A size callback, when given, takes
        # over the interior grading (the bracket stays as the guard rail).
        constrained = (ring + ring[:1], *chains,
                       *[loop + loop[:1] for loop in holes])
        lengths = np.concatenate(
            [np.linalg.norm(np.diff(Xall[c], axis=0), axis=1)
             for c in constrained if len(c) > 1])
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5 * float(lengths.min()))
        gmsh.option.setNumber("Mesh.MeshSizeMax", 2.0 * float(lengths.max()))
        if size_of is not None:
            # the callback is AUTHORITATIVE for the interior: without these
            # gmsh extends the constrained curves' segmentation inward and
            # takes the minimum, and a graded callback has no effect
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.model.mesh.setSizeCallback(
                lambda dim, tag, x, y, z, lc: float(size_of(x, y)))
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
        for c in chains:
            wanted += list(zip(c, c[1:]))
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

    # The one rebuild for every dimension and rank count (the vertex SF is
    # attached before the interpolate — issue #520's property — and the
    # made cells are re-oriented to the kept convention at wiring).
    made = [tuple(int(v) if v < len(X) else -(int(v) - len(X) + 1)
                  for v in t) for t in tris]
    new_dm, point_map, placed_points = _rebuild_sewn(
        dm, drop, np.flatnonzero(victim), made, placed)
    pStart0 = dm.getChart()[0]

    def new_point(v):
        return (int(placed_points[v - len(X)]) if v >= len(X)
                else int(point_map[v + vS - pStart0]))

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


def _true_wall_segments_2d(dm, X):
    """The DOMAIN wall's edges as coordinate segments: support 1 AND unshared.

    A partition seam's edges also have local support 1; the shared test is
    what separates the wall from the seam (the 3-D lesson, one dimension
    down).
    """
    shared = _shared_point_flags(dm).astype(bool)
    pStart, _pEnd = dm.getChart()
    vS, _vE = dm.getDepthStratum(0)
    segs = []
    for f in range(*dm.getHeightStratum(1)):
        if len(dm.getSupport(f)) == 1 and not shared[f - pStart]:
            a, b = (int(p) for p in dm.getCone(f))
            segs.append((X[a - vS], X[b - vS]))
    return segs


def _place_one_parallel(dm, pts, label, label_value, clearance, spacing):
    """One INTERIOR polyline into a distributed mesh: gather-first.

    The parallel form of :func:`_place_one`, scoped to surfaces whose ends
    terminate inside the mesh — the fault case the lifecycle ruling needs.
    A surface reaching the domain wall carries the end-settling machinery
    (wall-vertex slides, facet splits) whose collective form is not built;
    it is refused here with the reason, and the serial path keeps it.

    The mechanism is the 3-D one verbatim: mark by distance, gather the
    region onto one rank, carve and fill there, rebuild collectively through
    the SF-first interpolate, label by the chain's new ids, gate globally.
    """
    comm = uw.mpi.comm
    pts = np.asarray(pts, dtype=float)[:, :2]

    vS, vE = dm.getDepthStratum(0)
    pStart, pEnd = dm.getChart()
    X = _coords(dm)[: vE - vS]
    cells = _cells_anticlockwise(dm, X)
    h_vertex, _hc = _vertex_h_3d(dm, cells, len(X))
    d_line = _distance_to_lines(X, [pts])

    h_stats = np.array([float(h_vertex.sum()), float(len(h_vertex))])
    comm.Allreduce(MPI.IN_PLACE, h_stats, op=MPI.SUM)
    h_glob = h_stats[0] / max(h_stats[1], 1.0)
    wall_segs = _true_wall_segments_2d(dm, X)
    end_wall = min(
        (float(np.linalg.norm(
            end - (A + np.clip(((end - A) @ (B - A))
                               / float((B - A) @ (B - A)), 0, 1) * (B - A))))
         for end in (pts[0], pts[-1]) for A, B in wall_segs),
        default=np.inf)
    end_wall = comm.allreduce(end_wall, op=MPI.MIN)
    if end_wall < 2.0 * h_glob:
        raise NotImplementedError(
            "parallel placement supports INTERIOR surfaces; an end this "
            "close to the domain wall needs the end-settling machinery, "
            "which is serial. Place wall-crossing surfaces before "
            "distributing, or in a serial pass.")

    area_before = np.array([float(cell_areas(dm).sum())
                            if len(cells) else 0.0])
    comm.Allreduce(MPI.IN_PLACE, area_before, op=MPI.SUM)

    mark = np.zeros(pEnd - pStart, dtype=np.int32)
    mark[np.flatnonzero(d_line < (clearance + 1.0) * h_vertex)
         + vS - pStart] = 1
    dm_work, moved = _gather_region(dm, mark)
    if moved:
        vS, vE = dm_work.getDepthStratum(0)
        pStart, pEnd = dm_work.getChart()
        X = _coords(dm_work)[: vE - vS]
        cells = _cells_anticlockwise(dm_work, X)
        h_vertex, _hc = _vertex_h_3d(dm_work, cells, len(X))
        d_line = _distance_to_lines(X, [pts])

    shared = _shared_point_flags(dm_work).astype(bool)
    on_wall = _true_wall_vertex_mask(dm_work, len(X))
    held_v, held_c = _interface_vertices_and_cells(dm_work, len(X),
                                                   len(cells))
    held_counts = _interface_facet_counts(dm_work)

    n_region = int((d_line < clearance * h_vertex).sum())
    owners = np.asarray(comm.allgather(n_region))
    if owners.sum() == 0:
        raise ValueError(
            "the surface meets no cell of this mesh: there is nothing to "
            "place it in.")
    target = int(np.argmax(owners))

    failure = None
    surgery = None
    if comm.rank == target:
        try:
            space = spacing if spacing is not None \
                else _spacing_near(dm_work, X, cells, pts)
            pts_r = _resample(pts, space)
            if not _inside_mesh(X, cells, pts_r).all():
                raise ValueError(
                    "the surface leaves the mesh; parallel placement takes "
                    "interior surfaces only.")

            beside_held = np.zeros(len(X), dtype=bool)
            beside_held[cells[held_c].ravel()] = True
            protected = on_wall | held_v | beside_held
            victim = ((_distance_to_lines(X, [pts_r])
                       < clearance * h_vertex) & ~protected)

            edge = np.linalg.norm(X[cells[:, 0]] - X[cells[:, 1]], axis=1)
            reachable = np.flatnonzero(
                _distance_to_lines(X[cells].mean(axis=1), [pts_r])
                < edge + space)
            crossed = _cells_meeting(X, cells, pts_r, reachable)
            drop = np.union1d(np.flatnonzero(victim[cells].any(axis=1)),
                              crossed)
            drop = drop[~held_c[drop]]
            if not len(drop):
                raise ValueError("the surface meets no cell of this mesh")
            need = victim[cells].any(axis=1)
            if (need & held_c).any():
                raise RuntimeError(
                    "the surface's cavity needs a cell that belongs to a "
                    "surface already embedded.")
            ring = _cavity_ring(cells, drop)
            if ring is None:
                raise RuntimeError(
                    "the cells cleared for the surface do not leave one "
                    "simple hole. Raise `clearance`.")
            if victim[np.asarray(ring)].any():
                raise RuntimeError(
                    "a deleted vertex is on the cavity boundary")
            ring_set = set(int(v) for v in ring)
            Xall = np.vstack([X, pts_r])
            chain = [len(X) + k for k in range(len(pts_r))]
            outside = [k for k in range(len(pts_r))
                       if not _inside_polygon(Xall[ring], pts_r[k])]
            if outside:
                raise RuntimeError(
                    f"{len(outside)} point(s) of the surface fall outside "
                    "the cavity cleared for it; raise `clearance`.")

            tris, extra = _gmsh_fill_2d(Xall, ring, chain)
            placed = np.vstack([pts_r, extra]) if len(extra) else pts_r

            # No surviving vertex without a surviving cell, and nothing the
            # surgery touches may be shared (the gather's contract).
            keep = np.ones(len(cells), dtype=bool)
            keep[drop] = False
            referenced = np.zeros(len(X), dtype=bool)
            if keep.any():
                referenced[cells[keep].ravel()] = True
            victim |= ~referenced & ~on_wall
            touched = set()
            cS0, _ = dm_work.getHeightStratum(0)
            for c in drop:
                for q in dm_work.getTransitiveClosure(int(c) + cS0)[0]:
                    touched.add(int(q))
            if any(shared[q - pStart] for q in touched):
                raise RuntimeError(
                    "place internal: the gathered region touches a shared "
                    "point; the gather mask under-reached.")

            made = [tuple(int(v) if v < len(X)
                          else -(int(v) - len(X) + 1) for v in t)
                    for t in tris]
            surgery = (np.flatnonzero(victim), drop, made, placed,
                       len(pts_r), len(extra))
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    failures = comm.allgather(failure)
    real = [f for f in failures if f]
    if real:
        raise RuntimeError(f"place_along_lines failed on the surgery "
                           f"rank: {real[0]}")

    if comm.rank == target:
        victims_arr, drop_arr, made, placed, n_chain, n_extra = surgery
    else:
        victims_arr = np.empty(0, dtype=np.int64)
        drop_arr = np.empty(0, dtype=np.int64)
        made = []
        placed = np.empty((0, 2), dtype=float)
        n_chain = n_extra = 0

    new_dm, point_map, placed_new = _rebuild_sewn(
        dm_work, drop_arr, victims_arr, made, placed)

    if not new_dm.hasLabel(label):
        new_dm.createLabel(label)
    n_facets_local = 0
    if comm.rank == target:
        chain_new = [int(placed_new[k]) for k in range(n_chain)]
        n_facets_local = _label_placed_edges(new_dm, chain_new, label,
                                             label_value)

    # The held gate, per rank: the mesh outside the gathered region is
    # untouched, so every rank's own interface counts must come through
    # unchanged (growing only for the label being written).
    after = _interface_facet_counts(new_dm)
    breach = None
    for key, before in held_counts.items():
        now = after.get(key, 0)
        if now < before or (now != before
                            and key != (label, int(label_value))):
            breach = (f"placing {label!r} would leave the surface "
                      f"{key[0]!r} with {now} facets instead of {before}.")
    breaches = comm.allgather(breach)
    real = [b for b in breaches if b]
    if real:
        raise RuntimeError(real[0])

    areas = cell_areas(new_dm)
    stats = np.array([float(areas.sum()) if len(areas) else 0.0,
                      float((areas <= 0.0).sum()) if len(areas) else 0.0,
                      float(n_facets_local), float(len(victims_arr)),
                      float(n_chain), float(n_extra)])
    comm.Allreduce(MPI.IN_PLACE, stats, op=MPI.SUM)
    if stats[1]:
        raise RuntimeError(f"{int(stats[1])} cell(s) of the result are "
                           "inverted")
    if abs(stats[0] - area_before[0]) > 1e-9 * area_before[0]:
        raise RuntimeError(
            f"the placement changed the domain area: {area_before[0]:.12f} "
            f"-> {stats[0]:.12f}")
    _validity_and_orientation_gates(new_dm, comm)

    return new_dm, {"n_placed": int(stats[4]),
                    "n_on_surface": 0,
                    "n_removed": int(stats[3]),
                    "n_fill_points": int(stats[5]),
                    "n_surface_facets": int(stats[2])}


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
    out = dm
    totals = {"n_placed": 0, "n_on_surface": 0, "n_removed": 0,
              "n_fill_points": 0, "n_surface_facets": 0}
    for pts in lines:
        if uw.mpi.size > 1:
            # Gather-first, interior surfaces only; self-gating (area,
            # inversion, validity battery — all collective).
            out, one = _place_one_parallel(
                out, np.asarray(pts, dtype=float)[:, :2],
                label, label_value, clearance, spacing)
        else:
            out, one = _place_one(out, np.asarray(pts, dtype=float)[:, :2],
                                  label, label_value, clearance, spacing,
                                  end_snap)
        for key in totals:
            totals[key] += one[key]

    if uw.mpi.size == 1:
        areas = cell_areas(out)
        over = sum(1 for f in range(*out.getHeightStratum(1))
                   if len(out.getSupport(f)) > 2)
        if over:
            raise RuntimeError(
                f"{over} facet(s) of the result have more than two cells: "
                "the retriangulated cavity is not conforming.")
        if (areas <= 0.0).any():
            raise RuntimeError(
                f"{int((areas <= 0.0).sum())} cell(s) of the result are "
                "inverted.")
        info = dict(totals, min_area=float(areas.min()),
                    min_angle=float(min_angles(out).min()))
    else:
        areas = cell_areas(out)
        local = np.array([float(areas.min()) if len(areas) else np.inf,
                          float(min_angles(out).min())
                          if len(areas) else np.inf])
        uw.mpi.comm.Allreduce(MPI.IN_PLACE, local, op=MPI.MIN)
        info = dict(totals, min_area=float(local[0]),
                    min_angle=float(local[1]))
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


def _interface_faces_3d(dm, exclude=()):
    """Interior faces carrying a non-topology label — an embedded surface.

    The 3-D analogue of :func:`reconnect._interface_edges`: in 3-D an
    interface is a surface and is identified by its FACES. Cell labels are
    volumes, not interfaces, and are skipped for the same reason as in 2-D;
    exterior faces are the domain's own walls and are handled separately.

    ``exclude`` names ``(label, value)`` pairs to leave out — a removal must
    not hold its own object's cells against itself.
    """
    fS, fE = dm.getHeightStratum(1)
    cS, cE = dm.getHeightStratum(0)
    exclude = set(exclude)
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
            if (name, int(val)) in exclude:
                continue
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


def _interior_face_counts_3d(dm, exclude=()):
    """{(label, value): count of interior faces} — the breach detector."""
    interface = _interface_faces_3d(dm, exclude=exclude)
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


def _reach_query(X, reach):
    """A reach-aware spatial hash over ``X``: query(lo, hi) -> indices.

    Points are binned on uniform grids, one per octave of their
    ``reach`` (the per-point distance beyond which the caller does not
    care), so a box query returns every point whose reach-padded
    position could intersect ``[lo, hi]`` — and, on a graded mesh, only
    those: fine-reach points bin finely and are returned only near the
    box, the few coarse-reach points bin coarsely. This is what keeps
    sheet-against-mesh sweeps O(near-band) instead of O(domain x sheet)
    (#613 — measured 3,217 s of a 3,705 s placement without it).
    """
    X = np.asarray(X, dtype=float)
    reach = np.broadcast_to(np.asarray(reach, dtype=float), (len(X),))
    if not len(X):
        return lambda lo, hi: np.empty(0, dtype=np.int64)
    origin = X.min(axis=0)
    r_pos = np.maximum(reach, 1e-30)
    octave = np.floor(np.log2(r_pos / r_pos.min())).astype(np.int64)
    grids = []
    for b in np.unique(octave):
        sel = np.flatnonzero(octave == b)
        pitch = float(r_pos[sel].max())
        keys = np.floor((X[sel] - origin) / pitch).astype(np.int64)
        bins = {}
        for i, k in zip(sel, map(tuple, keys)):
            bins.setdefault(k, []).append(int(i))
        grids.append((pitch, bins))

    def query(lo, hi):
        out = []
        for pitch, bins in grids:
            k_lo = np.floor((np.asarray(lo) - pitch - origin)
                            / pitch).astype(np.int64)
            k_hi = np.floor((np.asarray(hi) + pitch - origin)
                            / pitch).astype(np.int64)
            for kx in range(k_lo[0], k_hi[0] + 1):
                for ky in range(k_lo[1], k_hi[1] + 1):
                    for kz in range(k_lo[2], k_hi[2] + 1):
                        got = bins.get((kx, ky, kz))
                        if got:
                            out.extend(got)
        return np.asarray(out, dtype=np.int64)

    return query


def _sheet_distance_within(X, pts, tris, reach):
    """Point-to-sheet distance, EXACT wherever it is below ``reach``.

    ``reach`` (scalar or per-point) is the threshold the caller compares
    against; a point farther than its reach from every triangle gets a
    sentinel above its reach, never the true distance. Every caller
    thresholds where it reads this, so decisions match
    :func:`_sheet_distance` exactly at O(near-band) cost (#613).
    """
    X = np.asarray(X, dtype=float)
    tris = np.asarray(tris, dtype=np.int64)
    reach = np.broadcast_to(np.asarray(reach, dtype=float),
                            (len(X),)).copy()
    best = 2.0 * np.maximum(reach, 1e-30)
    if not len(X) or not len(tris):
        return best
    query = _reach_query(X, reach)
    for t in tris:
        A, B, C = pts[t[0]], pts[t[1]], pts[t[2]]
        P3 = np.array([A, B, C])
        idx = query(P3.min(axis=0), P3.max(axis=0))
        if not len(idx):
            continue
        best[idx] = np.minimum(
            best[idx], _sheet_distance(X[idx], pts, [t]))
    return best


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


def _nearest_facet(X, pts, tris):
    """``(distance, facet index)`` from each point of ``X`` to a
    triangulated sheet — :func:`_sheet_distance` keeping the argmin, for
    when the answer is WHICH facet a point lies on, not how far it is."""
    best = np.full(len(X), np.inf)
    which = np.zeros(len(X), dtype=np.int64)
    for k, t in enumerate(tris):
        A, B, C = pts[t[0]], pts[t[1]], pts[t[2]]
        ab, ac = B - A, C - A
        n = np.cross(ab, ac)
        nn = float(n @ n)
        rel = X - A
        d00, d01, d11 = float(ab @ ab), float(ab @ ac), float(ac @ ac)
        d20, d21 = rel @ ab, rel @ ac
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        inside = (v >= 0.0) & (w >= 0.0) & (v + w <= 1.0)
        d_plane = np.abs(rel @ n) / np.sqrt(nn)
        d_edge = np.full(len(X), np.inf)
        for P, Q in ((A, B), (B, C), (C, A)):
            e = Q - P
            u = np.clip(((X - P) @ e) / float(e @ e), 0.0, 1.0)
            d_edge = np.minimum(
                d_edge, np.linalg.norm(X - (P + u[:, None] * e), axis=1))
        d = np.where(inside, d_plane, d_edge)
        closer = d < best
        best[closer] = d[closer]
        which[closer] = k
    return best, which


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


def _assembly_components(cells):
    """Connected components of a standalone assembly mesh, by shared facets.

    Returns a 1-based component id per cell, the same on every rank (the
    assembly is broadcast, so this is a pure function of it). Two zones of
    a network that are fused touch through shared faces and are one
    component; zones a domain apart are separate ones, and #670 places
    each on a rank of its own.
    """
    from itertools import combinations
    n = len(cells)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    nv = cells.shape[1]
    owner_of_face = {}
    for c, cell in enumerate(cells):
        for f in combinations(sorted(int(v) for v in cell), nv - 1):
            other = owner_of_face.setdefault(f, c)
            if other != c:
                ra, rb = find(other), find(c)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(c) for c in range(n)], dtype=np.int64)
    _uniq, comp = np.unique(roots, return_inverse=True)
    return (comp + 1).astype(np.int32)


def _gather_region(dm, vertex_mark_chart, verbose=False):
    """Redistribute so every marked vertex's cell star (+1 layer) is one rank's.

    The one-region form of :func:`_gather_regions` (a 0/1 mask is one
    region). Returns ``(new_dm, n_moved)``: the global count of cells
    gathered, ``0`` when nothing moved (serial, or a region already
    interior to one rank), so callers can both branch on it and report
    it. The input is untouched.
    """
    ids = (np.asarray(vertex_mark_chart) != 0).astype(np.int32)
    work, n_region, _n_moved, _owner, _canon = _gather_regions(
        dm, ids, verbose=verbose, layers=1)
    return work, n_region


def _gather_regions(dm, vertex_region_chart, verbose=False, layers=1):
    """Redistribute so each marked REGION's cell star (+ ``layers``) is one rank's.

    A mask-driven port of the contact stream's ``_redistribute_fault_interior``
    (feature/fault-split-node c8693579 / 1d487319; the label-driven original
    is measured at np=2..8 with serial-identical topology), generalised to
    several regions (#670). ``vertex_region_chart`` is chart-length: ``0``
    for an unmarked point, ``k >= 1`` the region a vertex belongs to. Each
    region's star and layer go to ONE rank, chosen per region — the rank
    that already holds most of it — and a region that is already interior
    to one rank is left where it is. Two regions whose stars or layers
    touch (share a cell or a vertex, on any rank) are merged: their
    cavities would share ring points, so they must be one rank's. Only the
    marked stars move — everything else keeps its load-balanced home — via
    one shell partition.

    ``layers`` is the growth beyond the star, one by default: the star is
    the marked vertices' cells, and the layer makes every point in the
    closure of a star cell unshared. Both the placement and the split
    need it. Measured (#670): with ``layers=0`` the placement's own gate
    ("the gathered region touches a shared point") fires at np=2, 3 and
    4 on the thin-volume suite — the carve drops cells beyond the marked
    vertices' star, so the star alone under-reaches.

    Returns ``(new_dm, n_region, n_moved, owner, canon)``: the global size
    of the regions (star and layer — the seam rule's footprint, a function
    of the mesh alone), the count of cells that changed rank (``0`` when
    none did; the input dm is then returned untouched), ``owner`` mapping
    each merged region's id to its rank, and ``canon`` mapping every input
    region id to its merged id. The decisions are COLLECTIVE: overlaps and
    counts are gathered before any branch.
    """
    comm = dm.getComm().tompi4py()
    ids_in = np.asarray(vertex_region_chart, dtype=np.int32)
    n_ids = int(comm.allreduce(int(ids_in.max()) if ids_in.size else 0,
                               op=MPI.MAX))
    if n_ids == 0:
        raise ValueError("place_sheet: the sheet meets no cell on any rank")
    if comm.size == 1:
        canon = {k: k for k in range(1, n_ids + 1)}
        return dm, 0, 0, {k: 0 for k in canon}, canon

    work = dm.clone()
    cS, cE = work.getHeightStratum(0)
    vS, vE = work.getDepthStratum(0)
    pStart, pEnd = work.getChart()
    pairs = set()          # (a, b): regions that touch, to be merged

    def reconcile(chart):
        # every rank agrees on every point it can see; where a rank's own
        # id loses to a neighbour's under MAX the two regions touch
        before = chart.copy()
        after = _propagate_vertex(work, chart, MPI.MAX, np.maximum)
        touch = (before > 0) & (after != before)
        for a, b in zip(before[touch], after[touch]):
            pairs.add((int(a), int(b)))
        return after

    cell_region = np.zeros(cE - cS, dtype=np.int32)

    def claim_cells(chart):
        for v in range(vS, vE):
            k = int(chart[v - pStart])
            if k == 0:
                continue
            for q in work.getTransitiveClosure(v, useCone=False)[0]:
                if cS <= int(q) < cE:
                    c = int(q) - cS
                    if cell_region[c] and cell_region[c] != k:
                        pairs.add((int(cell_region[c]), k))
                    cell_region[c] = max(cell_region[c], k)

    claim_cells(reconcile(ids_in.copy()))
    # Growth layers: a point is unshared exactly when all its incident
    # cells are co-resident, so each layer makes the previous cells'
    # closures unshared.
    for _grow in range(int(layers)):
        layer = np.zeros(pEnd - pStart, dtype=np.int32)
        for c in np.flatnonzero(cell_region):
            k = int(cell_region[c])
            for q in work.getTransitiveClosure(int(c) + cS)[0]:
                if vS <= int(q) < vE:
                    i = int(q) - pStart
                    if layer[i] and layer[i] != k:
                        pairs.add((int(layer[i]), k))
                    layer[i] = max(layer[i], k)
        claim_cells(reconcile(layer))

    # merge touching regions: union-find on the gathered pair set, the same
    # on every rank
    parent = list(range(n_ids + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in sorted(set().union(*comm.allgather(pairs))):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    canon = {k: find(k) for k in range(1, n_ids + 1)}
    for c in np.flatnonzero(cell_region):
        cell_region[c] = canon[int(cell_region[c])]

    regions = sorted(set(canon.values()))
    local = np.array([int((cell_region == k).sum()) for k in regions],
                     dtype=np.int64)
    counts = np.asarray(comm.allgather(local))          # (size, n_regions)
    if counts.sum() == 0:
        raise ValueError("place_sheet: the sheet meets no cell on any rank")
    owner = {}
    assign = np.full(cE - cS, comm.rank, dtype=np.int32)
    n_moved = 0
    for j, k in enumerate(regions):
        col = counts[:, j]
        owner[k] = int(np.argmax(col))
        if np.count_nonzero(col) <= 1:
            # star and layer already interior to one rank: the seam rule
            # holds where the mesh stands, nothing moves (#670)
            continue
        n_moved += int(col.sum() - col[owner[k]])
        assign[cell_region == k] = owner[k]
    n_region = int(counts.sum())
    if n_moved == 0:
        if verbose:
            uw.pprint(f"[place_sheet] {len(regions)} region(s), "
                      f"{n_region} cells, already interior to their ranks; "
                      f"no gather")
        return dm, n_region, 0, owner, canon

    order = np.argsort(assign, kind="stable").astype(np.int32)
    sizes = np.bincount(assign, minlength=comm.size).astype(np.int32)
    part = work.getPartitioner()
    part.setType(PETSc.Partitioner.Type.SHELL)
    part.setShellPartition(comm.size, sizes=sizes, points=order)
    work.distribute()
    if verbose:
        uw.pprint(f"[place_sheet] gathered {n_moved} cells: "
                  + ", ".join(f"region {k} ({int(counts[:, j].sum())} "
                              f"cells) onto rank {owner[k]}"
                              for j, k in enumerate(regions)))
    return work, n_region, n_moved, owner, canon


def _carve_cavity_3d(dm, X, cells, sheet_pts, sheet_tris, clearance,
                     held_cells, h_vertex, on_wall, shared_chart,
                     open_deletable=None, open_near=None):
    """Victims, dropped tets and the closed cavity shell around the sheet.

    The same two-part rule as 2-D — vertices within the clearance go, and any
    tet the sheet passes through goes (all four corners can sit outside the
    clearance while the sheet crosses the interior) — with the same guards:
    wall vertices are never victims, cells of an embedded surface are held,
    a victim's whole star must be dropped, and the shell must be a closed
    manifold that never touches the domain wall. Rank-local: the caller
    guarantees (and this function asserts) that the whole region is interior
    to this rank — the gather's contract.

    An OUTCROP passes its frame rule as two vertex masks
    (:func:`_outcrop_frame_3d`, the same contract as
    :func:`_carve_around_volume_3d`): ``open_deletable`` — wall vertices the
    carve may take — and ``open_near`` — wall vertices near the outcrop,
    where the cavity may open into cap faces; the wall faces of dropped
    cells come back as ``cap_faces`` for the caller to pre-mesh. Returns
    ``(victims, drop_ids, shell, cap_faces)``.
    """
    from underworld3.utilities.edge_split import cell_diameters

    h_cell = cell_diameters(dm)
    d_sheet = _sheet_distance_within(X, sheet_pts, sheet_tris,
                                     clearance * h_vertex)
    on_open = (open_deletable if open_deletable is not None
               else np.zeros(len(X), dtype=bool))

    held_vertex = np.zeros(len(X), dtype=bool)
    if held_cells:
        for c in held_cells:
            held_vertex[cells[c]] = True
    victim = ((d_sheet < clearance * h_vertex)
              & (~on_wall | on_open) & ~held_vertex)

    drop = victim[cells].any(axis=1)
    cen = X[cells].mean(axis=1)
    cen_query = _reach_query(cen, h_cell)
    for t in sheet_tris:
        A, B, C = sheet_pts[t[0]], sheet_pts[t[1]], sheet_pts[t[2]]
        n = np.cross(B - A, C - A)
        n = n / np.linalg.norm(n)
        diam = max(np.linalg.norm(B - A), np.linalg.norm(C - A),
                   np.linalg.norm(C - B))
        centre = (A + B + C) / 3.0
        cand = cen_query(centre - diam, centre + diam)
        if not len(cand):
            continue
        near_c = (np.linalg.norm(cen[cand] - centre, axis=1)
                  < diam + h_cell[cand])
        sub0 = cand[near_c]
        if not len(sub0):
            continue
        s = (X[cells[sub0]] - A) @ n
        straddle = (s.max(axis=1) > 1e-12) & (s.min(axis=1) < -1e-12)
        sub = sub0[straddle]
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

    # The plane-straddle test covers triangle INTERIORS; a cell crossed by a
    # rim EDGE can slip it when the cell is smaller than the sheet spacing —
    # which the refill after a removal produces (measured: gmsh "a segment
    # and a facet intersect" on a sheet re-placed into a cleared region).
    # The centroid rule from the volume carve closes the gap: any cell whose
    # centre is within reach of the BOUNDED sheet is dropped.
    drop |= (_sheet_distance_within(cen, sheet_pts, sheet_tris,
                                    0.6 * h_cell) < 0.6 * h_cell)
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

    if not drop.any():
        raise ValueError("the sheet meets no cell of this mesh")

    shell, cap_faces, drop = _closed_shell_3d(
        dm, X, cells, drop, victim, held_cells, shared_chart, "sheet",
        open_vertex=open_near)

    # The straddle rule — or the shell growth — can drop the whole star of
    # a vertex that is NOT a victim; such a vertex is on no shell face and
    # would come through the rebuild as an ISOLATED point (global Euler 2,
    # not 1 — the measured defect class). Every surviving vertex must have
    # a surviving cell — except a PROTECTED wall vertex near the outcrop,
    # which is a COLLAR node: the cap is re-triangulated through it and the
    # fill's tets reference it (the volume carve's measured rule).
    referenced = np.zeros(len(X), dtype=bool)
    if (~drop).any():
        referenced[cells[~drop].ravel()] = True
    orphan = ~referenced & ~victim
    if open_near is not None:
        orphan &= ~(orphan & on_wall & ~on_open & open_near)
    if (orphan & on_wall & ~on_open).any():
        raise RuntimeError(
            "the sheet's cavity would strand a domain-wall vertex; the sheet "
            "must be interior, with clearance to spare")
    victim |= orphan
    return np.flatnonzero(victim), np.flatnonzero(drop), shell, cap_faces


def _orient_boundary_complex(verts, tris):
    """Orient a closed boundary complex so every facet normal points OUT of
    the domain.

    The gathered complex carries no orientation, so it is propagated by
    shared-edge parity within each connected component, and each component's
    global sign comes from its enclosed signed volume: the component
    enclosing the greatest volume is the outer boundary (normals away from
    the domain, signed volume positive), every other component bounds an
    interior cavity — a spherical shell's inner surface — whose
    domain-outward normals point INTO the cavity, so its signed volume is
    made negative. Returns the reoriented ``(nf, 3)`` triangle array.
    """
    tris = [list(map(int, t)) for t in np.asarray(tris)]
    edge_owner = {}
    for k, (a, b, c) in enumerate(tris):
        for e in ((a, b), (b, c), (c, a)):
            edge_owner.setdefault(frozenset(e), []).append(k)
    seen = np.zeros(len(tris), dtype=bool)
    comp = np.full(len(tris), -1, dtype=np.int64)
    n_comp = 0
    for start in range(len(tris)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp[start] = n_comp
        while stack:
            k = stack.pop()
            a, b, c = tris[k]
            for e in ((a, b), (b, c), (c, a)):
                for j in edge_owner[frozenset(e)]:
                    if j == k or seen[j]:
                        continue
                    ja, jb, jc = tris[j]
                    # Consistent orientation: the shared edge must appear in
                    # OPPOSITE directions in its two facets.
                    if e in {(ja, jb), (jb, jc), (jc, ja)}:
                        tris[j] = [ja, jc, jb]
                    seen[j] = True
                    comp[j] = n_comp
                    stack.append(j)
        n_comp += 1
    T = np.asarray(tris, dtype=np.int64)
    P = verts[T]
    vol6 = np.einsum("ij,ij->i", np.cross(P[:, 0], P[:, 1]), P[:, 2])
    comp_vol = np.array([vol6[comp == c].sum() for c in range(n_comp)])
    outer = int(np.argmax(np.abs(comp_vol)))
    for c in range(n_comp):
        if (comp_vol[c] > 0.0) != (c == outer):
            rows = comp == c
            T[rows] = T[rows][:, [0, 2, 1]]
    return T


def _boundary_signed_distance(X, verts, tris):
    """Signed distance to an ORIENTED closed complex: NEGATIVE inside.

    The sign comes from the nearest facet's outward normal; a tie at a
    crease is broken by whichever facet is truly nearest, which is
    unambiguous for query points off the surface by more than rounding.
    """
    d, at = _nearest_facet(X, verts, tris)
    T = verts[np.asarray(tris)[at]]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n = n / np.linalg.norm(n, axis=1)[:, None]
    side = np.einsum("ij,ij->i", X - T[:, 0], n)
    return np.where(side >= 0.0, d, -d)


def _concave_crease_depth(verts, tris):
    """Per-facet DEPTH of the deepest locally concave crease it touches.

    A crease between two facets of an ORIENTED complex is concave when
    either facet's far vertex lies on the OUTWARD side of the other's
    plane, and its depth is that offset: zero for a convex or coplanar
    boundary (a box, a uniformly-faceted sphere); the SAGITTA MISMATCH —
    metres at Earth scale — where a refined-and-snapped curved boundary
    changes resolution; the facet size itself on an inner boundary. The
    sequential plane clip over-cuts by at most this depth, so the caller
    weighs it against the setback rather than refusing on existence. The
    noise floor sits at rounding.
    """
    tris = np.asarray(tris, dtype=np.int64)
    T = verts[tris]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n = n / np.linalg.norm(n, axis=1)[:, None]
    depth = np.zeros(len(tris))
    edge_first = {}
    for t, tri in enumerate(tris):
        opp = {frozenset((int(tri[0]), int(tri[1]))): int(tri[2]),
               frozenset((int(tri[1]), int(tri[2]))): int(tri[0]),
               frozenset((int(tri[2]), int(tri[0]))): int(tri[1])}
        for e, far in opp.items():
            if e not in edge_first:
                edge_first[e] = (t, far)
                continue
            s, far_s = edge_first[e]
            ea, eb = (int(v) for v in e)
            scale = float(np.linalg.norm(verts[eb] - verts[ea]))
            d = max(float((verts[far] - T[s, 0]) @ n[s]),
                    float((verts[far_s] - T[t, 0]) @ n[t]))
            if d > 1e-12 * scale:
                depth[t] = max(depth[t], d)
                depth[s] = max(depth[s], d)
    return depth


def _split_safe_triangulation(pts, tris):
    """Give every all-rim face an interior vertex, by interior-edge splits.

    A face with ALL THREE vertices on the sheet's rim cannot be
    split-node duplicated (its two copies would carry the same vertex
    triple), and both the clip's corner polygons and a gmsh planar mesh
    of a polygon produce a few however fine the triangulation. Splitting
    the face's longest INTERIOR edge at its midpoint bisects both
    sharing faces — the midpoint is a new interior vertex, rim edges
    (the trace included) are untouched, and no child is worse-shaped
    than its parent (a centroid inside a sliver corner face was tried
    and drove the fill to 1e-23 cell volumes).
    """
    from collections import Counter

    P = [np.asarray(p, dtype=float) for p in pts]
    T = [[int(v) for v in t] for t in tris]
    while True:
        edge_use = Counter()
        for a, b, c in T:
            for e in ((a, b), (b, c), (c, a)):
                edge_use[(e[0], e[1]) if e[0] < e[1] else (e[1], e[0])] += 1
        rim_v = {v for e, k in edge_use.items() if k == 1 for v in e}
        bad = next((k for k, t in enumerate(T)
                    if all(v in rim_v for v in t)), None)
        if bad is None:
            break
        a, b, c = T[bad]
        splittable = [(u, w) for u, w in ((a, b), (b, c), (c, a))
                      if edge_use[(u, w) if u < w else (w, u)] == 2]
        if not splittable:
            raise RuntimeError(
                "a sheet face has no interior vertex and no interior "
                "edge; the sheet is a single sliver and cannot be placed "
                "as a splittable fault")
        u, w = max(splittable,
                   key=lambda e: float(np.linalg.norm(P[e[0]] - P[e[1]])))
        m = len(P)
        P.append(0.5 * (P[u] + P[w]))
        fresh = []
        for row in T:
            if u in row and w in row:
                fresh.append([m if v == w else v for v in row])
                row[:] = [m if v == u else v for v in row]
        T.extend(fresh)
    return np.asarray(P, dtype=float), np.asarray(T, dtype=np.int64)


def _resample_planar_sheet(pts, tris, size):
    """Re-triangulate a PLANAR sheet at the given size, rim geometry kept.

    The authored triangulation is DATA — its spacing is whatever the
    source provided — but the embedded fault's resolution must match the
    mesh it cuts, or the verbatim embed forces sliver cells around every
    mismatched sheet triangle (the ruling, 2026-08-19). The sheet's rim
    is compressed to its corners (exact, collinear runs removed — this
    is where a too-fine authored rim coarsens), each straight run is
    resampled at the target size, and the interior is re-meshed by gmsh
    in the sheet's own plane. Cut corners and quality are gmsh's
    business afterwards, not the clip's. Non-planar sheets refuse — a
    curved surface needs a parametric remesh, which is not built.

    ``size`` is a length, or a callable ``(x, y, z) -> h`` for a
    VARIABLE resolution — the middle ground: a fault fine only where it
    approaches the surface (where the carve needs its clearance) and
    coarsening with depth to match the mesh it cuts, so the cut never
    slices a giant cell yet the deep fault is not over-resolved.
    """
    pts = np.asarray(pts, dtype=float)
    tris = np.asarray(tris, dtype=np.int64)
    centre = pts.mean(axis=0)
    rel = pts - centre
    _u_svd, _s, vt = np.linalg.svd(rel, full_matrices=False)
    normal = vt[2]
    span = float(np.linalg.norm(rel, axis=1).max())
    off = rel @ normal
    if float(np.abs(off).max()) > 1e-9 * max(span, 1.0):
        raise NotImplementedError(
            "resampling a non-planar sheet is not built (deviation "
            f"{float(np.abs(off).max()):.2e} from the fitted plane); "
            "author the sheet at the target size instead.")
    e0, e1 = vt[0], vt[1]

    from collections import Counter
    edge_use = Counter()
    for a, b, c in tris:
        for e in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            edge_use[tuple(sorted(e))] += 1
    loops = _skin_loops([e for e, k in edge_use.items() if k == 1],
                        what="the sheet's rim")

    size_at = size if callable(size) else (lambda x, y, z: float(size))

    resampled = []
    for loop in loops:
        corners = _compress_collinear_loop(pts[np.asarray(loop)])
        ring = []
        n_c = len(corners)
        for i in range(n_c):
            A, B = corners[i], corners[(i + 1) % n_c]
            # Local target along the run: sampled at both ends and the
            # middle, subdivided to the finest of them — a straight run
            # spanning shallow to deep takes the shallow (fine) budget,
            # which errs toward resolution, never toward a giant segment.
            probes = [A, 0.5 * (A + B), B]
            h_run = min(float(size_at(*p)) for p in probes)
            n_seg = max(1, int(np.ceil(np.linalg.norm(B - A) / h_run)))
            ring += [A + (k / n_seg) * (B - A) for k in range(n_seg)]
        resampled.append(np.asarray(ring))

    def area2(P3):
        q = np.column_stack([(P3 - centre) @ e0, (P3 - centre) @ e1])
        return 0.5 * float(q[:, 0] @ np.roll(q[:, 1], -1)
                           - q[:, 1] @ np.roll(q[:, 0], -1))

    order = sorted(range(len(resampled)),
                   key=lambda k: -abs(area2(resampled[k])))
    P3 = np.vstack([resampled[k] for k in order])
    P2 = np.column_stack([(P3 - centre) @ e0, (P3 - centre) @ e1])
    def oriented(ids, anticlockwise):
        q = P2[np.asarray(ids)]
        a = 0.5 * float(q[:, 0] @ np.roll(q[:, 1], -1)
                        - q[:, 1] @ np.roll(q[:, 0], -1))
        return ids if (a > 0.0) == anticlockwise else ids[::-1]

    ring_ids, holes, start = [], [], 0
    for j, k in enumerate(order):
        ids = list(range(start, start + len(resampled[k])))
        start += len(resampled[k])
        if j == 0:
            ring_ids = oriented(ids, True)
        else:
            holes.append(oriented(ids, False))
    size_2d = (None if not callable(size)
               else (lambda x, y: size_at(*(centre + x * e0 + y * e1))))
    new_tris, extra2 = _gmsh_fill_2d(P2, ring_ids, None, holes=holes,
                                     size_of=size_2d)
    lifted = np.vstack([P3, centre + extra2 @ np.vstack([e0, e1])]) \
        if len(extra2) else P3
    return _split_safe_triangulation(
        lifted, np.asarray(new_tris, dtype=np.int64))


def _clip_sheet_to_boundary(pts, tris, dom_verts, dom_tris, tol=1e-12,
                            setback=0.0):
    """Clip a triangulated sheet against the mesh's OWN boundary complex.

    ``setback > 0`` clips against the boundary OFFSET INWARD by that
    distance (each region plane shifted along its inward normal): the
    sheet stops deliberately short of the surface — a BLIND fault, whose
    rim is strictly interior and therefore splittable — rather than
    outcropping. On a curved boundary the offset is the shifted faceted
    planes' intersection, within a sagitta of the true offset surface,
    which is the right contract for "stop an element or two below".

    The specify-long contract (ruling, 2026-08-11) on a general boundary:
    fault surfaces are defined generously PAST the domain and prep trims
    them. Sheet vertices classify by signed distance to the oriented
    complex; a mixed triangle is cut sequentially by the planes of the
    boundary facets it can cross, keeping the inside — exact where the
    crossed boundary is locally CONVEX (a box wall, a sphere's outer
    surface), which covers the outcrop cases. A locally concave crossing
    over-cuts by at most the crease's DEPTH
    (:func:`_concave_crease_depth`), so it is allowed only when the
    setback dwarfs that depth (a resolution transition of a snapped
    curved boundary, metres under a kilometres-deep blind tip) and
    refused otherwise (an inner boundary; ANY concavity at setback zero,
    where cut nodes must land on the complex exactly — the true polyline
    cut is not built). The cutting planes are the COPLANAR REGIONS'
    (:func:`_coplanar_regions`) — a box wall cuts as one plane however it
    is faceted — and cut nodes are computed from canonically ordered edge
    endpoints and interned by the ``(edge, region)`` identity, so the two
    triangles sharing a cut edge produce the SAME node and the clipped
    sheet stays conforming; each cut node lands exactly on its region's
    plane. Uncut geometry is preserved verbatim. Every kept node is gated
    inside the domain on exit — a violation means the crossing was not
    locally convex at the sheet's scale and is a refusal, not a tolerance.
    Returns ``(pts, tris)``.
    """
    pts_in = np.asarray(pts, dtype=float)
    tris_in = np.asarray(tris, dtype=np.int64)
    setback = float(setback)
    oriented = _orient_boundary_complex(dom_verts, dom_tris)
    sd = _boundary_signed_distance(pts_in, dom_verts, oriented)
    inside = sd < tol - setback
    if inside.all():
        return pts_in, tris_in

    DT = dom_verts[oriented]
    f_lo = DT.min(axis=1)
    f_hi = DT.max(axis=1)
    concave_depth = _concave_crease_depth(dom_verts, oriented)
    # The cutting planes are the coplanar regions': two coplanar facets of
    # one wall must cut with the IDENTICAL plane, or the two triangles
    # sharing a cut edge key their cut by different facets and the sheet
    # tears along a duplicated node (measured on the box top wall).
    region, planes_of = _coplanar_regions(dom_verts, oriented)

    out_pts = [p for p in pts_in]
    cut_id = {}

    def cut_on(key, A, B, r):
        # Interned by identity, endpoints canonical: the two triangles
        # sharing a cut compute the bitwise-identical point, ONE node, and
        # the clipped sheet stays conforming. The plane is the region's,
        # shifted inward by the setback.
        if key not in cut_id:
            anchor, nrm = planes_of[int(r)]
            oa = float((A - anchor) @ nrm) + setback
            ob = float((B - anchor) @ nrm) + setback
            p = A + (oa / (oa - ob)) * (B - A)
            p -= (float((p - anchor) @ nrm) + setback) * nrm
            cut_id[key] = len(out_pts)
            out_pts.append(p)
        return cut_id[key]

    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    out_tris = []
    for tri in tris_in:
        flags = inside[tri]
        if flags.all():
            out_tris.append([int(v) for v in tri])
            continue
        if not flags.any():
            continue
        # The facet planes this triangle can cross: bbox overlap, widened
        # by the setback — a shifted plane cuts a triangle that never
        # comes within bbox reach of the facet itself.
        P3 = pts_in[tri]
        t_lo = P3.min(axis=0) - tol - setback
        t_hi = P3.max(axis=0) + tol + setback
        near = np.flatnonzero(((f_lo <= t_hi) & (f_hi >= t_lo)).all(axis=1))
        # The sequential plane clip over-cuts a locally concave crossing
        # by at most the crease DEPTH. Weighed against the setback: metres
        # of sagitta mismatch on a refined-and-snapped sphere are harmless
        # under a kilometres-deep blind tip, while an inner boundary's
        # facet-scale concavity refuses at any realistic setback — and at
        # setback zero (an outcrop, where cut nodes must land ON the
        # complex exactly) any concavity at all refuses, as before.
        delta = float(concave_depth[near].max()) if len(near) else 0.0
        if delta > 0.2 * setback + 1e-15:
            raise NotImplementedError(
                f"the sheet crosses the domain boundary near a locally "
                f"concave crease of depth {delta:.2e} (an inner boundary, "
                f"or a resolution transition of a snapped curved "
                f"boundary); the sequential plane clip can over-cut by "
                f"that much, which setback={setback:.2e} does not cover. "
                "Raise the setback above ~5x the depth, refine the "
                "boundary uniformly under the crossing, or wait for the "
                "polyline cut.")
        # Each polygon vertex carries the ORIGINAL sheet edges it lies on:
        # a cut on a triangle side is re-derived from that side's original
        # endpoints and interned by (edge, region), so the neighbouring
        # triangle — whose side may be truncated differently — produces the
        # bitwise-identical node (a sub-segment interpolation differs by
        # rounding, measured as ~1e-17 duplicate pairs on the sphere). A
        # cut on a CHORD (a previous cut's trace across the interior, i.e.
        # a crease crossing) is triangle-local and keys by its endpoints.
        a0, b0, c0 = (int(v) for v in tri)
        sides = {a0: {edge_key(a0, b0), edge_key(a0, c0)},
                 b0: {edge_key(a0, b0), edge_key(b0, c0)},
                 c0: {edge_key(a0, c0), edge_key(b0, c0)}}
        poly = [pts_in[v].copy() for v in (a0, b0, c0)]
        rows = [a0, b0, c0]
        srcs = [sides[a0], sides[b0], sides[c0]]
        for r in sorted({int(region[f]) for f in near}):
            anchor, nrm = planes_of[r]
            offs = [float((q - anchor) @ nrm) + setback for q in poly]
            if all(o > -tol for o in offs):     # wholly outside (or on)
                poly, rows, srcs = [], [], []
                break
            if all(o < tol for o in offs):      # wholly inside: no cut
                continue
            new_poly, new_rows, new_srcs = [], [], []
            m = len(poly)
            for i in range(m):
                j = (i + 1) % m
                oi, oj = offs[i], offs[j]
                if oi < tol:
                    new_poly.append(poly[i])
                    new_rows.append(rows[i])
                    new_srcs.append(srcs[i])
                if (oi < -tol and oj > tol) or (oi > tol and oj < -tol):
                    common = srcs[i] & srcs[j]
                    if common:
                        e, = common
                        w = cut_on((e[0], e[1], r), pts_in[e[0]],
                                   pts_in[e[1]], r)
                        src = {e}
                    else:
                        lo_r, hi_r = edge_key(rows[i], rows[j])
                        w = cut_on(('x', lo_r, hi_r, r),
                                   np.asarray(out_pts[lo_r]),
                                   np.asarray(out_pts[hi_r]), r)
                        src = set()
                    new_poly.append(np.asarray(out_pts[w]))
                    new_rows.append(w)
                    new_srcs.append(src)
            poly, rows, srcs = new_poly, new_rows, new_srcs
        if len(poly) < 3:
            continue
        for k in range(1, len(rows) - 1):
            if len({rows[0], rows[k], rows[k + 1]}) == 3:
                out_tris.append([rows[0], rows[k], rows[k + 1]])

    if not out_tris:
        raise ValueError("the sheet lies entirely outside the domain")
    used = sorted({v for t in out_tris for v in t})
    remap = {v: i for i, v in enumerate(used)}
    new_pts = np.array([out_pts[v] for v in used])
    new_tris = np.array([[remap[v] for v in t] for t in out_tris],
                        dtype=np.int64)

    new_pts, new_tris = _split_safe_triangulation(new_pts, new_tris)

    sd_out = _boundary_signed_distance(new_pts, dom_verts, oriented)
    if (sd_out > 1e-9 - setback).any():
        raise RuntimeError(
            "the clip kept a sheet node outside the domain (or inside the "
            "setback strip); the boundary is not locally convex at the "
            "crossing's scale, or the crossing spans facets the clip did "
            "not see. A defect, not a tolerance.")
    return new_pts, new_tris


def _sheet_boundary_intersection_chain(pts, tris, dom_verts, dom_tris):
    """The sheet's intersection polyline with the boundary — the
    surface-trace LOCATOR, by contouring the boundary signed distance
    over the sheet's own triangulation (marching triangles).

    Robust by construction on ANY boundary — concave, resolution-graded,
    snapped — because every crossing point is interpolated on a sheet
    EDGE from that edge's two vertex distances: the two triangles
    sharing the edge produce the identical point, so the polyline chains
    exactly with no tolerance welding (a direct triangle-triangle
    intersection was tried first and fragmented into hundreds of
    components at crease endpoints). Accuracy is the linear interpolant's
    — O(spacing^2 / R), metres at Earth scale — which is the locator
    contract: the deliberate blind-fault workflow needs to know WHERE
    the sheet would daylight so it can stop short of it and put the
    damage region there, not to mesh against it. Returns the ordered
    ``(n, 3)`` polyline, or ``None`` when the sheet never reaches the
    boundary; refuses when the intersection is not one open chain.
    """
    pts = np.asarray(pts, dtype=float)
    oriented = _orient_boundary_complex(dom_verts, dom_tris)
    sd = _boundary_signed_distance(pts, dom_verts, oriented)
    outside = sd >= 0.0            # the tie sits with outside: 0 or 2
    if outside.all() or not outside.any():
        return None                # never reaches, or never inside

    cut = {}

    def crossing(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in cut:
            oa, ob = float(sd[key[0]]), float(sd[key[1]])
            w = oa / (oa - ob)
            cut[key] = pts[key[0]] + w * (pts[key[1]] - pts[key[0]])
        return key

    adj = {}
    for t in np.asarray(tris, dtype=np.int64):
        a, b, c = (int(v) for v in t)
        crossed = [crossing(u, w) for u, w in ((a, b), (b, c), (c, a))
                   if outside[u] != outside[w]]
        if not crossed:
            continue
        k0, k1 = crossed             # a consistent tie-break gives 0 or 2
        adj.setdefault(k0, []).append(k1)
        adj.setdefault(k1, []).append(k0)
    if not adj:
        return None
    ends = [k for k, ns in adj.items() if len(ns) == 1]
    if len(ends) != 2 or any(len(ns) > 2 for ns in adj.values()):
        raise NotImplementedError(
            "the sheet's intersection with the boundary is not one open "
            "chain; multiple or closed traces are not built.")
    chain, prev, cur = [ends[0]], None, ends[0]
    while cur != ends[1]:
        ns = adj[cur]
        nxt = ns[0] if ns[0] != prev else ns[1]
        chain.append(nxt)
        prev, cur = cur, nxt
    return np.asarray([cut[k] for k in chain])


def _outcrop_chain(pts, tris, dom_verts, dom_tris, tol=1e-9):
    """The clipped sheet's boundary polyline ON the domain boundary, ordered.

    Boundary edges of the clipped sheet whose ends AND midpoint lie on the
    domain's boundary complex form the outcrop trace — the membership rule
    of :func:`_split_skin_trace` one dimension down, metric but unambiguous:
    the clip put cut nodes ON the crossed facets, while an interior rim node
    is a sheet spacing away. One open chain is the supported case; multiple
    chains or a closed loop are refused with the reason. Returns the ordered
    chain of point rows, or ``None``.
    """
    from collections import Counter

    edge_count = Counter()
    for a, b, c in tris:
        for e in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            edge_count[tuple(sorted(e))] += 1
    boundary_edges = [e for e, k in edge_count.items() if k == 1]
    if not boundary_edges:
        return None
    on_v = _sheet_distance(pts, dom_verts, dom_tris) < tol
    both_on = [e for e in boundary_edges if on_v[e[0]] and on_v[e[1]]]
    if not both_on:
        return None
    mids = 0.5 * (pts[[e[0] for e in both_on]]
                  + pts[[e[1] for e in both_on]])
    mid_on = _sheet_distance(mids, dom_verts, dom_tris) < tol
    chain_edges = [e for e, ok in zip(both_on, mid_on) if ok]
    if not chain_edges:
        return None
    adj = {}
    for a, b in chain_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    ends = [v for v, ns in adj.items() if len(ns) == 1]
    if len(ends) != 2 or any(len(ns) > 2 for ns in adj.values()):
        raise NotImplementedError(
            "the sheet's trace on the wall is not one open chain; multiple "
            "or closed outcrops are not built.")
    chain, prev, cur = [ends[0]], None, ends[0]
    while cur != ends[1]:
        ns = adj[cur]
        nxt = ns[0] if ns[0] != prev else ns[1]
        chain.append(nxt)
        prev, cur = cur, nxt
    return chain


def _closed_shell_3d(dm, X, cells, drop, victim, held_cells, shared_chart,
                     noun, open_vertex=None):
    """The cavity shell over ``drop``, GROWN at pinch edges until manifold.

    Shared by every 3-D carve. The union of victim stars around any object
    can PINCH — a shell edge whose surrounding cells are part-dropped in two
    wedges — and after a removal has refilled a region, even a thin sheet's
    cavity can pinch against the new connectivity (measured: a sheet placed
    into a cleared region wedged where the same sheet placed into the
    original mesh did not). Growing the drop at every non-manifold edge
    merges the wedges; dropping more cells only enlarges the fill.

    ``open_vertex`` (a vertex mask) lets the cavity OPEN onto the boundary
    near an outcrop — the bowl. A dropped cell's wall face whose vertices
    all carry the mask becomes a CAP face (returned separately; the caller
    pre-meshes the cap); any other wall contact still refuses. The mask is
    the caller's frame rule: the wall plane for a box-framed sheet, the
    band-touched coplanar regions for a general zone. The manifold check
    runs on shell ∪ cap, which together must close.

    Refusals stay per-object worded via ``noun``; a wall or seam contact and
    a growth that would need a held cell are refusals, not growth. Returns
    ``(shell, cap_faces, drop)`` with ``drop`` possibly grown; ``cap_faces``
    is empty when ``open_vertex`` is None.
    """
    from collections import Counter

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

    held = np.zeros(len(cells), dtype=bool)
    if held_cells:
        held[list(held_cells)] = True

    for _round in range(20):
        shell = []
        cap_faces = []
        for f in range(fS, fE):
            support = face_support[f]
            n_in = sum(1 for c in support if drop[c])
            if n_in == 0:
                continue
            if len(support) == 1:
                if shared_chart[f - pStart]:
                    raise RuntimeError(
                        f"the {noun}'s cavity touches a partition seam "
                        "after the gather; the region marking under-reached. "
                        "A defect, not a configuration error.")
                verts = face_verts[f]
                if open_vertex is not None and all(
                        open_vertex[v] for v in verts):
                    cap_faces.append((f, verts))
                    continue
                raise RuntimeError(
                    f"the {noun}'s cavity reached the domain wall away from "
                    f"the outcrop; the {noun} must clear the other walls")
            if n_in == 1:
                shell.append((f, face_verts[f]))
        edge_count = Counter()
        for _f, verts in shell + cap_faces:
            a, b, c = sorted(verts)
            for e in ((a, b), (a, c), (b, c)):
                edge_count[e] += 1
        bad = [e for e, k in edge_count.items() if k != 2]
        if not bad:
            break
        for a, b in bad:
            grow = (cells == a).any(axis=1) & (cells == b).any(axis=1)
            if (grow & held).any():
                raise RuntimeError(
                    f"closing the {noun}'s cavity shell needs a cell held "
                    "for a surface already embedded; move the object away "
                    "or raise `clearance`.")
            drop |= grow
    else:
        raise RuntimeError(
            f"the {noun}'s cavity shell did not close in 20 growth rounds; "
            "raise `clearance`.")

    shell_verts = sorted({v for _f, verts in shell for v in verts})
    if victim[shell_verts].any():
        raise RuntimeError("a deleted vertex is on the cavity shell")
    return shell, cap_faces, drop


def _gmsh_fill_3d(shell_xyz, shell_tris, sheet_pts, sheet_tris, h, cap=None):
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

        cap_tag = None
        if cap is not None:
            # The OUTCROP cap, PRE-MESHED (a geo surface bounded by the
            # discrete rim RESAMPLES the rim — measured — so the cap arrives
            # discrete). Node references: rim -> shell tags, outcrop chain
            # -> sheet tags, interior extras -> new tags after everything.
            cap_tag = gmsh.model.addDiscreteEntity(2)
            extra_first = n_shell + n_sheet + 1
            if len(cap["extra_xyz"]):
                gmsh.model.mesh.addNodes(
                    2, cap_tag,
                    list(range(extra_first,
                               extra_first + len(cap["extra_xyz"]))),
                    np.asarray(cap["extra_xyz"], dtype=float)
                    .reshape(-1).tolist())

            n_rim = len(cap["rim_shell_local"])
            n_chain = len(cap["chain_sheet_local"])

            def cap_tag_of(k):
                if k < n_rim:
                    return int(cap["rim_shell_local"][k]) + 1
                if k < n_rim + n_chain:
                    return (n_shell
                            + int(cap["chain_sheet_local"][k - n_rim]) + 1)
                return extra_first + (k - n_rim - n_chain)

            gmsh.model.mesh.addElementsByType(
                cap_tag, 2, [],
                [cap_tag_of(int(v)) for t in cap["tris"] for v in t])

        loop = gmsh.model.geo.addSurfaceLoop(
            [shell_tag] + ([cap_tag] if cap_tag is not None else []))
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
        cap_out = None
        if cap_tag is not None:
            cet, _cei, cen_ = gmsh.model.mesh.getElements(2, cap_tag)
            for et, nodes in zip(cet, cen_):
                if et == 2:
                    cap_out = np.array([renum[int(t)] for t in nodes],
                                       dtype=np.int64).reshape(-1, 3)
        return points, tets, sheet_out, moved, n_shell, cap_out
    finally:
        gmsh.finalize()


def _attach_uninterp_vertex_sf(new, dm, v_old_to_compact, nc_new, nroots,
                               extra_leaves=()):
    """Give the uninterpolated plex its vertex star-forest, BEFORE interpolate.

    This ordering is the whole fix for issue #520. Interpolating first and
    attaching a star-forest afterwards leaves each rank's faces and edges
    with whatever cone order its own local interpolation chose — nothing
    ever reconciles a leaf's cone against its root's, ``DMPlexCheckFaces``
    fails at the seam, and P2 cross-seam assembly builds a wrong operator
    (measured: plain Stokes 13 s -> 3594 s at np=2). With the vertex SF in
    place, ``DMPlexInterpolate`` runs its distributed path: it creates the
    faces and edges, ORIENTS the interface cones consistently across ranks,
    and extends the star-forest to the new points itself.

    Cells are never shared, so the old SF's vertex leaves are the whole
    graph; face/edge leaves of the old mesh are skipped — interpolate
    recreates them. The owner's new index for each leaf arrives by the
    one-broadcast renumbering trick (the leaf set is unchanged, only
    numbers move). A shared vertex the surgery deleted must be deleted on
    every rank holding it (the seam-conforming placement synchronises its
    victims); its entry is dropped. ``extra_leaves`` are NEW shared points —
    ``(local_point, owner_rank, owner_point)`` in the uninterpolated
    numbering — the band's vertices that both sides of a seam use.
    """
    vS, vE = dm.getDepthStratum(0)
    pStart, pEnd = dm.getChart()
    sf = dm.getPointSF()
    try:
        _nroots, ilocal, iremote = sf.getGraph()
    except (ValueError, TypeError):
        return                            # unpopulated: nothing is shared

    root_new = np.full(pEnd - pStart, -1, dtype=np.int32)
    owned_vertices = np.flatnonzero(v_old_to_compact >= 0)
    root_new[owned_vertices + vS - pStart] = (
        nc_new + v_old_to_compact[owned_vertices]).astype(np.int32)
    leaf_new = np.full(pEnd - pStart, -1, dtype=np.int32)
    # COLLECTIVE: a rank sharing nothing still participates.
    sf.bcastBegin(MPI.INT32_T, root_new, leaf_new, MPI.REPLACE)
    sf.bcastEnd(MPI.INT32_T, root_new, leaf_new, MPI.REPLACE)

    new_sf = PETSc.SF().create(comm=dm.comm)
    if ilocal is None or not len(ilocal):
        local = np.asarray([int(lp) for lp, _o, _p in extra_leaves],
                           dtype=PETSc.IntType)
        remote = np.asarray([(int(o), int(p)) for _l, o, p in extra_leaves],
                            dtype=PETSc.IntType).reshape(-1, 2)
        order = np.argsort(local, kind="stable")
        new_sf.setGraph(nroots, local[order], remote[order].reshape(-1))
        new.setPointSF(new_sf)
        return

    leaves = np.asarray(ilocal, dtype=np.int64)
    is_vertex = (leaves >= vS) & (leaves < vE)
    vleaves = leaves[is_vertex]
    keep = v_old_to_compact[vleaves - vS]
    remote_index = leaf_new[vleaves - pStart]
    if ((keep < 0) != (remote_index < 0)).any():
        raise RuntimeError(
            "place_sheet internal: a shared vertex was deleted on one side "
            "of a seam only; the gather mask under-reached, or the seam "
            "victims were not synchronised.")
    alive = keep >= 0
    local = nc_new + keep[alive]
    remote = np.empty((int(alive.sum()) + len(extra_leaves), 2),
                      dtype=PETSc.IntType)
    remote[:int(alive.sum()), 0] = \
        np.asarray(iremote).reshape(-1, 2)[is_vertex, 0][alive]
    remote[:int(alive.sum()), 1] = remote_index[alive]
    local = list(local)
    for k, (lp, owner, op) in enumerate(extra_leaves):
        local.append(int(lp))
        remote[int(alive.sum()) + k] = (int(owner), int(op))
    local = np.asarray(local, dtype=PETSc.IntType)
    order = np.argsort(local, kind="stable")
    new_sf.setGraph(nroots, local[order], remote[order].reshape(-1))
    new.setPointSF(new_sf)


def _rebuild_sewn(dm, drop_cell_ids, victim_ids, made_cells, placed,
                  shared_rows=()):
    """Rebuild the local chart with cells replaced; every rank, collectively.

    The uninterpolated-cells + ``DMPlexInterpolate`` pattern (taken from
    fault_split.split_along_label_3d on feature/fault-split-node, recorded in
    the ledger): only cell-to-vertex cones are wired by hand — trivially
    orientation-free — and PETSc derives the faces, edges and every cone
    orientation. A rank with no surgery rebuilds its chart unchanged, because
    ``interpolate`` is collective and the star-forest of the interpolated
    mesh must be built by every rank together. The vertex star-forest is
    attached BEFORE the interpolate (:func:`_attach_uninterp_vertex_sf`) so
    the interface cones come out consistently ordered across ranks — the
    issue #520 fix.

    ``made_cells`` entries are mixed: a non-negative value is an OLD vertex
    point; ``-(k+1)`` is row ``k`` of ``placed``. ``shared_rows`` are placed
    rows that another rank owns — ``(row, owner_rank, owner_row)``, the
    seam-conforming placement's band vertices — and become leaves of the
    vertex star-forest before the interpolate. Returns the interpolated
    mesh, the chart point map (old -> new, -1 for deleted), and the new
    vertex ids of the placed rows.
    """
    dim = dm.getDimension()
    nvc = dim + 1
    pStart, pEnd = dm.getChart()
    cS, cE = dm.getHeightStratum(0)
    vS, vE = dm.getDepthStratum(0)
    eS, eE = dm.getDepthStratum(1)
    nv_old = vE - vS

    keep_cell = np.ones(cE - cS, dtype=bool)
    keep_cell[np.asarray(drop_cell_ids, dtype=np.int64)] = False
    keep_vertex = np.ones(nv_old, dtype=bool)
    keep_vertex[np.asarray(victim_ids, dtype=np.int64)] = False
    v_old_to_compact = -np.ones(nv_old, dtype=np.int64)
    v_old_to_compact[keep_vertex] = np.arange(int(keep_vertex.sum()))
    n_surv = int(keep_vertex.sum())
    placed = np.asarray(placed, dtype=float).reshape(-1, dim)

    cell_verts = np.array(
        [[int(p) - vS for p in dm.getTransitiveClosure(c)[0]
          if vS <= p < vE] for c in range(cS, cE)],
        dtype=np.int64).reshape(cE - cS, nvc)
    kept = cell_verts[keep_cell]
    nc_new = int(keep_cell.sum()) + len(made_cells)
    nv_new = n_surv + len(placed)

    # Orient every made cell to the KEPT cells' handedness before wiring.
    # DMPlexInterpolate derives face cones and orientations from the
    # cell-vertex cones but does NOT normalise the cells' own vertex
    # order; the fill's cells arrive in gmsh's convention, which is
    # opposite to the plex closure convention the kept cells carry, and
    # a mixed-handedness mesh assembles negative Jacobians (measured:
    # the first Stokes solve on the sewn mesh never converged, while
    # every abs()-based volume gate stayed green). Read the convention off
    # the mesh's own cells, never assume it.
    made_cells = np.asarray(made_cells, dtype=np.int64).reshape(-1, nvc)
    if len(made_cells) and len(kept):
        X_old = _coords(dm)[:nv_old]

        if dim == 3:
            def signed(P):
                return float(np.dot(np.cross(P[1] - P[0], P[2] - P[0]),
                                    P[3] - P[0]))
            flip = np.array([0, 1, 3, 2])
        else:
            def signed(P):
                return float((P[1][0] - P[0][0]) * (P[2][1] - P[0][1])
                             - (P[1][1] - P[0][1]) * (P[2][0] - P[0][0]))
            flip = np.array([0, 2, 1])

        ref_sign = np.sign(signed(X_old[kept[0]]))

        def xyz_of(x):
            return placed[-int(x) - 1] if x < 0 else X_old[int(x)]

        made_cells = made_cells.copy()
        for j, cell in enumerate(made_cells):
            P = np.array([xyz_of(v) for v in cell])
            if np.sign(signed(P)) != ref_sign:
                made_cells[j] = cell[flip]

    def v_uninterp(x):
        if x < 0:
            return nc_new + n_surv + (-int(x) - 1)
        return nc_new + int(v_old_to_compact[int(x)])

    new = PETSc.DMPlex().create(comm=dm.comm)
    new.setDimension(dim)
    new.setChart(0, nc_new + nv_new)
    for i in range(nc_new):
        new.setConeSize(i, nvc)
    new.setUp()
    for i, cell in enumerate(kept):
        new.setCone(i, [nc_new + int(v_old_to_compact[v]) for v in cell])
    for j, cell in enumerate(made_cells):
        new.setCone(int(keep_cell.sum()) + j,
                    [v_uninterp(v) for v in cell])
    new.symmetrize()
    new.stratify()
    if uw.mpi.size > 1:
        # every rank's first placed point in the uninterpolated numbering,
        # so a leaf can name its owner's row (COLLECTIVE, every rank)
        bases = uw.mpi.comm.allgather(int(nc_new + n_surv))
        extra = [(nc_new + n_surv + int(row), int(owner),
                  bases[int(owner)] + int(orow))
                 for row, owner, orow in shared_rows]
        _attach_uninterp_vertex_sf(new, dm, v_old_to_compact, nc_new,
                                   nc_new + nv_new, extra_leaves=extra)
    new.interpolate()

    vS2, vE2 = new.getDepthStratum(0)
    if (new.getHeightStratum(0) != (0, nc_new)
            or (vS2, vE2) != (nc_new, nc_new + nv_new)):
        raise RuntimeError(
            "rebuild internal: DMPlexInterpolate moved the cell or "
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

    # Old faces and edges (3-D) or edges (2-D) are recovered by JOINING
    # their surviving vertex tuples in the new chart (the contact stream's
    # recovery move). One that does not join back was interior to the cavity
    # and is legitimately gone; the breach detector downstream is what
    # confirms nothing LABELLED went with it.
    if dim == 3:
        strata = (dm.getHeightStratum(1), (eS, eE))
    else:
        strata = ((eS, eE),)
    for lo, hi in strata:
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
    # The point SF was attached before the interpolate and extended by it to
    # the faces and edges (with consistently oriented interface cones — the
    # issue #520 property). Do NOT rebuild it from the old chart here: that
    # was the defect. Mirror it onto the coordinate DM, which snapshots
    # whatever SF existed when it was created (the parallel-checkpoint fix
    # reconnect._install_point_sf documents).
    if uw.mpi.size > 1:
        new.getCoordinateDM().setPointSF(new.getPointSF())
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


def _validity_and_orientation_gates(new, comm):
    """PETSc's DMPlex validity battery + the handedness census, as one gate.

    The battery (maintainer ruling, 2026-08-10) runs via the options route —
    petsc4py 3.25 exposes no check methods. ``check_faces`` runs at EVERY
    rank count: it is issue #520's oracle (leaf/root cone agreement), the
    plain distributed box passes it cleanly, and the rebuilt mesh passes it
    once the vertex SF is attached before the interpolate. ``check_geometry``
    stays serial-only — measured false-positives on every rank of a plain
    distributed UnstructuredSimplexBox on this stack. Failures are reduced
    before anyone raises; a rank-local raise in parallel is a hang.

    The handedness census is the finding-1 gate from the #518 review:
    abs()-based volume checks are structurally blind to inversion — a
    mixed-handedness mesh passed every other gate here while assembling an
    indefinite operator.
    """
    _checks = ["check_symmetry", "check_skeleton", "check_pointsf",
               "check_faces"]
    if comm.size == 1:
        _checks.append("check_geometry")
    chk = new.clone()
    chk.setOptionsPrefix("uw_place_gate_")
    _opts = PETSc.Options()
    for _k in _checks:
        _opts[f"uw_place_gate_dm_plex_{_k}"] = ""
    _gate_fail = None
    try:
        chk.setFromOptions()
    except PETSc.Error as exc:
        _gate_fail = f"DMPlex validity check failed: {exc}"
    finally:
        for _k in _checks:
            del _opts[f"uw_place_gate_dm_plex_{_k}"]
    _fails = comm.allgather(_gate_fail)
    _real = [f for f in _fails if f]
    if _real:
        raise RuntimeError(f"the sewn mesh fails PETSc's checks: "
                           f"{_real[0]}")

    v6 = _cell_volumes_signed6(new)
    signs = np.array([float((v6 > 0).sum()), float((v6 < 0).sum()),
                      float((v6 == 0).sum())])
    comm.Allreduce(MPI.IN_PLACE, signs, op=MPI.SUM)
    if signs[2] or (signs[0] and signs[1]):
        raise RuntimeError(
            f"the sewn mesh has mixed cell orientation "
            f"({int(signs[0])} positive, {int(signs[1])} negative, "
            f"{int(signs[2])} degenerate) — the fill's cells were not "
            "oriented to the kept convention.")


def place_sheet(dm, points, triangles, label=CUT_LABEL, label_value=1,
                clearance=0.6, verbose=False, *, setback=0.0, size=None):
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
        form :class:`~underworld3.meshing.FaultSurface` carries.
        Non-self-intersecting, at least a cell from any embedded surface.
        The sheet may run PAST the domain: it is clipped against the mesh's
        own boundary, and a trace left ON the boundary becomes the sheet's
        OUTCROP — the trace chain's edges carry ``<label>_trace`` in the
        result and the wall's own labels are restored beside them, so the
        split machinery can carry a daylighting fault through the wall. The
        boundary need not be axis-aligned; a crossing at a locally concave
        crease (an inner boundary) is refused.
    label, label_value : str, int
        Label put on the sheet's faces in the result.
    clearance : float
        Delete a mesh vertex within this multiple of its local ``h`` of the
        sheet.
    verbose : bool
        Report the counts.
    setback : float, keyword-only
        Stop the sheet this far INSIDE the boundary instead of
        outcropping: the clip runs against the boundary offset inward, so
        the placed sheet is a BLIND fault whose rim is strictly interior
        — splittable by :func:`~underworld3.utilities.fault_split.split_along_label_3d`
        as it stands (the ruling: faults reach daylight blind, under a
        damage region, never split through the surface). The would-be
        intersection with the true boundary is still computed and
        returned as ``info["surface_trace"]`` — the locator for the
        damage region above. Use at least a couple of background cells,
        so the carve's cavity clears the wall.
    size : float, callable or None, keyword-only
        Re-triangulate the (clipped) sheet at this size before embedding
        — the counterpart of :func:`place_thin_volume`'s ``size``. The
        authored triangulation is DATA at whatever spacing the source
        provided, but the embedded fault's resolution must match the
        mesh it cuts, or the verbatim embed forces slivers around every
        mismatched triangle. A callable ``(x, y, z) -> h`` gives a
        VARIABLE resolution — the middle ground for a blind fault: fine
        only near the surface (where the carve needs its clearance),
        coarsening with depth to match the graded mesh around it.
        Planar sheets only, and not with an outcropping sheet (the trace
        chain must stay on the boundary complex verbatim); the rim's
        corner geometry is preserved exactly.

    Returns
    -------
    placed : PETSc.DMPlex
        A new mesh (distributed as the input was, with the sheet's region
        resident on one rank) in which every sheet triangle is a face
        carrying ``label``.
    info : dict
        Global counts: ``n_placed``, ``n_on_surface`` (always 0 in 3-D),
        ``n_removed``, ``n_surface_facets``, ``n_trace_edges``,
        ``min_volume``; with ``setback > 0`` also ``surface_trace`` — the
        would-be intersection polyline with the true boundary as a list
        of ``[x, y, z]`` points (or ``None`` if the sheet never reaches
        it).

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

    # The specify-long contract (ruling 2026-08-11): the sheet may extend
    # PAST the domain; it is clipped against the mesh's OWN boundary — any
    # boundary, axis-aligned or not — and a trace left ON the boundary
    # becomes the OUTCROP: the cavity opens into a bowl there and the wall
    # collar over it is remeshed to conform, with the trace chain embedded.
    # The boundary complex is gathered identically everywhere and the clip
    # is deterministic from it, so every rank computes the same sheet.
    dom_verts, dom_tris = _domain_boundary_facets(dm)
    setback = float(setback)
    if setback < 0.0:
        raise ValueError("setback must be non-negative")
    surface_trace = None
    if setback > 0.0:
        # The intersection the sheet is deliberately stopped short of:
        # computed from the UNCLIPPED input by direct triangle-triangle
        # intersection — the job is still to figure out where daylight
        # would be, so the damage region above the blind fault can be
        # placed there. The direct route needs no convexity, so it works
        # on the graded, snapped boundaries the blind workflow targets.
        chain_xyz = _sheet_boundary_intersection_chain(
            sheet_pts, sheet_tris, dom_verts, dom_tris)
        if chain_xyz is not None:
            surface_trace = chain_xyz.tolist()
    sheet_pts, sheet_tris = _clip_sheet_to_boundary(
        sheet_pts, sheet_tris, dom_verts, dom_tris, setback=setback)
    chain = _outcrop_chain(sheet_pts, sheet_tris, dom_verts, dom_tris)
    if size is not None:
        if chain is not None:
            raise NotImplementedError(
                "resampling an OUTCROPPING sheet is not built — its trace "
                "chain must stay on the boundary complex verbatim. Place "
                "blind with a setback, or pass size=None.")
        sheet_pts, sheet_tris = _resample_planar_sheet(
            sheet_pts, sheet_tris,
            size if callable(size) else float(size))

    # -------------------------------------------------- mark, then gather
    vS, vE = dm.getDepthStratum(0)
    pStart, pEnd = dm.getChart()
    X = _coords(dm)[: vE - vS]
    cells = _tet_vertices(dm)
    h_vertex, _h_cell = _vertex_h_3d(dm, cells, len(X))
    d_sheet = _sheet_distance_within(X, sheet_pts, sheet_tris,
                                     (clearance + 1.0) * h_vertex)
    # The gather mask covers everything the carve may DROP: the victims
    # (clearance) plus the crossed cells' vertices, which sit within a cell
    # diameter of the sheet. The seam rule needs the dropped cells, the ring
    # (their vertex star) and one more layer so the ring's points are
    # unshared; _gather_region grows the star and the layer from this mask,
    # so a wider mask here only moves cells the surgery never touches (#670:
    # a +2 margin gathered 91% of a fixture whose cavity was 6%). The carve
    # asserts nothing shared afterwards, so an under-reach is loud.
    mark = np.zeros(pEnd - pStart, dtype=np.int32)
    mark[np.flatnonzero(d_sheet < (clearance + 1.0) * h_vertex)
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
        d_sheet = _sheet_distance_within(X, sheet_pts, sheet_tris,
                                         (clearance + 1.0) * h_vertex)

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
            deletable = near = None
            dom_region = dom_planes = None
            if chain is not None:
                chain_edges = np.column_stack([chain[:-1], chain[1:]])
                dom_region, dom_planes = _coplanar_regions(dom_verts,
                                                           dom_tris)
                deletable, near, _regions = _outcrop_frame_3d(
                    X, on_wall, dom_verts, dom_tris, dom_region,
                    sheet_pts, chain_edges)
            victims, drop_ids, shell, cap_faces = _carve_cavity_3d(
                dm_work, X, cells, sheet_pts, sheet_tris, clearance,
                held_cells, h_vertex, on_wall, shared,
                open_deletable=deletable, open_near=near)
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

            shell_vert_ids = sorted(
                {v for _f, verts in shell for v in verts}
                | {v for _f, verts in cap_faces for v in verts})
            local = {v: i for i, v in enumerate(shell_vert_ids)}
            shell_xyz = X[shell_vert_ids]
            shell_tris = np.array([[local[v] for v in verts]
                                   for _f, verts in shell], dtype=np.int64)

            cap_payload = None
            removed_wall = []
            if cap_faces:
                # Labels each removed wall face carried, read PER FACE
                # before the rebuild forgets them — the bowl can span
                # several walls (a curved boundary's facets), so each new
                # wall triangle takes the labels of the old face it lies
                # on, not a set common to the whole bowl (the volume
                # path's rule).
                names = [dm_work.getLabelName(i)
                         for i in range(dm_work.getNumLabels())
                         if dm_work.getLabelName(i)
                         not in reconnect._TOPOLOGY_LABELS]
                for f, verts in cap_faces:
                    pairs_f = []
                    for name in names:
                        val = dm_work.getLabel(name).getValue(int(f))
                        if val >= 0:
                            pairs_f.append((name, int(val)))
                    removed_wall.append((X[np.asarray(verts)].copy(),
                                         pairs_f))

                cap_tris_mesh = np.array([verts for _f, verts in cap_faces],
                                         dtype=np.int64)
                d_cap, at_cap = _nearest_facet(
                    X[cap_tris_mesh].mean(axis=1), dom_verts, dom_tris)
                if (d_cap > 1e-9).any():
                    raise RuntimeError(
                        "an outcrop bowl wall face lies off the gathered "
                        "boundary complex; the wall mask and the complex "
                        "disagree")
                alive = np.ones(len(X), dtype=bool)
                alive[np.asarray(victims, dtype=np.int64)] = False
                cap_nodes, chain_nodes, cap_extra, cap_tris = \
                    _outcrop_collar_3d(
                        X, alive, cap_tris_mesh, dom_region[at_cap],
                        dom_planes, sheet_pts,
                        np.zeros((0, 3), dtype=np.int64),
                        np.zeros(0, dtype=np.int64), None, chain=chain)
                cap_payload = {
                    "rim_shell_local": [local[v] for v in cap_nodes],
                    "chain_sheet_local": list(chain_nodes),
                    "tris": cap_tris,
                    "extra_xyz": cap_extra,
                }

            fill = _gmsh_fill_3d(shell_xyz, shell_tris, sheet_pts,
                                 sheet_tris, h, cap=cap_payload)
            (fill_pts, fill_tets, sheet_out, moved_nodes, n_shell,
             cap_out) = fill
            if moved_nodes:
                raise RuntimeError(
                    f"the fill moved {moved_nodes} constrained node(s); the "
                    "cavity cannot be sewn back. A defect, not a tolerance.")
            if sheet_out is None or len(sheet_out) != len(sheet_tris):
                raise RuntimeError(
                    "the fill remeshed the sheet "
                    f"({0 if sheet_out is None else len(sheet_out)} "
                    f"triangles for {len(sheet_tris)} given).")
            if cap_payload is not None and (
                    cap_out is None
                    or len(cap_out) != len(cap_payload["tris"])):
                raise RuntimeError(
                    "the fill remeshed the outcrop cap "
                    f"({0 if cap_out is None else len(cap_out)} triangles "
                    f"for {len(cap_payload['tris'])} given).")
        # Exception, not just RuntimeError/ValueError: a raw gmsh error
        # (e.g. a PLC intersection) is a plain Exception, and an
        # uncaught raise on the surgery rank is a HANG for its peers —
        # every failure must become a collective refusal.
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"

    failures = comm.allgather(failure)
    real = [f for f in failures if f]
    if real:
        raise RuntimeError(f"place_sheet failed on the surgery rank: "
                           f"{real[0]}")

    # ------------------------------------------------ rebuild, every rank
    if comm.rank == target:
        fill_pts, fill_tets, sheet_out, _moved, n_shell, cap_out = fill
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

    new, point_map, placed_new = _rebuild_sewn(
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

    # The outcrop cap's new wall faces, relabelled EXPLICITLY with whatever
    # the replaced faces carried ("Top" and friends): joins cannot recover
    # new points, and an unlabelled patch of boundary silently loses its
    # Dirichlet conditions. Each new cap triangle takes the labels of the
    # removed face it lies on (the volume path's per-face rule — the bowl
    # can span several walls), whole face closures included; and the trace
    # chain's edges are labelled ``<label>_trace`` so the split machinery
    # can carry the fault THROUGH the wall.
    trace_label = label + "_trace"
    if not new.hasLabel(trace_label):
        new.createLabel(trace_label)
    pairs = comm.bcast(
        sorted({p for _tri, pairs_f in removed_wall for p in pairs_f})
        if comm.rank == target else None, root=target)
    n_cap_expect = comm.bcast(
        (len(cap_out) if (comm.rank == target and cap_out is not None)
         else 0), root=target)
    for name, val in (pairs or []):
        if not new.hasLabel(name):
            new.createLabel(name)
    n_cap_local = 0
    n_trace_local = 0
    if comm.rank == target and cap_out is not None:
        n_shell_ids = np.asarray(shell_vert_ids, dtype=np.int64)

        def new_id(v):
            if v < n_shell:
                old_pt = (int(n_shell_ids[v])
                          + dm_work.getDepthStratum(0)[0])
                return int(point_map[old_pt - pStart])
            return int(placed_new[v - n_shell])

        centres = np.array([fill_pts[np.asarray(t)].mean(axis=0)
                            for t in cap_out])
        old_pts = np.vstack([tri for tri, _p in removed_wall])
        old_tris = np.arange(3 * len(removed_wall)).reshape(-1, 3)
        d_old, at_old = _nearest_facet(centres, old_pts, old_tris)
        # A raise here is rank-local — a hang at np>=2 — so every refusal
        # in this block goes through the collective failure.
        if (d_old > 1e-9).any():
            failure = ("an outcrop cap triangle lies on no removed wall "
                       "face; the collar and the bowl disagree")
        else:
            for k, t in enumerate(cap_out):
                joined = new.getFullJoin([new_id(int(v)) for v in t])
                if len(joined) != 1:
                    failure = ("an outcrop cap triangle is not a face of "
                               "the sewn mesh; the cap was not sewn onto "
                               "the wall.")
                    break
                for q in new.getTransitiveClosure(int(joined[0]))[0]:
                    for name, val in removed_wall[int(at_old[k])][1]:
                        new.getLabel(name).setValue(int(q), int(val))
                n_cap_local += 1
            else:
                out_trace = new.getLabel(trace_label)
                for a, b in zip(chain[:-1], chain[1:]):
                    joined = new.getFullJoin([int(placed_new[int(a)]),
                                              int(placed_new[int(b)])])
                    if len(joined) != 1:
                        failure = ("a trace edge is not an edge of the "
                                   "sewn mesh; the trace was not sewn "
                                   "onto the wall.")
                        break
                    for q in new.getTransitiveClosure(int(joined[0]))[0]:
                        out_trace.setValue(int(q), int(label_value))
                    n_trace_local += 1
    failures = comm.allgather(failure)
    real = [f for f in failures if f]
    if real:
        raise RuntimeError(real[0])
    n_cap = int(comm.allreduce(n_cap_local, op=MPI.SUM))
    if n_cap != n_cap_expect:
        raise RuntimeError(
            f"{n_cap} outcrop cap faces relabelled for {n_cap_expect} "
            "given.")
    n_trace = int(comm.allreduce(n_trace_local, op=MPI.SUM))
    # The chain is computed identically on every rank, before the gather.
    n_trace_expect = (len(chain) - 1
                      if (chain is not None and n_cap_expect) else 0)
    if n_trace != n_trace_expect:
        raise RuntimeError(
            f"{n_trace} trace edges labelled for {n_trace_expect} given.")

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

    # The surgery must CONSERVE the domain's topology, not assume it: a
    # box is a ball (Euler 1) but a spherical shell is S^2 x I (Euler 2).
    owned = np.asarray(_owned_stratum_counts(dm), dtype=np.int64)
    comm.Allreduce(MPI.IN_PLACE, owned, op=MPI.SUM)
    euler_before = int(owned[0] - owned[1] + owned[2] - owned[3])
    owned = np.asarray(_owned_stratum_counts(new), dtype=np.int64)
    comm.Allreduce(MPI.IN_PLACE, owned, op=MPI.SUM)
    nv_g, ne_g, nf_g, nc_g = (int(x) for x in owned)
    if nv_g - ne_g + nf_g - nc_g != euler_before:
        raise RuntimeError(
            f"the placement changed the global Euler number: "
            f"{euler_before} -> {nv_g - ne_g + nf_g - nc_g}")

    after = _interior_face_counts_3d(new)
    for key, before in held_counts.items():
        now = after.get(key, 0)
        if now < before or (now != before
                            and key != (label, int(label_value))):
            raise RuntimeError(
                f"placing {label!r} would leave the surface {key[0]!r} with "
                f"{now} interior faces instead of {before}.")

    _validity_and_orientation_gates(new, comm)

    min_vol = np.array([_owned_min_cell_volume(new)], dtype=float)
    comm.Allreduce(MPI.IN_PLACE, min_vol, op=MPI.MIN)

    info = {"n_placed": n_placed, "n_on_surface": 0,
            "n_removed": n_removed, "n_surface_facets": n_facets,
            "n_trace_edges": (len(chain) - 1 if chain is not None else 0),
            "min_volume": float(min_vol[0])}
    if setback > 0.0:
        info["surface_trace"] = surface_trace
    if verbose:
        uw.pprint(f"[place_sheet {label!r}] placed {info['n_placed']} "
                  f"vertices, removed {info['n_removed']}; "
                  f"{info['n_surface_facets']} sheet faces")
    info["n_gathered"] = int(moved)   # cells the gather moved (#670)
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

# ------------------------------------------- the domain boundary as a tool

def _domain_boundary_facets(dm):
    """The DOMAIN boundary as one small global complex, identical everywhere.

    A facet of the domain boundary has support 1 AND is unshared — a
    partition-seam facet also has local support 1, and clipping against a
    seam would eat the mesh differently at every rank count (the
    :func:`_true_wall_vertex_mask` distinction). Each boundary facet lives
    on exactly one rank, so the gathered set holds each facet once; the
    facets are stitched by exact coordinate identity, which is sound
    because a shared vertex's coordinates are copies of the same bytes on
    every rank. COLLECTIVE.

    Returns ``(verts, facets)``: coordinates ``(nv, dim)`` and vertex-index
    facets ``(nf, dim)`` — edges in 2-D, triangles in 3-D.
    """
    dim = dm.getDimension()
    vS, vE = dm.getDepthStratum(0)
    pStart, _pEnd = dm.getChart()
    X = _coords(dm)[: vE - vS]
    shared = _shared_point_flags(dm).astype(bool)
    corners = []
    for f in range(*dm.getHeightStratum(1)):
        if len(dm.getSupport(f)) == 1 and not shared[f - pStart]:
            vv = [int(q) - vS for q in dm.getTransitiveClosure(f)[0]
                  if vS <= int(q) < vE]
            corners.append(X[vv])
    local = (np.asarray(corners, dtype=float) if corners
             else np.zeros((0, dim, dim), dtype=float))
    pieces = [g for g in uw.mpi.comm.allgather(local) if len(g)]
    if not pieces:
        raise RuntimeError("the mesh has no domain boundary facet")
    stack = np.concatenate(pieces)
    verts, inverse = np.unique(stack.reshape(-1, dim), axis=0,
                               return_inverse=True)
    return verts, inverse.reshape(-1, dim).astype(np.int64)


def _domain_loops_2d(dm):
    """The 2-D domain boundary as closed coordinate loops. COLLECTIVE."""
    verts, edges = _domain_boundary_facets(dm)
    loops = _skin_loops([(int(a), int(b)) for a, b in edges],
                        what="the domain boundary")
    return [verts[loop] for loop in loops]


def _compress_collinear_loop(loop_xy):
    """The loop's vertices where the boundary actually TURNS.

    The OCC tool needs the boundary's geometry, not its segmentation: a box
    wall's collinear run collapses to its corners, so a box builds the same
    four-sided (six-faced, one dimension up) tool the analytic box gave,
    and the boolean does not imprint the wall's mesh spacing onto the
    clipped faces. The compressed polygon carries the identical point set,
    so the cut itself is unchanged.
    """
    P = np.asarray(loop_xy, dtype=float)
    e1 = P - np.roll(P, 1, axis=0)
    e2 = np.roll(P, -1, axis=0) - P
    if P.shape[1] == 2:
        cross = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    else:
        cross = np.linalg.norm(np.cross(e1, e2), axis=1)
    turn = cross > (1e-12 * np.linalg.norm(e1, axis=1)
                    * np.linalg.norm(e2, axis=1))
    if turn.sum() < 3:
        raise RuntimeError(
            "a domain boundary loop compresses to fewer than three corners")
    return P[turn]


def _occ_domain_2d(occ, loops):
    """The domain as ONE OCC plane surface built from its boundary loops.

    ``loops`` are the (compressed) boundary polygons of the mesh's own
    vertices; the largest-area loop is the exterior, the rest are holes —
    an annulus arrives natively. The tool is exactly the discrete domain,
    so a boolean against it lands on the mesh's own facets (measured:
    3.5e-17 from the chords, against the 8.6e-3 sagitta a cut on the smooth
    circle would have left). Returns the surface tag; the caller
    synchronizes.
    """
    def area(P):
        return 0.5 * float(P[:, 0] @ np.roll(P[:, 1], -1)
                           - P[:, 1] @ np.roll(P[:, 0], -1))

    order = sorted(range(len(loops)), key=lambda k: -abs(area(loops[k])))
    rings = []
    for k in order:
        P = loops[k]
        pts = [occ.addPoint(x, y, 0.0) for x, y in P]
        lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)])
                 for i in range(len(pts))]
        rings.append(occ.addCurveLoop(lines))
    return occ.addPlaneSurface(rings)


def _snap_to_boundary_2d(xy, loops, tol=1e-9):
    """Snap nodes within ``tol`` of the domain boundary ONTO it, exactly.

    OCC's clipped edges sit within rounding of the tool; the band logic and
    the sew need EXACT membership. A node near a boundary corner takes the
    corner; a node near a segment loses only its normal component — on an
    axis-aligned wall that is exactly the wall-plane snap the box clip
    used, the tangential coordinate untouched.
    """
    for P in loops:
        for v in P:
            xy[np.linalg.norm(xy - v, axis=1) < tol] = v
        n = len(P)
        for i in range(n):
            A, B = P[i], P[(i + 1) % n]
            e = B - A
            length = float(np.linalg.norm(e))
            nrm = np.array([-e[1], e[0]]) / length
            off = (xy - A) @ nrm
            u = ((xy - A) @ e) / (length * length)
            near = (np.abs(off) < tol) & (u > 0.0) & (u < 1.0)
            xy[near] -= off[near, None] * nrm
    return xy


def _coplanar_regions(verts, tris):
    """The boundary's COPLANAR REGIONS: adjacent coplanar facets merged.

    Two triangles sharing an edge join when their planes agree to rounding,
    so a box wall is one region while a faceted sphere keeps every triangle
    its own — the region is the 3-D analogue of the compressed straight
    segment of a 2-D boundary loop. Region ids are member triangle indices,
    arbitrary but deterministic. Returns ``(region, planes)``: the region id
    per triangle and ``{region: (anchor, unit_normal)}``.
    """
    tris = np.asarray(tris, dtype=np.int64)
    T = verts[tris]
    raw_n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    nn = np.linalg.norm(raw_n, axis=1)
    if (nn == 0.0).any():
        raise RuntimeError("a domain boundary triangle is degenerate")
    unit_n = raw_n / nn[:, None]

    parent = list(range(len(tris)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edge_first = {}
    for t, tri in enumerate(tris):
        opp = {frozenset((int(tri[0]), int(tri[1]))): int(tri[2]),
               frozenset((int(tri[1]), int(tri[2]))): int(tri[0]),
               frozenset((int(tri[2]), int(tri[0]))): int(tri[1])}
        for key, far in opp.items():
            if key not in edge_first:
                edge_first[key] = t
                continue
            s = edge_first[key]
            scale = max(nn[t], nn[s]) ** 0.5
            coplanar = (np.linalg.norm(np.cross(unit_n[t], unit_n[s]))
                        < 1e-12
                        and abs((verts[far] - T[s, 0]) @ unit_n[s])
                        < 1e-12 * scale)
            if coplanar:
                parent[find(t)] = find(s)

    region = np.array([find(t) for t in range(len(tris))], dtype=np.int64)
    planes = {int(r): (verts[int(tris[r][0])], unit_n[r])
              for r in np.unique(region)}
    return region, planes


def _occ_domain_3d(occ, verts, tris):
    """The domain as ONE OCC solid built from its boundary triangles.

    Adjacent coplanar triangles merge into single planar faces first — a
    box wall becomes one rectangle, so a box builds the same six-faced tool
    the analytic box gave, and the boolean does not imprint the wall's mesh
    spacing onto the clipped faces — and each merged face's rim compresses
    to its corners. The chain shared by two merged faces lies in both
    planes, hence on a straight line, so both faces compress it to the same
    endpoints; a vertex where three or more faces meet is always kept, or
    two faces whose planes cross the same line would disagree about it.
    Faces share OCC points and lines, so the shells they close into are
    topologically sewn. Returns ``(volume_tag, planes)`` with ``planes`` a
    list of ``(anchor, unit_normal)`` per merged face, for the snap; the
    caller synchronizes.
    """
    tris = np.asarray(tris, dtype=np.int64)
    T = verts[tris]
    region, planes_of = _coplanar_regions(verts, tris)

    def find(t):
        return int(region[t])

    regions = {}
    for t in range(len(tris)):
        regions.setdefault(find(t), []).append(t)
    regions_at = {}
    for t, tri in enumerate(tris):
        r = find(t)
        for v in tri:
            regions_at.setdefault(int(v), set()).add(r)

    point_tag, line_tag = {}, {}

    def point(v):
        if v not in point_tag:
            point_tag[v] = occ.addPoint(*verts[v])
        return point_tag[v]

    def line(a, b):
        key = (a, b) if a < b else (b, a)
        if key not in line_tag:
            line_tag[key] = occ.addLine(point(key[0]), point(key[1]))
        return line_tag[key]

    def compress_rim(loop):
        P = verts[np.asarray(loop)]
        e1 = P - np.roll(P, 1, axis=0)
        e2 = np.roll(P, -1, axis=0) - P
        cross = np.linalg.norm(np.cross(e1, e2), axis=1)
        turn = cross > (1e-12 * np.linalg.norm(e1, axis=1)
                        * np.linalg.norm(e2, axis=1))
        junction = np.array([len(regions_at[int(v)]) >= 3 for v in loop])
        keep = turn | junction
        if keep.sum() < 3:
            raise RuntimeError(
                "a domain face's rim compresses to fewer than three "
                "corners")
        return [int(v) for v, k in zip(loop, keep) if k]

    from collections import Counter

    face_of_region = {}
    planes = []
    for r, members in regions.items():
        rim = Counter()
        for t in members:
            a, b, c = (int(v) for v in tris[t])
            for e in (frozenset((a, b)), frozenset((b, c)),
                      frozenset((c, a))):
                rim[e] += 1
        rim_edges = [tuple(e) for e, k in rim.items() if k == 1]
        loops = _skin_loops(rim_edges, what="a domain face's rim")
        anchor, m = planes_of[r]
        planes.append((anchor, m))
        # In-plane coordinates for the outer-vs-hole ranking only.
        u = T[members[0], 1] - T[members[0], 0]
        u = u / np.linalg.norm(u)
        w = np.cross(m, u)

        def rim_area(loop):
            Q = verts[np.asarray(loop)] - anchor
            x, y = Q @ u, Q @ w
            return 0.5 * float(x @ np.roll(y, -1) - y @ np.roll(x, -1))

        loops = sorted((compress_rim(lp) for lp in loops),
                       key=lambda lp: -abs(rim_area(lp)))
        rings = []
        for lp in loops:
            rings.append(occ.addCurveLoop(
                [line(lp[i], lp[(i + 1) % len(lp)])
                 for i in range(len(lp))]))
        face_of_region[r] = occ.addPlaneSurface(rings)

    # Shells: connected components of the triangulation; the component
    # enclosing the greatest volume is the outer boundary, the others are
    # interior cavities (a spherical shell's inner surface).
    sparent = list(range(len(tris)))

    def sfind(a):
        while sparent[a] != a:
            sparent[a] = sparent[sparent[a]]
            a = sparent[a]
        return a

    edge_seen = {}
    for t, tri in enumerate(tris):
        a, b, c = (int(v) for v in tri)
        for e in (frozenset((a, b)), frozenset((b, c)), frozenset((c, a))):
            if e in edge_seen:
                sparent[sfind(t)] = sfind(edge_seen[e])
            else:
                edge_seen[e] = t
    shells, shell_tris = {}, {}
    for t in range(len(tris)):
        s = sfind(t)
        shells.setdefault(s, set()).add(find(t))
        shell_tris.setdefault(s, []).append(t)
    enclosed = {}
    for s, ts in shell_tris.items():
        Q = T[ts]
        enclosed[s] = abs(float(np.einsum(
            "ij,ij->i", np.cross(Q[:, 0], Q[:, 1]), Q[:, 2]).sum()) / 6.0)
    order = sorted(shells, key=lambda s: -enclosed[s])
    loops3 = [occ.addSurfaceLoop([face_of_region[r] for r in shells[s]])
              for s in order]
    return occ.addVolume(loops3), planes


def _snap_to_boundary_3d(xyz, verts, tris, planes, tol=1e-6):
    """Snap nodes within ``tol`` of the domain boundary ONTO it, exactly.

    Boundary corners first, then each merged face's plane, each a pure
    normal-component move — on an axis-aligned wall that is exactly the
    wall-plane snap the box clip used, the in-plane coordinates untouched.
    A node near two planes (a domain edge) is snapped by both passes, as
    the box path snapped both axes at a corner, so a band-outline node at
    a crease crossing lands ON the crease line.

    ``tol`` must sit above OCC's boolean placement noise, which reaches
    ~4e-7 on O(1) geometry against a many-faceted tool (measured: a
    crease-crossing node left 3.8e-7 off its crease, which the collar
    then meshed as a razor sliver) — and far below the layer mesh size,
    which keeps genuinely interior nodes out of reach. Candidates are
    masked by distance to the boundary FACETS first: the planes are
    infinite, and at this tolerance every point in space is near SOME
    plane of a many-faceted tool.
    """
    near_boundary = _sheet_distance(xyz, verts, tris) < tol
    idx = np.flatnonzero(near_boundary)
    sub = xyz[idx]
    for v in np.unique(tris):
        p = verts[int(v)]
        sub[np.linalg.norm(sub - p, axis=1) < tol] = p
    for anchor, m in planes:
        off = (sub - anchor) @ m
        near = np.abs(off) < tol
        sub[near] -= off[near, None] * m
    xyz[idx] = sub
    return xyz


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


def _grid_normals(grid):
    """Per-vertex unit normals of an ``(nu, nv, 3)`` sheet grid.

    Central differences along the two grid directions, crossed. One
    field, computed once: nested levels must SUBSAMPLE this (never
    recompute on the coarse grid — two independently computed normal
    fields differ at shared points and break vertex coincidence, the
    whole point of the ladder)."""
    G = np.asarray(grid, dtype=float)
    du = np.gradient(G, axis=0)
    dv = np.gradient(G, axis=1)
    N = np.cross(du, dv)
    n = np.linalg.norm(N, axis=2)
    if (n <= 0.0).any():
        raise ValueError("the sheet grid is degenerate (zero-area quads); "
                         "cannot derive normals.")
    return N / n[..., None]


def _prism_tets(prism, gids):
    """Split one prism into three tets, quad diagonals globally consistent.

    ``prism`` is six point indices — bottom triangle then top, ``i+3``
    above ``i`` — and ``gids`` their global ids. The subdivision is
    Dompierre et al.'s: normalise so the smallest global id sits at
    bottom position 0 (cyclic rotations; an upside-down flip is the
    orientation-preserving ``(3,5,4|0,2,1)``), then each quad face's
    diagonal passes through that face's smallest-id vertex — which is a
    face-local rule, so the two prisms sharing a quad face cut it the
    same way. Compatibility is verified globally by the face-pairing
    check in :func:`_ladder_assembly_3d`, not assumed here.
    """
    order = list(prism)
    g = list(gids)
    if min(g[3:]) < min(g[:3]):                 # smallest on top: flip
        order = [order[k] for k in (3, 5, 4, 0, 2, 1)]
        g = [g[k] for k in (3, 5, 4, 0, 2, 1)]
    r = int(np.argmin(g[:3]))                   # rotate smallest to slot 0
    rot = [(0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3), (2, 0, 1, 5, 3, 4)][r]
    v = [order[k] for k in rot]
    g = [g[k] for k in rot]
    if min(g[1], g[5]) < min(g[2], g[4]):
        return [(v[0], v[1], v[2], v[5]),
                (v[0], v[1], v[5], v[4]),
                (v[0], v[4], v[5], v[3])]
    return [(v[0], v[1], v[2], v[4]),
            (v[0], v[4], v[2], v[5]),
            (v[0], v[4], v[5], v[3])]


def _ladder_assembly_3d(grid, normals, width):
    """The extruded band: the sheet triangulation offset to prisms (#629).

    The 3-D transfinite ladder, built from the surface's own
    discretisation instead of a CAD remesh: the ``(nu, nv, 3)`` sheet
    grid is offset by ``-w/2, 0, +w/2`` along its per-vertex ``normals``
    (three vertex sheets — rails plus an exact mid-surface, the #595
    structure one dimension up), each grid quad becomes two triangles,
    each triangle a prism per layer, each prism three tets. Because the
    band inherits the sheet's vertices, a 2:1 sub-sampled grid produces
    a band whose every vertex is a vertex of the fine band — the placed
    level pairs NEST, which is what keeps the Galerkin chain native
    (measured: two independent CAD fills share nothing and fatten the
    chain 1.3–2x per level while making the patch band-wide).

    The mid-surface sheet is the split target: its faces are interior
    faces BY CONSTRUCTION, so a fault label needs only vertex-coordinate
    selection, no rim erosion geometry.

    Normals are the CALLER's: for nested levels they must be sampled
    from one parametrisation (subsample the fine grid's normals — two
    independently computed normal fields differ at shared points and
    break vertex coincidence). Raises when the extrusion inverts (sheet
    curvature too tight for the width) and when the prism subdivision
    leaves an incompatible interior face.
    """
    G = np.asarray(grid, dtype=float)
    N = np.asarray(normals, dtype=float)
    if G.ndim != 3 or G.shape[2] != 3 or min(G.shape[:2]) < 2:
        raise ValueError(
            f"the ladder sheet must be an (nu, nv, 3) grid with nu, nv >= 2; "
            f"got shape {G.shape}")
    if N.shape != G.shape:
        raise ValueError(
            f"normals must match the grid shape {G.shape}; got {N.shape}")
    nu, nv = G.shape[:2]
    N = N / np.linalg.norm(N, axis=2)[..., None]

    sheets = [G + s * (0.5 * width) * N for s in (-1.0, 0.0, 1.0)]
    pts = np.concatenate([s.reshape(-1, 3) for s in sheets], axis=0)
    nsheet = nu * nv

    def vid(sheet, i, j):
        return sheet * nsheet + i * nv + j

    # Grid quads -> two triangles, fixed diagonal (i,j)-(i+1,j+1). Winding
    # is set so the triangle is CCW seen from the +normal side, making
    # every prism right-handed and its canonical tets positively oriented
    # — an inverted (over-curved) extrusion then shows up as a NEGATIVE
    # volume rather than being silently "fixed" by reordering.
    tris = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a, b = (i, j), (i + 1, j)
            c, d = (i + 1, j + 1), (i, j + 1)
            e1 = G[b] - G[a]
            e2 = G[d] - G[a]
            flip = float(np.dot(np.cross(e1, e2), N[a])) < 0.0
            for tri in ((a, b, c), (a, c, d)):
                tris.append(tri[::-1] if flip else tri)

    tets = []
    for layer in (0, 1):
        for tri in tris:
            prism = ([vid(layer, i, j) for i, j in tri]
                     + [vid(layer + 1, i, j) for i, j in tri])
            tets.extend(_prism_tets(prism, prism))
    tets = np.asarray(tets, dtype=np.int64)

    P = pts[tets]
    vol6 = np.einsum(
        "ij,ij->i", np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]),
        P[:, 3] - P[:, 0])
    if (vol6 <= 0.0).any():
        raise ValueError(
            f"the ladder extrusion inverts {int((vol6 <= 0).sum())} of "
            f"{len(tets)} tets: the sheet's curvature is too tight for "
            f"width {width} (offset sheets cross), or the normals are "
            f"inconsistent with the grid orientation.")

    # Face-pairing check: the whole-mesh proof that every shared quad
    # face was cut the same way. A mismatched diagonal leaves its two
    # half-faces UNMATCHED (each used by one tet), so it shows up as
    # surplus boundary faces against the analytic skin count — which is
    # exactly the failure that would poison the skin extraction.
    from collections import Counter
    face_use = Counter()
    for t in tets:
        a, b, c, d = (int(q) for q in t)
        for f in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            face_use[tuple(sorted(f))] += 1
    n_boundary = sum(1 for k in face_use.values() if k == 1)
    expected = 4 * (nu - 1) * (nv - 1) + 8 * ((nu - 1) + (nv - 1))
    if n_boundary != expected or any(k > 2 for k in face_use.values()):
        raise RuntimeError(
            f"ladder prism subdivision internal: {n_boundary} boundary "
            f"faces against the analytic skin's {expected} (and "
            f"{sum(1 for k in face_use.values() if k > 2)} over-shared) — "
            f"incompatible quad diagonals.")
    return pts, tets


def _occ_assembly_3d(patches, width, size, domain=None, assembly="fuse",
                     embed=None):
    """Thicken each planar patch by ±width/2, resolve overlaps, mesh.

    ``assembly`` is :func:`place_thin_volume`'s: ``"fuse"`` returns the union
    as one solid, ``"fragment"`` keeps every overlap piece.

    ``domain = (verts, tris)`` — the mesh's boundary complex
    (:func:`_domain_boundary_facets`) — applies the specify-long contract:
    the thickened solids are INTERSECTED with the domain built as OCC
    geometry from those very facets, so patches may protrude — an assembly
    reaching a boundary leaves its clipped face exactly on the boundary's
    own facets (snapped onto their planes after meshing, defensively).

    ``embed`` is the 2-D network lesson one dimension up: a sequence of
    planar polygons (typically the patches themselves, un-thickened)
    FRAGMENTED INTO the fused solid before meshing, so each becomes a
    conforming interior surface of the band — the mid-surface a split
    walks, at any width. Requires ``domain=None`` for now (an embedded
    surface with an outcropping band would need the same clip — refused).

    Returns ``(points, tets, cad_volume, embedded)`` — the assembly mesh
    in its own numbering, the CAD volume of the (clipped) pieces against
    which the meshed volume is gated (planar-faced solids mesh to their
    exact volume), and per ``embed`` entry the ``(m, 3)`` triangles of
    its embedded faces in assembly numbering (``None`` without ``embed``).
    """
    if embed is not None and domain is not None:
        raise NotImplementedError(
            "embedded mid-surfaces with a domain clip (outcropping bands) "
            "are not built yet — the surfaces would need the same clip.")
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
            # The 2-D lesson one dimension up: the seams fragment leaves at
            # an overlap have no physics on them (properties reach the cells
            # from the Surface objects), and a shallow-angle overlap gives
            # them a spike to mesh. See :func:`_occ_assembly_2d`.
            if assembly == "fuse":
                occ.fuse([(3, solids[0])], [(3, t) for t in solids[1:]])
            else:
                occ.fragment([(3, solids[0])], [(3, t) for t in solids[1:]])
        planes = None
        if domain is not None:
            dom_verts, dom_tris = domain
            occ.synchronize()
            solids = [t for _d, t in gmsh.model.getEntities(3)]
            tool, planes = _occ_domain_3d(occ, dom_verts, dom_tris)
            occ.intersect([(3, t) for t in solids], [(3, tool)])
        per_embed = None
        if embed is not None:
            occ.synchronize()
            host = [t for _d, t in gmsh.model.getEntities(3)]
            mids = []
            for poly in embed:
                Q = np.asarray(poly, dtype=float)
                mpts = [occ.addPoint(*q) for q in Q]
                mlines = [occ.addLine(mpts[i], mpts[(i + 1) % len(Q)])
                          for i in range(len(Q))]
                mids.append(occ.addPlaneSurface([occ.addCurveLoop(mlines)]))
            _frag, frag_map = occ.fragment([(3, t) for t in host],
                                           [(2, t) for t in mids])
            # frag_map aligns with the input: host volumes first, then
            # each mid-surface's descendants
            per_embed = [[t for d, t in frag_map[len(host) + k] if d == 2]
                         for k in range(len(mids))]
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
        tets = np.vstack(tets)
        # The meshed-vs-CAD gate runs BEFORE the snap (the 2-D collapse
        # precedent): it tests gmsh's fidelity to the boolean, while the
        # snap is our own exact bookkeeping whose volume effect is gated
        # by the caller's domain-conservation check. The reference, OCC's
        # boolean mass, is itself only good to ~5e-7 relative on a
        # many-faceted tool, hence the tolerance.
        P = xyz[tets]
        v6 = np.einsum("ij,ij->i",
                       np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]),
                       P[:, 3] - P[:, 0])
        mesh_vol = float(np.abs(v6).sum() / 6.0)
        if abs(mesh_vol - cad_volume) > 1e-6 * cad_volume:
            raise RuntimeError(
                f"the assembly meshed to volume {mesh_vol:.12e} against "
                f"CAD {cad_volume:.12e}; the layer mesh does not fill its "
                "own solids.")
        if planes is not None:
            # OCC's clipped faces sit within rounding of the boundary; the
            # band logic and the cap's node sharing need EXACT membership,
            # so snap defensively.
            xyz = _snap_to_boundary_3d(xyz, dom_verts, dom_tris, planes)
        embedded = None
        if embed is not None:
            embedded = []
            for k, faces in enumerate(per_embed):
                tris = []
                for sf in faces:
                    et, _ei, en = gmsh.model.mesh.getElements(2, sf)
                    for ty, nodes in zip(et, en):
                        if ty == 2:
                            tris.append(np.array(
                                [renum[int(x)] for x in nodes],
                                dtype=np.int64).reshape(-1, 3))
                if not tris:
                    raise RuntimeError(
                        f"embedded surface {k} meshed to no faces")
                embedded.append(np.vstack(tris))
        return xyz, tets, float(cad_volume), embedded
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


def _split_skin_trace(skin_xyz, skin_facets, dom_verts, dom_facets,
                      tol=1e-9):
    """Split a skin into its interior part and the boundary TRACE — the
    facets of the assembly's clipped face lying ON the domain's own
    boundary facets (the zone's outcrop). Membership is metric but
    unambiguous: the post-clip snap puts a clipped vertex within rounding
    of the boundary complex, while an interior skin vertex is a layer
    mesh-size away. Every vertex AND the centroid must be on the boundary,
    so a facet spanning between two boundary contacts through the interior
    cannot pass.

    Dimension-free: edges against boundary edges in 2-D, triangles against
    boundary triangles in 3-D. Returns ``(interior_idx, trace_idx)``.
    """
    distance = (_segments_distance if skin_xyz.shape[1] == 2
                else _sheet_distance)
    d_vertex = distance(skin_xyz, dom_verts, dom_facets)
    corners = d_vertex[skin_facets].max(axis=1)
    d_centre = distance(skin_xyz[skin_facets].mean(axis=1),
                        dom_verts, dom_facets)
    on = (corners < tol) & (d_centre < tol)
    return np.flatnonzero(~on), np.flatnonzero(on)


def _collapse_boundary_imprints(asm_pts, asm_tris, loops, delta):
    """Merge a clipped-corner node into the domain vertex it almost hits.

    The boolean imprints the tool's corners into the clipped face. Where
    the assembly's own clipped corner lands within ``delta`` of an
    imprinted domain vertex, two forced nodes sit a hair apart on the same
    boundary segment and the layer meshes a sliver spanning them
    (measured: 0.18 degrees on an annulus radial outcrop, against 29.6 for
    its interior twin). Taking the domain vertex moves the band's end
    ALONG the boundary, so the domain's shape and area are untouched; the
    assembly's side edge tilts by under ``delta`` across its last cell —
    the end-snap contract, one level up. Runs after the meshed-vs-CAD
    gate: the collapse is our own exact bookkeeping, not the mesher's.
    """
    comp = [_compress_collinear_loop(L) for L in loops]
    seg_pts, seg_edges = [], []
    for L in comp:
        base = len(seg_pts)
        n = len(L)
        seg_edges += [(base + i, base + (i + 1) % n) for i in range(n)]
        seg_pts += list(L)
    on_bound = (_segments_distance(asm_pts, np.asarray(seg_pts), seg_edges)
                < 1e-9)
    remap = np.arange(len(asm_pts))
    for c in np.vstack(loops):
        hit = np.flatnonzero((asm_pts == c).all(axis=1))
        if not len(hit):
            continue
        j = int(hit[0])
        d = np.linalg.norm(asm_pts - c, axis=1)
        for m in np.flatnonzero((d > 0.0) & (d < delta) & on_bound):
            remap[int(m)] = j
    if (remap == np.arange(len(asm_pts))).all():
        return asm_pts, asm_tris
    tris = remap[asm_tris]
    degenerate = ((tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2])
                  | (tris[:, 2] == tris[:, 0]))
    tris = tris[~degenerate]
    used = np.unique(tris)
    compact = np.full(len(asm_pts), -1, dtype=np.int64)
    compact[used] = np.arange(len(used))
    return asm_pts[used], compact[tris]


def _collapse_boundary_imprints_3d(asm_pts, asm_tets, dom_verts, dom_tris,
                                   delta):
    """Route the band outline THROUGH any boundary vertex it grazes.

    The 2-D collapse one dimension up. The clip lands the band's outline
    where geometry puts it, and over a long band it passes arbitrarily
    close to some of the wall's own vertices; the collar then meshes a
    sliver between the vertex and the outline (measured: a 2.4 m gap on
    a 1000 km megathrust, refused by the 2-D fill as moved nodes). Where
    a kept-able boundary vertex lies within ``delta`` of the outline,
    the outline is made to pass THROUGH it: the nearest outline node
    moves onto the vertex when one is close enough, otherwise the
    outline edge is split at the vertex — every tetrahedron on that edge
    bisects, shape-safe because the new node sits within ``delta`` of
    the edge it splits. The vertex is then band-covered exactly, the
    frame rule deletes it, and the outline node re-provides its
    position; the band still tiles the same faceted surface, so the
    domain's shape is untouched. The band's side faces tilt by under
    ``delta`` across their last cell — the end-snap contract, one level
    up.
    """
    from collections import Counter

    skin_xyz, skin_local, node_ids = _assembly_skin(asm_pts, asm_tets)
    _interior, band_idx = _split_skin_trace(skin_xyz, skin_local,
                                            dom_verts, dom_tris)
    if not len(band_idx):
        return asm_pts, asm_tets
    cnt = Counter()
    for t in skin_local[band_idx]:
        a, b, c = sorted(int(v) for v in t)
        for e in ((a, b), (a, c), (b, c)):
            cnt[e] += 1
    outline = [(int(node_ids[a]), int(node_ids[b]))
               for (a, b), k in cnt.items() if k == 1]

    pts = list(asm_pts)
    tets = [list(t) for t in asm_tets]
    onode_xyz = asm_pts[np.unique(np.asarray(outline).ravel())]
    lo = onode_xyz.min(axis=0) - 2.0 * delta
    hi = onode_xyz.max(axis=0) + 2.0 * delta
    candidates = [int(v) for v in np.unique(dom_tris)
                  if (lo <= dom_verts[v]).all()
                  and (dom_verts[v] <= hi).all()]

    def v6_of(row, at=None, coord=None):
        P = [coord if at is not None and x == at else np.asarray(pts[x])
             for x in row]
        return float(np.cross(P[1] - P[0], P[2] - P[0]) @ (P[3] - P[0]))

    for v in candidates:
        V = dom_verts[v]
        best_d, best_k = np.inf, -1
        for k, (a, b) in enumerate(outline):
            A, B = np.asarray(pts[a]), np.asarray(pts[b])
            e = B - A
            u = float(np.clip((V - A) @ e / float(e @ e), 0.0, 1.0))
            d = float(np.linalg.norm(V - (A + u * e)))
            if d < best_d:
                best_d, best_k = d, k
        if best_d >= delta or best_d < 1e-12:
            continue
        a, b = outline[best_k]
        da = float(np.linalg.norm(np.asarray(pts[a]) - V))
        db = float(np.linalg.norm(np.asarray(pts[b]) - V))
        # A move or split may not flatten any incident tetrahedron — two
        # grazes competing for the same neighbourhood would otherwise
        # stack nodes and degenerate cells. A skipped vertex keeps its
        # sliver, which is the status quo, not a defect.
        if min(da, db) < 1.5 * delta:
            n = a if da <= db else b
            star = [row for row in tets if n in row]
            if all(abs(v6_of(row, at=n, coord=V))
                   > 0.2 * abs(v6_of(row)) for row in star):
                pts[n] = V.copy()
            continue
        edge_rows = [row for row in tets if a in row and b in row]
        m = len(pts)
        ok = all(min(abs(v6_of([m if x == a else x for x in row],
                              at=m, coord=V)),
                     abs(v6_of([m if x == b else x for x in row],
                              at=m, coord=V)))
                 > 0.05 * abs(v6_of(row)) for row in edge_rows)
        if not ok:
            continue
        pts.append(V.copy())
        fresh = []
        for row in edge_rows:
            two = [m if x == b else x for x in row]
            row[:] = [m if x == a else x for x in row]
            fresh.append(two)
        tets.extend(fresh)
        outline[best_k] = (a, m)
        outline.append((m, b))
    return np.asarray(pts, dtype=float), np.asarray(tets, dtype=np.int64)


def _refuse_multiple_bands(band_facets):
    """One contiguous band only: a single carve-and-splice is built.

    Two ribbons out of two walls, or one arch out of the same wall twice,
    both arrive here as more than one connected component of band facets.
    """
    parent = {}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for facet in band_facets:
        for v in facet:
            parent.setdefault(int(v), int(v))
        roots = {find(int(v)) for v in facet}
        anchor = roots.pop()
        for r in roots:
            parent[r] = anchor
    if len({find(v) for v in parent}) > 1:
        raise NotImplementedError(
            "the zone meets the domain boundary in more than one band; a "
            "multiply-outcropping zone is not built.")


def _outcrop_frame_2d(comp_loops, asm_pts, band_pairs, X, on_wall):
    """Which mesh boundary vertices an outcrop may DELETE, and which ring
    vertices count as boundary contact AWAY from the outcrop.

    A boundary vertex may go when the band re-provides its geometry (the
    vertex lies ON the band — the assembly holds a node at its exact
    position), or when it lies on a band-touched straight segment of the
    compressed boundary, where the splice's cap segments stay collinear
    with the segment and the domain shape is preserved. A compressed
    CORNER off the band is the domain's shape itself and is never deleted.
    On a box this reproduces the wall-plane rule exactly: the touched
    segment is the whole wall, and the box corners are the protected ones.

    Returns ``(deletable, near_touched)`` — masks over the mesh vertices.
    """
    band_nodes = sorted({int(v) for pair in band_pairs for v in pair})
    B = asm_pts[band_nodes]
    seg_pts, seg_edges, corners = [], [], []
    for L in comp_loops:
        corners.append(L)
        n = len(L)
        for i in range(n):
            A, C = L[i], L[(i + 1) % n]
            e = C - A
            u = np.clip(((B - A) @ e) / float(e @ e), 0.0, 1.0)
            d = np.linalg.norm(B - (A + u[:, None] * e), axis=1)
            if (d < 1e-9).any():
                seg_edges.append((len(seg_pts), len(seg_pts) + 1))
                seg_pts += [A, C]
    if not seg_edges:
        raise RuntimeError(
            "the outcrop band lies on no boundary segment; the clip and "
            "the boundary disagree")
    d_seg = _segments_distance(X, np.asarray(seg_pts), seg_edges)
    d_band = _segments_distance(X, asm_pts,
                                [tuple(p) for p in band_pairs])
    is_corner = np.zeros(len(X), dtype=bool)
    for c in np.vstack(corners):
        is_corner |= np.linalg.norm(X - c, axis=1) < 1e-9
    near_touched = d_seg < 1e-9
    deletable = on_wall & near_touched & (~is_corner | (d_band < 1e-9))
    return deletable, near_touched


# Adjacent boundary facets whose normals differ by less than this are the
# same SMOOTH WALL: the faceting of a curved boundary (angle -> 0 under
# refinement) groups into one wall, while a box-like corner (90 degrees)
# separates two. The outcrop bowl may open anywhere on a band-touched wall.
_WALL_DIHEDRAL_COS = np.cos(np.deg2rad(45.0))


def _boundary_walls(dom_verts, dom_tris, region):
    """Group coplanar regions into SMOOTH WALLS across low-angle creases.

    Returns ``{region: wall}`` with wall ids arbitrary but deterministic.
    """
    tris = np.asarray(dom_tris, dtype=np.int64)
    T = dom_verts[tris]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n = n / np.linalg.norm(n, axis=1)[:, None]

    parent = {int(r): int(r) for r in np.unique(region)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edge_first = {}
    for t, tri in enumerate(tris):
        a, b, c = sorted(int(v) for v in tri)
        for e in ((a, b), (a, c), (b, c)):
            if e not in edge_first:
                edge_first[e] = t
                continue
            s = edge_first[e]
            # The gathered complex carries no consistent orientation, so
            # the dihedral test must not depend on the facet normals'
            # signs; |cos| also merges a near-zero fold, which a domain
            # boundary does not have.
            if abs(float(n[t] @ n[s])) > _WALL_DIHEDRAL_COS:
                parent[find(int(region[t]))] = find(int(region[s]))
    return {r: find(r) for r in parent}


def _outcrop_frame_3d(X, on_wall, dom_verts, dom_tris, region,
                      skin_xyz, band_facets, tol=1e-9):
    """Which mesh boundary vertices a 3-D outcrop may DELETE, and which
    count as boundary contact NEAR the outcrop.

    The outcrop's footprint arrives as ``band_facets`` over ``skin_xyz``:
    a BAND of skin triangles (a fat zone) or a trace CHAIN of edges (a
    sheet, whose outcrop has no area) — the frame never asks whether the
    object is fat or thin, only where its footprint lies.

    The 2-D frame rule one dimension up, with the two notions the third
    dimension separates. NEAR — where the cavity may open into cap faces —
    is by SMOOTH WALL (:func:`_boundary_walls`): on a faceted sphere the
    bowl legitimately spills onto facets beside the band's own, while a
    box's other walls stay refused across their sharp creases. DELETABLE
    is by SHAPE: a vertex may go when the band re-provides its geometry
    (it lies ON the band — :func:`_collapse_boundary_imprints_3d` routes
    the outline THROUGH any vertex it merely grazes, so the exact test
    suffices), when it is interior to a single band-touched coplanar
    region (the collar re-mesh stays in the region's own plane), or when
    it is interior to a STRAIGHT crease between two touched regions — a
    straight crease's interior vertices carry no shape, and the overlay
    rebuilds the line through the survivors. A vertex where three or
    more regions meet (a box corner, every vertex of a faceted sphere)
    is the domain's shape itself — the 2-D compressed corner — and is
    never deleted off the band: its uncrossed facets' crease chains
    would dangle without it.

    Membership is exact: the gathered complex's vertex coordinates are the
    mesh's own bytes, so a mesh vertex's regions are read off the facets
    incident to its coordinate, not off a distance.

    Returns ``(deletable, near_touched, touched)`` — masks over the mesh
    vertices, and the set of touched region ids.
    """
    band_facets = np.asarray(band_facets, dtype=np.int64)
    band_cen = skin_xyz[band_facets].mean(axis=1)
    d_cen, at = _nearest_facet(band_cen, dom_verts, dom_tris)
    if (d_cen > tol).any():
        raise RuntimeError(
            "the outcrop band lies off the domain boundary; the clip and "
            "the boundary disagree")
    touched = {int(region[k]) for k in at}
    wall_of = _boundary_walls(dom_verts, dom_tris, region)
    touched_walls = {wall_of[r] for r in touched}

    row_of = {tuple(v): i for i, v in enumerate(dom_verts)}
    regions_at = {}
    crease_dirs = {}
    edge_first = {}
    for t, tri in enumerate(dom_tris):
        for v in tri:
            regions_at.setdefault(int(v), set()).add(int(region[t]))
        a, b, c = sorted(int(v) for v in tri)
        for e in ((a, b), (a, c), (b, c)):
            if e not in edge_first:
                edge_first[e] = t
                continue
            if int(region[t]) != int(region[edge_first[e]]):
                d = dom_verts[e[1]] - dom_verts[e[0]]
                d = d / np.linalg.norm(d)
                for v in e:
                    crease_dirs.setdefault(int(v), []).append(d)

    d_band = (_segments_distance(X, skin_xyz,
                                 [tuple(e) for e in band_facets])
              if band_facets.shape[1] == 2
              else _sheet_distance(X, skin_xyz, band_facets))
    deletable = np.zeros(len(X), dtype=bool)
    near_touched = np.zeros(len(X), dtype=bool)
    for i in np.flatnonzero(on_wall):
        dv = row_of.get(tuple(X[i]))
        if dv is None:
            raise RuntimeError(
                "a domain-wall vertex is missing from the gathered "
                "boundary complex; the wall mask and the complex disagree")
        mine = regions_at[dv]
        if not any(wall_of[r] in touched_walls for r in mine):
            continue
        near_touched[i] = True
        if d_band[i] < tol:
            deletable[i] = True
        elif len(mine) == 1:
            deletable[i] = mine <= touched
        elif len(mine) == 2 and mine <= touched:
            dirs = crease_dirs.get(dv, [])
            deletable[i] = (len(dirs) == 2 and
                            float(np.linalg.norm(
                                np.cross(dirs[0], dirs[1]))) < 1e-9)
    return deletable, near_touched, touched


def _outcrop_collar_3d(X, alive, cap_faces, cap_region, planes_of,
                       skin_xyz, band_tris, band_tri_region, band_outline,
                       chain=None, tol=1e-9):
    """Re-triangulate the outcrop bowl's wall collar, region by region.

    The collar is the bowl's wall footprint minus the band — in the box
    case a flat annulus between the cavity's wall rim and the band's
    outline. On a general boundary the footprint spans several COPLANAR
    REGIONS, so each region's piece is meshed flat in its own plane — the
    cap stays exactly on the faceted surface, which is what conserves the
    domain volume — and the pieces conform along the CREASES between
    regions.

    A SHEET's outcrop has no area: instead of a band it passes ``chain`` —
    the ordered trace polyline as rows of ``skin_xyz`` (``band_tris``
    empty, ``band_outline`` None) — and the collar covers the WHOLE bowl
    with the chain EMBEDDED in the pieces rather than a hole cut out of
    them. A chain node at a crease crossing lands ON the crease and enters
    the 1-D overlay as an 'a' node exactly as a band's outline node does;
    the chain's runs between crossings are embedded per region, free ends
    (a trace ending mid-wall) as gmsh's ordinary free-end embed. A trace
    edge lying ALONG a crease, or off the bowl entirely, is refused.

    Along a crease the two sides segment the line differently: the bowl's
    nodes are mesh vertices, the band's crossing nodes are assembly nodes,
    so edge identity cannot match them. The 1-D overlay merges both node
    sets by arc coordinate along the (straight) crease and keeps each
    elementary sub-segment for the side(s) whose bowl piece covers it and
    the band does not. Deleted crease vertices (``alive`` False — the
    frame rule lets a straight crease's interior go) contribute geometry
    but not nodes: the chain is rebuilt through the survivors. A
    sub-segment covered from ONE side only must be a whole mesh edge,
    because the face matching it in the fill's closed surface is a
    cavity-shell face carrying the mesh segmentation — anything else there
    is a refusal, not a crack.

    ``cap_faces`` are the bowl's wall triangles as rows of ``X`` with
    ``alive`` the surviving-vertex mask, ``cap_region`` their coplanar
    region ids, ``planes_of`` the region planes; ``band_tris`` are the
    band's skin triangles (rows of ``skin_xyz``) with ``band_tri_region``
    their region ids and ``band_outline`` the band's closed outline loop
    of skin nodes.

    Returns ``(mesh_nodes, skin_nodes, extra_xyz, tris)`` — the cap in the
    fill's three node namespaces (kept mesh vertices, assembly skin nodes,
    new in-plane points), ``tris`` indexing their concatenation in that
    order, exactly as :func:`_gmsh_fill_annulus_3d` reads its payload.
    """
    cap_faces = np.asarray(cap_faces, dtype=np.int64)
    cap_region = np.asarray(cap_region, dtype=np.int64)
    band_tris = np.asarray(band_tris, dtype=np.int64).reshape(-1, 3)
    if (band_outline is None) == (chain is None):
        raise RuntimeError(
            "the outcrop collar takes a band outline (a zone) or a trace "
            "chain (a sheet), exactly one")
    chain = [int(v) for v in chain] if chain is not None else None

    # The bowl's wall edges. Within one region an edge seen once is the
    # bowl rim (matched by a cavity-shell face, so it stays a mesh edge
    # verbatim); an edge shared by two regions' faces is a crease edge and
    # goes to the overlay; an edge seen twice within one region is
    # interior to that region's piece.
    edge_faces = {}
    for k, tri in enumerate(cap_faces):
        a, b, c = sorted(int(v) for v in tri)
        for e in ((a, b), (a, c), (b, c)):
            edge_faces.setdefault(e, []).append(k)

    soup = {}
    crease_edges = {}
    for e, ks in edge_faces.items():
        if len(ks) > 2:
            raise RuntimeError(
                "an outcrop bowl wall edge belongs to more than two wall "
                "faces; the boundary complex is not manifold")
        rs = sorted({int(cap_region[k]) for k in ks})
        if len(ks) == 1:
            soup.setdefault(rs[0], []).append((('m', e[0]), ('m', e[1])))
        elif len(rs) == 2:
            crease_edges.setdefault(tuple(rs), []).append(e)

    # Region soups of the coverage tests, once.
    region_faces = {int(r): cap_faces[cap_region == r]
                    for r in np.unique(cap_region)}

    def covered(r, P):
        return bool(_sheet_distance(P[None, :], X, region_faces[r])[0]
                    < tol)

    def in_band(P):
        if not len(band_tris):
            return False
        return bool(_sheet_distance(P[None, :], skin_xyz, band_tris)[0]
                    < tol)

    outline_edges = []
    if band_outline is not None:
        n_out = len(band_outline)
        outline_edges = [(int(band_outline[i]),
                          int(band_outline[(i + 1) % n_out]))
                         for i in range(n_out)]
    # The skin nodes that may sit on a crease: a band's outline vertices,
    # or every node of a sheet's trace chain.
    overlay_candidates = ({v for e in outline_edges for v in e}
                          if band_outline is not None else set(chain))

    band_edge_owner = {}
    for k, tri in enumerate(band_tris):
        a, b, c = sorted(int(v) for v in tri)
        for e in ((a, b), (a, c), (b, c)):
            band_edge_owner.setdefault(e, []).append(k)

    # ------------------------------------------------- the crease overlay
    on_crease_edge = {}
    for pair, edges in crease_edges.items():
        parent = {}

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        for a, b in edges:
            parent.setdefault(a, a)
            parent.setdefault(b, b)
            parent[find(a)] = find(b)
        crease_chains = {}
        for a, b in edges:
            crease_chains.setdefault(find(a), []).append((a, b))

        # Not `chain` — that name is the sheet-trace parameter one scope up.
        for crease_chain in crease_chains.values():
            chain_all = sorted({v for e in crease_chain for v in e})
            nodes_m = [v for v in chain_all if alive[v]]
            origin = X[chain_all[0]]
            far = max(chain_all, key=lambda v: float(
                np.linalg.norm(X[v] - origin)))
            u = X[far] - origin
            u = u / np.linalg.norm(u)

            def arc(P):
                return float((P - origin) @ u)

            def off_line(P):
                return float(np.linalg.norm((P - origin)
                                            - arc(P) * u))

            span = [min(arc(X[v]) for v in chain_all) - tol,
                    max(arc(X[v]) for v in chain_all) + tol]
            resolved = [('m', v, arc(X[v])) for v in nodes_m]
            for s in overlay_candidates:
                P = skin_xyz[s]
                if off_line(P) < tol and span[0] < arc(P) < span[1]:
                    resolved.append(('a', int(s), arc(P)))
            for a, b in outline_edges:
                mid = 0.5 * (skin_xyz[a] + skin_xyz[b])
                if (off_line(skin_xyz[a]) < tol
                        and off_line(skin_xyz[b]) < tol
                        and span[0] < arc(mid) < span[1]):
                    # The outline runs ALONG this crease for one edge: the
                    # band reaches the crease there, so the edge bounds
                    # the collar piece on the OTHER side, not the owning
                    # band triangle's own region.
                    key = (a, b) if a < b else (b, a)
                    on_crease_edge[key] = pair
            for kind, i, s in resolved:
                if kind != 'a':
                    continue
                if any(k2 == 'm' and abs(s2 - s) < tol
                       for k2, _i2, s2 in resolved):
                    raise RuntimeError(
                        "a band outline node lands within rounding of a "
                        "kept mesh vertex on a domain crease; the collapse "
                        "of boundary imprints is not built in 3-D — move "
                        "the patch or change the mesh spacing.")
            chain_set = {(a, b) if a < b else (b, a)
                         for a, b in crease_chain}
            resolved.sort(key=lambda n: n[2])
            for (k0, i0, s0), (k1, i1, s1) in zip(resolved, resolved[1:]):
                mid = origin + (0.5 * (s0 + s1)) * u
                if in_band(mid):
                    continue
                sides = [r for r in pair if covered(r, mid)]
                if not sides:
                    continue
                if len(sides) == 1 and (
                        'a' in (k0, k1)
                        or ((i0, i1) if i0 < i1 else (i1, i0))
                        not in chain_set):
                    raise RuntimeError(
                        "the outcrop band splits a wall edge that the "
                        "cavity shell keeps whole; raise `clearance` so "
                        "the carve covers the crease on both sides")
                for r in sides:
                    soup.setdefault(r, []).append(((k0, i0), (k1, i1)))

    # The band outline bounds the collar from inside; each outline edge
    # belongs to exactly one band triangle, whose region takes it.
    for a, b in outline_edges:
        e = (a, b) if a < b else (b, a)
        owners = band_edge_owner.get(e, [])
        if len(owners) != 1:
            raise RuntimeError(
                "a band outline edge does not bound exactly one band "
                "facet; the outline and the band disagree")
        r = int(band_tri_region[owners[0]])
        if e in on_crease_edge:
            pair = on_crease_edge[e]
            if r not in pair:
                raise RuntimeError(
                    "a band outline edge lies on a crease its own band "
                    "facet does not touch; the trace and the boundary "
                    "disagree")
            other = pair[0] if pair[1] == r else pair[1]
            if other not in region_faces:
                raise NotImplementedError(
                    "the outcrop band runs ALONG a domain crease with no "
                    "bowl beyond it; widen `clearance` so the cavity "
                    "covers both sides.")
            r = other
        soup.setdefault(r, []).append((('a', a), ('a', b)))

    # A sheet's trace, split into per-region RUNS at its crease crossings.
    # Each trace edge belongs to the one region covering its midpoint: two
    # regions is an edge ALONG a crease (needs an overlay the trace cannot
    # provide), none is a trace off the bowl — both are refusals with the
    # reason, not mis-assignments.
    runs = {}
    if chain is not None:
        prev_r = None
        for a, b in zip(chain[:-1], chain[1:]):
            mid = 0.5 * (skin_xyz[a] + skin_xyz[b])
            rs = [r for r in region_faces if covered(r, mid)]
            if len(rs) > 1:
                raise NotImplementedError(
                    "the sheet's trace runs ALONG a domain crease; "
                    "embedding a trace edge into a crease is not built — "
                    "move the sheet or change the mesh spacing.")
            if not rs:
                raise RuntimeError(
                    "a trace edge lies on no collar region; the trace and "
                    "the bowl disagree")
            r = rs[0]
            if r == prev_r:
                runs[r][-1].append(b)
            else:
                runs.setdefault(r, []).append([a, b])
            prev_r = r

    # ------------------------------------- mesh each region's piece flat
    mesh_nodes, skin_nodes = [], []
    m_row, a_row = {}, {}

    def namespace(kind, i):
        if kind == 'm':
            if i not in m_row:
                m_row[i] = len(mesh_nodes)
                mesh_nodes.append(int(i))
            return ('m', m_row[i])
        if i not in a_row:
            a_row[i] = len(skin_nodes)
            skin_nodes.append(int(i))
        return ('a', a_row[i])

    extra_xyz = []
    tris_out = []
    for r, segments in sorted(soup.items()):
        local_of, local_nodes = {}, []
        for e in segments:
            for kind, i in e:
                if (kind, i) not in local_of:
                    local_of[(kind, i)] = len(local_nodes)
                    local_nodes.append((kind, i))
        # The region's embedded trace runs: their crossing ends are already
        # ring nodes (the overlay put them there); interior run nodes join
        # the table as new 'a' rows.
        r_runs = runs.get(int(r), [])
        for run in r_runs:
            for v in run:
                if ('a', v) not in local_of:
                    local_of[('a', v)] = len(local_nodes)
                    local_nodes.append(('a', v))
        coords = np.array([X[i] if kind == 'm' else skin_xyz[i]
                           for kind, i in local_nodes])
        m_coords = np.array([X[i] for kind, i in local_nodes
                             if kind == 'm'])
        if any(kind == 'm' and not alive[i] for kind, i in local_nodes):
            raise RuntimeError(
                "an outcrop collar node was deleted by the carve; the "
                "frame rule and the collar disagree")
        for kind, i in local_nodes:
            if kind != 'a' or not len(m_coords):
                continue
            if np.linalg.norm(m_coords - skin_xyz[i], axis=1).min() < tol:
                raise RuntimeError(
                    "a band outline node lands within rounding of a kept "
                    "mesh vertex; the collapse of boundary imprints is "
                    "not built in 3-D — move the patch or change the mesh "
                    "spacing.")

        anchor, nrm = planes_of[int(r)]
        f0 = region_faces[int(r)][0]
        e0 = X[f0[1]] - X[f0[0]]
        e0 = e0 / np.linalg.norm(e0)
        w0 = np.cross(nrm, e0)
        P2 = np.column_stack([(coords - anchor) @ e0,
                              (coords - anchor) @ w0])

        loops = _skin_loops(
            [(local_of[e[0]], local_of[e[1]]) for e in segments],
            what="an outcrop collar piece's boundary")

        def signed_area(loop):
            Q = P2[np.asarray(loop)]
            return 0.5 * float(Q[:, 0] @ np.roll(Q[:, 1], -1)
                               - Q[:, 1] @ np.roll(Q[:, 0], -1))

        containers = {k: [j for j, other in enumerate(loops)
                          if j != k and _inside_polygon(
                              P2[np.asarray(other)], P2[loop[0]])]
                      for k, loop in enumerate(loops)}
        run_local = [[local_of[('a', v)] for v in run] for run in r_runs]
        run_assigned = [False] * len(run_local)
        for k, loop in enumerate(loops):
            if len(containers[k]) % 2:
                continue                      # a hole; its outer takes it
            holes = [j for j in range(len(loops))
                     if containers[j]
                     and max(containers[j],
                             key=lambda c: len(containers[c])) == k
                     and len(containers[j]) == len(containers[k]) + 1]
            ring = loop if signed_area(loop) > 0.0 else loop[::-1]
            # Each trace run goes to the outer loop containing it, tested
            # at its first edge's midpoint (the run's ends may lie ON the
            # ring at crease crossings; the midpoint is strictly inside).
            mine = []
            for q, run in enumerate(run_local):
                probe = 0.5 * (P2[run[0]] + P2[run[1]])
                if (_inside_polygon(P2[np.asarray(ring)], probe)
                        and not any(_inside_polygon(
                            P2[np.asarray(loops[j])], probe)
                            for j in holes)):
                    mine.append(run)
                    run_assigned[q] = True
            try:
                fill_tris, extra2 = _gmsh_fill_2d(
                    P2, list(ring), mine or None,
                    holes=[list(loops[j]) for j in holes])
            except RuntimeError as exc:
                # Name the pinch: the thinnest node-to-segment gap in the
                # polygon, with the node KINDS ('m' = mesh vertex, 'a' =
                # assembly outline node), says which near-coincidence to
                # chase.
                ids = list(ring) + [w for j in holes for w in loops[j]]
                gap, gi, gj0, gj1 = np.inf, -1, -1, -1
                for i2 in ids:
                    for j2 in range(len(ids)):
                        j3 = (j2 + 1) % len(ids)
                        if i2 in (ids[j2], ids[j3]):
                            continue
                        A, B = P2[ids[j2]], P2[ids[j3]]
                        e2 = B - A
                        u2 = float(np.clip((P2[i2] - A) @ e2
                                           / float(e2 @ e2), 0.0, 1.0))
                        d2 = float(np.linalg.norm(P2[i2]
                                                  - (A + u2 * e2)))
                        if d2 < gap:
                            gap, gi, gj0, gj1 = d2, i2, ids[j2], ids[j3]
                raise RuntimeError(
                    f"an outcrop collar piece failed to mesh ({exc}); "
                    f"thinnest feature {gap:.3e} between node "
                    f"{local_nodes[gi]} and segment "
                    f"{local_nodes[gj0]}-{local_nodes[gj1]}")
            base = len(local_nodes)
            extra_row = {}
            for t in fill_tris:
                row = []
                for v in t:
                    if v < base:
                        row.append(namespace(*local_nodes[int(v)]))
                    else:
                        j = int(v) - base
                        if j not in extra_row:
                            x2 = extra2[j]
                            extra_xyz.append(anchor + x2[0] * e0
                                             + x2[1] * w0)
                            extra_row[j] = len(extra_xyz) - 1
                        row.append(('x', extra_row[j]))
                tris_out.append(row)
        if not all(run_assigned):
            raise RuntimeError(
                "a trace run lies in no piece of its collar region; the "
                "trace and the bowl disagree")

    n_m, n_a = len(mesh_nodes), len(skin_nodes)

    def flat(kind, i):
        if kind == 'm':
            return i
        if kind == 'a':
            return n_m + i
        return n_m + n_a + i

    tris = np.array([[flat(*v) for v in row] for row in tris_out],
                    dtype=np.int64)
    extra = (np.asarray(extra_xyz, dtype=float) if extra_xyz
             else np.zeros((0, 3), dtype=float))
    return mesh_nodes, skin_nodes, extra, tris


def _single_loop(edges, what):
    """Order undirected edges into ONE closed loop of vertices, or refuse."""
    adj = {}
    for a, b in edges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    if any(len(v) != 2 for v in adj.values()):
        raise RuntimeError(f"the {what} is not a single closed loop")
    start = min(adj)
    loop, prev, cur = [start], None, start
    while True:
        a, b = adj[cur]
        nxt = b if a == prev else a
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
    if len(loop) != len(adj):
        raise RuntimeError(f"the {what} is not a single closed loop")
    return loop


def _carve_around_volume_3d(dm, X, cells, skin_pts, skin_tris, reach_vertex,
                            reach_cell, held_cells, on_wall, shared_chart,
                            seed_drop=None, open_deletable=None,
                            open_near=None):
    """Victims, dropped tets and the closed shell around a FAT object.

    Differs from the sheet's carve in two measured ways. The reach is a
    LENGTH per vertex/cell (``max(clearance*h, 0.6*width)``), not a bare
    multiple of h — it must cover the layer's own half-width however sub-h
    the layer is. And the union of victim stars around a volume can PINCH —
    a shell edge whose surrounding cells are part-dropped in two wedges — so
    the drop set is GROWN at every non-manifold shell edge until the shell
    closes; dropping more cells only enlarges the fill (thin_volume_spike:
    converges in a few rounds).

    An OUTCROP passes its frame rule as two vertex masks
    (:func:`_outcrop_frame_3d`): ``open_deletable`` — wall vertices the
    carve may take — and ``open_near`` — wall vertices near the outcrop,
    where the cavity may open into cap faces.
    """
    d_skin = _sheet_distance_within(X, skin_pts, skin_tris, reach_vertex)

    held_vertex = np.zeros(len(X), dtype=bool)
    if held_cells:
        for c in held_cells:
            held_vertex[cells[c]] = True
    on_open = (open_deletable if open_deletable is not None
               else np.zeros(len(X), dtype=bool))
    victim = ((d_skin < reach_vertex)
              & (~on_wall | on_open) & ~held_vertex)

    drop = victim[cells].any(axis=1)
    # A background cell can straddle the layer's rim with every corner
    # outside the reach; its centroid cannot be far from the skin.
    cen_d = _sheet_distance_within(X[cells].mean(axis=1), skin_pts,
                                   skin_tris, reach_cell)
    drop |= cen_d < reach_cell
    # A removal seeds the drop with the object's OWN cells — a fat zone's
    # interior can be further from its skin than any reach computes.
    if seed_drop is not None:
        drop |= seed_drop
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

    shell, cap_faces, drop = _closed_shell_3d(dm, X, cells, drop, victim,
                                              held_cells, shared_chart,
                                              "thin volume",
                                              open_vertex=open_near)

    # The carve can swallow the whole star of a vertex that is NOT itself
    # a victim. An unreferenced survivor would ride through the rebuild as
    # an isolated point (global Euler 2, not 1 — the class CI caught), so
    # an interior one is promoted to a victim. A PROTECTED wall vertex
    # near the outcrop needs no kept cell at all: it is a COLLAR node —
    # the cap is re-triangulated through the surviving wall vertices,
    # which is what preserves a faceted boundary's shape, and the gap
    # fill's tets reference it. On a curved wall this is the common case:
    # every facet vertex near the band is protected, and each cell of its
    # small star holds some interior victim.
    referenced = np.zeros(len(X), dtype=bool)
    if (~drop).any():
        referenced[cells[~drop].ravel()] = True
    orphan = ~referenced & ~victim
    if open_near is not None:
        collar_kept = orphan & on_wall & ~on_open & open_near
        orphan &= ~collar_kept
    if (orphan & on_wall & ~on_open).any():
        raise RuntimeError(
            "the cavity would strand a domain-wall vertex; the volume must "
            "be interior, with clearance to spare")
    victim |= orphan
    return np.flatnonzero(victim), np.flatnonzero(drop), shell, cap_faces


def _gmsh_fill_annulus_3d(shell_xyz, shell_tris, skin_xyz, skin_tris,
                          size_out, size_in, cap=None):
    """Tetrahedralise BETWEEN the cavity shell and the assembly skin.

    INTERIOR zone: the skin is a HOLE in the fill volume — outer surface
    loop the shell, inner loop the skin, both discrete and verbatim (the
    mechanism thin_volume_spike measured).

    OUTCROPPING zone (``cap`` given): the gap's boundary is ONE closed
    surface of three discrete pieces — the shell, the pre-meshed CAP over
    the bowl (the wall collar :func:`_outcrop_collar_3d` builds — in the
    box case an annulus between the shell's wall rim and the zone's band
    outline), and the INTERIOR part of the skin, which is open and
    rim-matched to the cap's hole. The band itself is not part of the
    gap's boundary — the zone touches the wall directly there.

    Returns ``(points, tets, moved, skin_out, n_shell, cap_out)`` with the
    fill's nodes ordered shell first, skin second, cap extras third, new
    points after.
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

        cap_tag = None
        if cap is None:
            outer = gmsh.model.geo.addSurfaceLoop([shell_tag])
            inner = gmsh.model.geo.addSurfaceLoop([skin_tag])
            vol = gmsh.model.geo.addVolume([outer, inner])
        else:
            cap_tag = gmsh.model.addDiscreteEntity(2)
            extra_first = n_shell + n_skin + 1
            if len(cap["extra_xyz"]):
                gmsh.model.mesh.addNodes(
                    2, cap_tag,
                    list(range(extra_first,
                               extra_first + len(cap["extra_xyz"]))),
                    np.asarray(cap["extra_xyz"], dtype=float)
                    .reshape(-1).tolist())
            n_rim = len(cap["rim_shell_local"])
            n_hole = len(cap["hole_skin_local"])

            def cap_tag_of(k):
                if k < n_rim:
                    return int(cap["rim_shell_local"][k]) + 1
                if k < n_rim + n_hole:
                    return (n_shell
                            + int(cap["hole_skin_local"][k - n_rim]) + 1)
                return extra_first + (k - n_rim - n_hole)

            gmsh.model.mesh.addElementsByType(
                cap_tag, 2, [],
                [cap_tag_of(int(v)) for t in cap["tris"] for v in t])
            loop = gmsh.model.geo.addSurfaceLoop(
                [shell_tag, cap_tag, skin_tag])
            vol = gmsh.model.geo.addVolume([loop])
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
        cap_out = None
        if cap_tag is not None:
            cet, _cei, cen_ = gmsh.model.mesh.getElements(2, cap_tag)
            for t, nodes in zip(cet, cen_):
                if t == 2:
                    cap_out = np.array([renum[int(x)] for x in nodes],
                                       dtype=np.int64).reshape(-1, 3)
        return points, tets, moved, skin_out, n_shell, cap_out
    finally:
        gmsh.finalize()


def _cell_centroids_of(dm, cell_mask):
    """Vertex-mean centroids of the masked cells (plex cell order)."""
    X = _coords(dm)
    vS, vE = dm.getDepthStratum(0)
    cS, _cE = dm.getHeightStratum(0)
    ids = np.flatnonzero(cell_mask)
    out = np.empty((len(ids), X.shape[1]))
    for k, c in enumerate(ids):
        verts = [int(q) - vS for q in dm.getTransitiveClosure(int(c) + cS)[0]
                 if vS <= int(q) < vE]
        out[k] = X[verts].mean(axis=0)
    return ids, out


def _footprint_from_samples(dm, band_mask, samples_ext, is_user_sample):
    """Band cells whose NEAREST extended-parametrisation sample is a USER
    point — the fault-footprint mask (#629, the honoured-paint rule): a
    painted rheology must stop at the fault the user specified, never
    extend into the extrapolated tip margin (measured: whole-band paint
    put the tip lobes ~2 elements past the mapped tips). Geometry-free:
    works for any curved strand because the parametrisation itself
    carries the user/extension distinction."""
    out = np.zeros_like(band_mask)
    ids, cen = _cell_centroids_of(dm, band_mask)
    if len(ids) == 0:
        return out
    d = cen[:, None, :] - samples_ext[None, :, :]
    nearest = np.argmin(np.einsum("ijk,ijk->ij", d, d), axis=1)
    out[ids[is_user_sample[nearest]]] = True
    return out


def _extend_polyline_2d(P, rings):
    """Continue a polyline ``rings`` points outward at both ends, linearly
    — the 2-D tip-margin builder (:func:`_extend_grid` one dimension
    down): end tangents at the local spacing, no invented curvature.
    ``rings`` is one count for both ends or a ``(start, end)`` pair (a
    junction end wants a longer reach than a free tip)."""
    P = np.asarray(P, dtype=float)
    r0, r1 = (rings, rings) if np.ndim(rings) == 0 else rings
    for _ in range(int(r0)):
        P = np.vstack([2.0 * P[0] - P[1], P])
    for _ in range(int(r1)):
        P = np.vstack([P, 2.0 * P[-1] - P[-2]])
    return P


def _mitred_reach_2d(S):
    """Per-vertex mitred unit-reach vectors of a sampled polyline.

    The outline() mechanics kept as a spine: segment normals averaged
    to the bisector, scaled 1/cos(θ/2) so the offset band keeps
    constant width through a turn. Sharp turns (interior angle under
    ~30°) are refused — no snap parameter, the #595 discipline."""
    t = np.diff(S, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    n = np.column_stack([-t[:, 1], t[:, 0]])
    reach = np.empty_like(S)
    reach[0], reach[-1] = n[0], n[-1]
    for k in range(1, len(S) - 1):
        m = n[k - 1] + n[k]
        nm = float(np.linalg.norm(m))
        half_cos = 0.5 * nm
        if half_cos < 0.25:
            raise ValueError(
                "the ladder polyline turns too sharply to rail with a "
                "mitre join (interior angle under ~30 degrees); smooth "
                "the trace or sample it finer.")
        reach[k] = m / nm / half_cos
    return reach


def _ladder_curved_assembly_2d(P, width, size, reach=None):
    """The curved 2-D ladder: rails + exact spine, built in numpy (#629).

    The 3-D extrusion (:func:`_ladder_assembly_3d`) one dimension down,
    for polylines the transfinite rectangle cannot represent (an
    S-bend, a mapped trace). The polyline is resampled equispaced in
    ARCLENGTH at ``size`` (ends always included — extent honoured;
    interior samples lie on the given chords), each sample offset
    ``±width/2`` along its mitred vertex normal — three rails of shared
    vertices, so a spine cut consumes existing vertices (#595) and a
    2:1 coarser band (``2 * size`` on the same polyline) nests
    vertex-for-vertex when the interval count halves exactly.

    Quads split to triangles with ALTERNATING diagonals (the
    transfinite band's pattern); every triangle's signed area must be
    positive — an over-tight bend for the width is a refusal, never a
    reorder. No gmsh, no CAD: the returned area is the mesh's own, so
    the meshed-vs-CAD gate holds trivially.
    """
    if reach is None:
        P = np.asarray(P, dtype=float)[:, :2]
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        L = float(arc[-1])
        if L <= 0.0:
            raise ValueError("the ladder polyline needs two distinct points")
        n_along = max(2, int(round(L / size)))
        s = np.linspace(0.0, L, n_along + 1)
        S = np.column_stack([np.interp(s, arc, P[:, 0]),
                             np.interp(s, arc, P[:, 1])])
        reach = _mitred_reach_2d(S)
    else:
        # Precomputed samples + reach: the NESTING contract. Coarser
        # levels must SUBSAMPLE one fine parametrisation (S[::2],
        # reach[::2]) — independently recomputed reach vectors differ
        # at shared points and break rail coincidence (measured: only
        # the spine nested).
        S = np.asarray(P, dtype=float)[:, :2]
        reach = np.asarray(reach, dtype=float)
        if reach.shape != S.shape:
            raise ValueError("reach must match the sample shape")
    n_along = len(S) - 1
    pts = np.concatenate([S + 0.5 * width * reach, S,
                          S - 0.5 * width * reach], axis=0)

    nv = n_along + 1
    tris = []
    for strip in (0, 1):                     # upper: rail+..spine; lower
        a0, b0 = strip * nv, (strip + 1) * nv
        for i in range(n_along):
            a, b = a0 + i, b0 + i
            if (i + strip) % 2 == 0:         # alternating diagonals
                tris += [[a, b, b + 1], [a, b + 1, a + 1]]
            else:
                tris += [[a, b, a + 1], [a + 1, b, b + 1]]
    tris = np.asarray(tris, dtype=np.int64)

    Q = pts[tris]
    twice = ((Q[:, 1, 0] - Q[:, 0, 0]) * (Q[:, 2, 1] - Q[:, 0, 1])
             - (Q[:, 1, 1] - Q[:, 0, 1]) * (Q[:, 2, 0] - Q[:, 0, 0]))
    if (twice >= 0.0).any() and (twice <= 0.0).any():
        raise ValueError(
            f"the curved ladder inverts "
            f"{int(min((twice >= 0).sum(), (twice <= 0).sum()))} of "
            f"{len(tris)} triangles: the bend is too tight for width "
            f"{width} (offset rails cross).")
    if (twice < 0.0).all():                  # normalise winding to CCW
        tris[:, [1, 2]] = tris[:, [2, 1]]
    return pts, tris, float(np.abs(twice).sum() / 2.0)


def _ladder_domain_gate_2d(pts, domain):
    """Refuse a structured band that protrudes past the domain bbox."""
    if domain is None:
        return
    lo = np.min([np.min(np.asarray(Lp, dtype=float), axis=0)
                 for Lp in domain], axis=0)
    hi = np.max([np.max(np.asarray(Lp, dtype=float), axis=0)
                 for Lp in domain], axis=0)
    if (pts < lo - 1e-12).any() or (pts > hi + 1e-12).any():
        raise ValueError(
            "the ladder band protrudes past the domain; it is structured "
            "and cannot be clipped. Shorten the trace or use the default "
            "band (which clips).")


def _occ_ladder_assembly_2d(polylines, width, size, assembly="fuse",
                            domain=None):
    """The cut-ready band: transfinite, THREE nodes across, spine included.

    The default (frontal-meshed) band's spine cut works but SNAPS — the cut
    machinery pulls the nearest vertex onto the line, and in a band whose
    half-width is inside the snap reach of the local cell size that nearest
    vertex can be a RAIL: measured, one rail vertex dragged onto the spine,
    pinching the band shut (#595). There is no snap parameter that
    guarantees rail safety, because snapping is measured along edges and is
    blind to a vertex's distance from the line.

    Prevention instead of tuning: mesh the band TRANSFINITE with three
    nodes across. That builds the rails plus an exact centreline row of
    vertices, edge-connected, at the rail spacing; a spine cut sampled AT
    those vertices crosses only existing vertices — nothing snaps, nothing
    is inserted, and the rails cannot be touched because nothing needs to
    move. This is the mandatory band for a centreline cut/split (the
    composed ribbon hierarchy, #629) and the reason the across-band node
    count sets the thinnest paintable interior layer.

    Same hook signature as :func:`_occ_assembly_2d`. Restrictions, each
    checked: exactly ONE polyline, and a STRAIGHT one (the transfinite
    frame is a single ruled rectangle). ``assembly`` is irrelevant to a
    single face and ignored. ``domain`` is accepted but NOT used to clip —
    the ladder must lie inside the domain; a protruding band is refused
    here rather than silently losing its transfinite structure to an OCC
    intersection.
    """
    import gmsh

    if len(polylines) != 1:
        raise ValueError("the ladder band takes exactly one polyline")
    p0 = polylines[0]
    if (isinstance(p0, tuple) and len(p0) == 2
            and np.asarray(p0[0]).ndim == 2):
        # A precomputed (samples, reach) pair: the NESTING contract —
        # rung positions and mitred reach vectors from ONE parametrisation
        # (a coarser level passes S[::2], reach[::2]). Used verbatim.
        S, R = (np.asarray(q, dtype=float) for q in p0)
        pts_c, tris_c, area_c = _ladder_curved_assembly_2d(
            S, width, size, reach=R)
        _ladder_domain_gate_2d(pts_c, domain)
        return pts_c, tris_c, area_c
    P = np.asarray(p0, dtype=float)[:, :2]
    a, b = P[0], P[-1]
    t = b - a
    L = float(np.linalg.norm(t))
    if L <= 0.0:
        raise ValueError("the ladder polyline needs two distinct end points")
    t = t / L
    n = np.array([-t[1], t[0]])
    off = np.abs((P - a) @ n)
    if off.max() > 1e-9 * max(L, 1.0):
        # A CURVED trace (an S-bend, a mapped fault): the numpy ladder —
        # same three-rail structure, mitred. The straight path below
        # stays on gmsh transfinite so the recorded composed benchmark
        # remains bit-identical.
        pts_c, tris_c, area_c = _ladder_curved_assembly_2d(P, width, size)
        _ladder_domain_gate_2d(pts_c, domain)
        return pts_c, tris_c, area_c
    corners = np.array([a + 0.5 * width * n, b + 0.5 * width * n,
                        b - 0.5 * width * n, a - 0.5 * width * n])
    if domain is not None:
        lo = np.min([np.min(np.asarray(Lp, dtype=float), axis=0)
                     for Lp in domain], axis=0)
        hi = np.max([np.max(np.asarray(Lp, dtype=float), axis=0)
                     for Lp in domain], axis=0)
        if (corners < lo - 1e-12).any() or (corners > hi + 1e-12).any():
            raise ValueError(
                "the ladder band protrudes past the domain; it is meshed "
                "transfinite and cannot be clipped. Shorten the trace or "
                "use the default band (which clips).")
    n_along = max(2, int(round(L / size)))

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("uw_ladder_band")
        occ = gmsh.model.occ
        pts = [occ.addPoint(q[0], q[1], 0.0) for q in corners]
        lines = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        surf = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        occ.synchronize()
        cad_area = occ.getMass(2, surf)
        gmsh.model.mesh.setTransfiniteCurve(lines[0], n_along + 1)
        gmsh.model.mesh.setTransfiniteCurve(lines[2], n_along + 1)
        gmsh.model.mesh.setTransfiniteCurve(lines[1], 3)   # 3 across:
        gmsh.model.mesh.setTransfiniteCurve(lines[3], 3)   # rails + spine
        gmsh.model.mesh.setTransfiniteSurface(surf, "AlternateLeft")
        gmsh.model.mesh.generate(2)

        tags, xyz, _ = gmsh.model.mesh.getNodes()
        xy = np.asarray(xyz).reshape(-1, 3)[:, :2]
        renum = {int(tg): i for i, tg in enumerate(tags)}
        et, _ei, en = gmsh.model.mesh.getElements(2, surf)
        tris = []
        for ty, nodes in zip(et, en):
            if ty == 2:
                tris.append(np.array([renum[int(x)] for x in nodes],
                                     dtype=np.int64).reshape(-1, 3))
        return xy, np.vstack(tris), float(cad_area)
    finally:
        gmsh.finalize()


def _occ_assembly_2d(polylines, width, size, assembly="fuse", domain=None,
                     embed=None):
    """Thicken each polyline into a ribbon, resolve overlaps, mesh.

    The 2-D thin volume: a ribbon is the mitred outline of one polyline, and
    the ribbons of a network are resolved against one another in CAD.
    ``assembly`` chooses that resolution: ``"fuse"`` returns the union as ONE
    face, ``"fragment"`` keeps every overlap piece as its own face. Both mesh
    the same region; they differ in the internal seams the mesher must
    honour. Returns ``(points, triangles, cad_area)``, the area being that of
    the resolved faces — the union — under either choice.

    ``domain`` — the mesh's boundary loops (:func:`_domain_loops_2d`) —
    applies the specify-long contract one dimension down from
    :func:`_occ_assembly_3d`: the resolved faces are INTERSECTED with the
    domain built as OCC geometry from those very loops, so polylines may
    protrude — a ribbon reaching a boundary leaves its clipped edge exactly
    on the boundary's own facets (snapped onto them after meshing,
    defensively). ``cad_area`` is then the clipped area, so the
    meshed-vs-CAD gate holds unchanged.

    ``embed`` — polylines (each ``(n, 2)``, strictly inside the resolved
    faces) whose points and segments are EMBEDDED in the face before
    meshing, so they are vertices and edges of the mesh exactly. This is
    the NETWORK path (``mesher="network"``): the fuse resolves junctions
    between touching ribbons as ordinary cells, and the embedded spines
    give a split cut its own vertices to walk (#595: nothing snaps) —
    the two properties the sequential ladder and the plain fuse each
    had only one of. Each embedded point carries its local segment
    length as mesh size so gmsh does not subdivide the spine.
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
            # Where ribbons overlap, fragment's seams are the boundary of the
            # overlap, and for a tangential merge that boundary is a lens
            # closing at the convergence angle — a chain of slivers the
            # mesher must resolve (measured on the sole geometry of
            # test_0855: minimum angle 0 degrees and 22 cells under 5,
            # against 37 degrees and none fused). The seams carry no physics
            # to preserve: the zone is one label, and a cell's fault
            # properties are read from the Surface objects by proximity, not
            # from the piece it was meshed in.
            if assembly == "fuse":
                occ.fuse([(2, surfs[0])], [(2, t) for t in surfs[1:]])
            else:
                occ.fragment([(2, surfs[0])], [(2, t) for t in surfs[1:]])
        comp = None
        if domain is not None:
            comp = [_compress_collinear_loop(np.asarray(L, dtype=float))
                    for L in domain]
            occ.synchronize()
            faces = [t for _d, t in gmsh.model.getEntities(2)]
            tool = _occ_domain_2d(occ, comp)
            occ.intersect([(2, t) for t in faces], [(2, tool)])
        occ.synchronize()

        faces = gmsh.model.getEntities(2)
        cad_area = sum(occ.getMass(2, t) for _d, t in faces)

        if embed:
            face_tags = [t for _d, t in faces]
            for P in embed:
                P = np.asarray(P, dtype=float)[:, :2]
                if len(P) < 2:
                    continue
                seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
                hp = np.concatenate([[seg[0]], 0.5 * (seg[1:] + seg[:-1]),
                                     [seg[-1]]])
                ptags = [occ.addPoint(q[0], q[1], 0.0, meshSize=float(h))
                         for q, h in zip(P, hp)]
                ltags = [occ.addLine(ptags[i], ptags[i + 1])
                         for i in range(len(ptags) - 1)]
                occ.synchronize()
                mid = P[len(P) // 2]
                host = [t for t in face_tags
                        if gmsh.model.isInside(2, t, [mid[0], mid[1], 0.0])]
                if not host:
                    raise RuntimeError(
                        "network assembly: an embedded spine lies outside "
                        "the resolved ribbon faces (trim its ends inside "
                        "the caps).")
                gmsh.model.mesh.embed(0, ptags, 2, host[0])
                gmsh.model.mesh.embed(1, ltags, 2, host[0])

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
        if comp is not None:
            # OCC's clipped edges sit within rounding of the boundary; the
            # band logic and the wall relabel need EXACT membership, so
            # snap defensively (the 3-D path does the same on its planes).
            xy = _snap_to_boundary_2d(xy, comp)
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


def _skin_loops(skin_edges, what="the assembly's skin"):
    """Order a 2-D skin's edges into closed loops of vertex ids.

    A manifold skin gives every vertex exactly two incident edges; anything
    else is a defect of the mesh the edges bound and is refused.
    """
    adj = {}
    for a, b in skin_edges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    if any(len(v) != 2 for v in adj.values()):
        raise RuntimeError(f"{what} is not a set of closed loops; "
                           "the mesh it bounds is defective")
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


def _outcrop_chain_2d(loops, band_pairs):
    """Split the outcropping skin loop into its wall band and interior chain.

    The clipped skin is still a set of closed loops — clipping puts part of
    a loop ON the wall line, it does not open the loop. The fill, though,
    cannot take the outcropping loop as a hole: its boundary role is played
    by the INTERIOR CHAIN, the open path left when the band's edges are
    removed. ``band_pairs`` is that band as a set of frozen vertex pairs.

    Exactly one loop may carry band edges, and they must form ONE contiguous
    arc of it — a ribbon meeting the wall in two separate bands is the 2-D
    face of the box-edge case the 3-D band split refuses. Returns
    ``(chain, holes)``: the chain as an open vertex path from one band
    endpoint to the other, in the loop's own orientation, and the loops
    without band edges unchanged.
    """
    chain, holes = None, []
    for loop in loops:
        n = len(loop)
        on_band = [frozenset((loop[i], loop[(i + 1) % n])) in band_pairs
                   for i in range(n)]
        if not any(on_band):
            holes.append(loop)
            continue
        if chain is not None:
            raise NotImplementedError(
                "two skin loops meet the domain wall; only one ribbon may "
                "outcrop.")
        starts = sum(1 for i in range(n)
                     if on_band[i] and not on_band[i - 1])
        if starts != 1:
            raise NotImplementedError(
                "the ribbon meets the wall in more than one band; a "
                "multiply-outcropping zone is not built.")
        i0 = next(i for i in range(n) if on_band[i] and not on_band[i - 1])
        k = sum(on_band)
        chain = [loop[(i0 + k + j) % n] for j in range(n - k + 1)]
    return chain, holes


def _arc_project(polyline, p):
    """``(distance, arc length)`` of the closest point on an open polyline."""
    best_d, best_s = np.inf, 0.0
    s0 = 0.0
    for i in range(len(polyline) - 1):
        A, B = polyline[i], polyline[i + 1]
        e = B - A
        length = float(np.linalg.norm(e))
        u = float(np.clip((p - A) @ e / (length * length), 0.0, 1.0))
        d = float(np.linalg.norm(p - (A + u * e)))
        if d < best_d:
            best_d, best_s = d, s0 + u * length
        s0 += length
    return best_d, best_s


def _outcrop_ring_splice(ring, X, boundary_pairs, chain_ids, chain_ends):
    """Replace the cavity ring's boundary span with the ribbon's interior
    chain.

    The raw ring of an outcropping carve runs ALONG the domain boundary
    through vertices about to be deleted. The fill's boundary instead
    descends around the ribbon: the ring's single contiguous run of
    boundary edges is removed and the chain is spliced between the run's
    surviving end vertices, oriented to meet them. The two splice segments
    are the 2-D cap — what remains of the 3-D wall annulus one dimension
    down.

    ``boundary_pairs`` is the mesh's own boundary-edge vertex pairs, so
    the span is a topological run, not a coordinate test.  ``chain_ids``
    are the chain's rows in the fill's combined numbering, ``chain_ends``
    the coordinates of its two end nodes; coverage and orientation are
    arc projections onto the removed span's own polyline, so a curved
    boundary orders exactly as a straight wall did. Returns
    ``(spliced_ring, removed_wall_pairs)``, the removed pairs as old
    vertex rows for the wall-label discovery. A second boundary run
    refuses — the carve spilled onto the boundary away from the outcrop.
    """
    n = len(ring)
    wall_edge = [frozenset((ring[i], ring[(i + 1) % n])) in boundary_pairs
                 for i in range(n)]
    if not any(wall_edge):
        raise RuntimeError(
            "the outcrop band does not meet the cavity's wall span; raise "
            "`clearance` so the carve reaches the wall.")
    starts = sum(1 for i in range(n) if wall_edge[i] and not wall_edge[i - 1])
    if starts != 1:
        raise RuntimeError(
            "the carve reached the domain wall in two separate spans; "
            "reduce `clearance` or move the zone off the wall.")
    i0 = next(i for i in range(n) if wall_edge[i] and not wall_edge[i - 1])
    k = sum(wall_edge)
    removed = [(ring[(i0 + j) % n], ring[(i0 + j + 1) % n])
               for j in range(k)]
    span = X[np.asarray([ring[(i0 + j) % n] for j in range(k + 1)])]
    ends = [_arc_project(span, np.asarray(c, dtype=float))
            for c in chain_ends]
    strictly_inside = all(
        d < 1e-9
        and np.linalg.norm(c - span[0]) > 1e-9
        and np.linalg.norm(c - span[-1]) > 1e-9
        for (d, _s), c in zip(ends, chain_ends))
    if not strictly_inside:
        raise RuntimeError(
            "the cavity's wall span does not cover the outcrop band; raise "
            "`clearance`.")
    # The surviving arc, corner_r around to corner_l, then the chain with
    # its corner_l-side end first — the loop's orientation is preserved.
    tail = [ring[(i0 + k + j) % n] for j in range(n - k + 1)]
    seq = chain_ids if ends[0][1] <= ends[1][1] else chain_ids[::-1]
    return tail + list(seq), removed


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
                # TODO(BUG): on a GRADED base (adapt() toward the traces) the cleared
                # cells stop forming one simple hole at the default clearance: the
                # S-fault rig at fine width (w = 0.01) builds on the uniform base at
                # clearance 0.3 but on the one-level graded base only at 1.0. The
                # clearing should be sized by the LOCAL cell size, not one factor.
                # Found 2026-08-27 running FaultNetwork.build on the rig.
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


def _ring_growing_multi(cells, drop, held_mask, allow_multiple=False,
                        X=None):
    """The cavity rings, one per connected component of the drop set.

    :func:`_ring_growing`'s pinch growth first (a pinch between two
    cavities merges them), then each component's own simple ring. With
    ``allow_multiple`` false a second component is the refusal
    :func:`_ring_growing` gives; with it true — the seam-ligament placement,
    where a band crossing a rank in and out leaves two cavities on it —
    every component is carved and filled on its own. Returns
    ``([(ring, component_mask), ...], drop)``.
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
            ids = np.flatnonzero(drop)
            comp = _assembly_components(cells[ids])
            n_comp = int(comp.max()) if len(comp) else 0
            if n_comp > 1 and not allow_multiple:
                raise RuntimeError(
                    "the cells cleared for the thin volume do not leave one "
                    "simple hole. Raise `clearance`.")
            out = []
            grew = False
            for k in range(1, n_comp + 1):
                sel = ids[comp == k]
                ring = _cavity_ring(cells, sel)
                if ring is None and X is not None:
                    # an ISLAND: kept cells enclosed by the cavity (a narrow
                    # clearance between two strands leaves them) make the
                    # ring an annulus. Drop the enclosed cells and go round
                    # again; a held cell inside refuses.
                    island = _enclosed_cells(cells, X, sel, drop)
                    if island.size:
                        if held_mask[island].any():
                            raise RuntimeError(
                                "the cavity encloses a cell held for a "
                                "surface already embedded; move the zone "
                                "away or raise `clearance`.")
                        drop[island] = True
                        grew = True
                        break
                if ring is None:
                    raise RuntimeError(
                        "the cells cleared for the thin volume do not leave "
                        "one simple hole. Raise `clearance`.")
                mask = np.zeros(len(cells), dtype=bool)
                mask[sel] = True
                out.append((ring, mask))
            if grew:
                continue
            return out, drop
        for v in pinch:
            grow = (cells == v).any(axis=1)
            if (grow & held_mask).any():
                if not allow_multiple:
                    raise RuntimeError(
                        "closing the cavity needs a cell held for a surface "
                        "already embedded; move the zone away or raise "
                        "`clearance`.")
                # The pinch abuts the seam ligament, which cannot be
                # carved: SHRINK the cavity instead — keep the largest fan
                # of dropped cells at the vertex, restore the rest. The
                # band is clipped by whatever the cavity ends up holding.
                fan = np.flatnonzero(drop & grow)
                comp = _assembly_components(cells[fan])
                if int(comp.max()) < 2:
                    raise RuntimeError(
                        "the cavity pinches at the seam ligament in a way "
                        "shrinking cannot resolve; raise `clearance` or "
                        "widen `ligament`.")
                sizes = np.bincount(comp)
                biggest = int(np.argmax(sizes[1:])) + 1
                drop[fan[comp != biggest]] = False
                break           # the ring changed: recompute the pinches
            else:
                drop |= grow
    raise RuntimeError(
        "the cavity did not become simple in 20 growth rounds; raise "
        "`clearance`.")


def _segments_cross(P, Q, A, B):
    """Whether segment PQ crosses segment AB (proper or touching)."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(A, B, P), orient(A, B, Q)
    d3, d4 = orient(P, Q, A), orient(P, Q, B)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    eps = 1e-14
    return (abs(d1) < eps or abs(d2) < eps or abs(d3) < eps
            or abs(d4) < eps) and (
        min(A[0], B[0]) - eps <= max(P[0], Q[0])
        and max(A[0], B[0]) + eps >= min(P[0], Q[0])
        and min(A[1], B[1]) - eps <= max(P[1], Q[1])
        and max(A[1], B[1]) + eps >= min(P[1], Q[1]))


def _conform_fill_loops_2d(rings, X, victim, seam_edge_set, asm_pts,
                           loops_asm, skin_edge_mine, nX):
    """This rank's fill loops when the band is meshed through the seam.

    The fill's boundary is a graph: every cavity ring edge that is not a
    seam edge the band crosses, every skin edge whose outside lies in this
    rank's cavity (``skin_edge_mine``, decided globally), and a connector
    from each crossed span's surviving end to the nearest free end of
    this rank's skin runs (a run's end is where the skin changes hands —
    the seam's own crossing of a rail — so the connector is the same edge
    on both sides). Every vertex then has two edges, and walking the graph
    gives closed loops: the outer ones are filled, each with the loops it
    contains as holes. Returns ``[(outer_loop, [hole_loops]), ...]`` in
    the fill's combined numbering (old vertices, then ``nX`` + band
    vertex), outer loops anticlockwise.
    """
    from collections import defaultdict

    adj = defaultdict(list)

    def link(a, b):
        adj[a].append(b)
        adj[b].append(a)

    span_ends = []
    for ring in rings:
        ring = [int(v) for v in ring]
        n = len(ring)
        removed, meets = [], []
        for i in range(n):
            a, b = ring[i], ring[(i + 1) % n]
            seam = (min(a, b), max(a, b)) in seam_edge_set
            hit = seam and _edge_meets_band(X[a], X[b], asm_pts, loops_asm)
            removed.append(seam and (victim[a] or victim[b] or hit))
            meets.append(hit)
        if all(removed):
            raise RuntimeError(
                "a cavity's whole ring lies on the seam inside the band; "
                "reduce `clearance`.")
        # a span is a maximal run of removed edges between surviving
        # vertices; one the band passes through pairs its ends with the
        # skin's run ends below, one the band only comes near (its seam
        # vertices within the carve's reach) is closed by a straight
        # connector between its ends — the same edge on both sides
        starts = [i for i in range(n) if removed[i]
                  and (not removed[i - 1] or not victim[ring[i]])]
        for i0 in starts:
            k, crossing = 1, meets[i0]
            while removed[(i0 + k) % n] and victim[ring[(i0 + k) % n]]:
                crossing |= meets[(i0 + k) % n]
                k += 1
            v1, v2 = ring[i0], ring[(i0 + k) % n]
            if victim[v1] or victim[v2]:
                raise RuntimeError(
                    "a crossed seam span has no surviving end; raise "
                    "`clearance`.")
            if crossing:
                span_ends += [v1, v2]
            else:
                link(v1, v2)
        for i in range(n):
            if not removed[i]:
                link(ring[i], ring[(i + 1) % n])
    # this rank's skin edges, and the ends of its runs
    deg = defaultdict(int)
    for a, b in skin_edge_mine:
        link(nX + a, nX + b)
        deg[a] += 1
        deg[b] += 1
    run_ends = [v for v, d in deg.items() if d == 1]
    if len(run_ends) != len(span_ends):
        raise RuntimeError(
            f"the seam crossings do not pair up on this rank: "
            f"{len(span_ends)} span end(s) for {len(run_ends)} skin run "
            "end(s); the crossing geometry is not one this placement "
            "handles (raise `clearance`, or move the seam).")
    free = list(run_ends)
    for v in span_ends:
        d = np.linalg.norm(asm_pts[free] - X[v], axis=1)
        t = free.pop(int(np.argmin(d)))
        link(v, nX + t)
    bad = [v for v, nb in adj.items() if len(nb) != 2]
    if bad:
        raise RuntimeError(
            f"the fill boundary is not a set of simple loops on this rank "
            f"({len(bad)} vertices of degree != 2); the crossing geometry "
            "is not one this placement handles.")
    seen = set()
    loops = []
    for start in list(adj):
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [w for w in adj[cur] if w != prev]
            nxt = nxt[0] if nxt else adj[cur][0]
            if nxt == start:
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
            if len(loop) > len(adj):
                raise RuntimeError("the fill boundary walk did not close")
        loops.append(loop)

    def xy(v):
        return X[v] if v < nX else asm_pts[v - nX]

    def area(loop):
        P = np.array([xy(v) for v in loop])
        return 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1)
                                  - np.roll(P[:, 0], -1) * P[:, 1]))

    polys = [np.array([xy(v) for v in loop]) for loop in loops]
    inside_of = []
    for i, loop in enumerate(loops):
        holders = [j for j, Pj in enumerate(polys) if j != i
                   and _inside_polygon(Pj, xy(loop[0]))]
        inside_of.append(holders)
    out = []
    for i, loop in enumerate(loops):
        if len(inside_of[i]) % 2 == 1:
            continue                        # a hole (odd nesting depth)
        outer = loop if area(loop) > 0 else loop[::-1]
        holes = [loops[j] for j in range(len(loops))
                 if i in inside_of[j] and len(inside_of[j]) == len(inside_of[i]) + 1]
        out.append((outer, holes))
    return out


def _edge_meets_band(A, B, asm_pts, loops_asm):
    """Whether segment AB crosses the band's skin or lies inside the band."""
    for loop in loops_asm:
        m = len(loop)
        for i in range(m):
            p, q = int(loop[i]), int(loop[(i + 1) % m])
            if _segments_cross(A, B, asm_pts[p], asm_pts[q]):
                return True
    mid = 0.5 * (A + B)
    return any(_inside_polygon(asm_pts[np.asarray(loop, dtype=int)], mid)
               for loop in loops_asm)


def _enclosed_cells(cells, X, sel, drop):
    """Kept cells enclosed by the cavity component ``sel`` — the cells inside
    its inner boundary loops. The component's boundary edges are walked
    into loops; the loop of largest area is the outer ring, and a kept
    cell whose centroid lies inside any other loop is an island cell."""
    directed = {}
    for ci in sel:
        v0, v1, v2 = (int(v) for v in cells[ci])
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            directed[(a, b)] = ci
    step = {}
    for (a, b) in directed:
        if (b, a) not in directed:
            step.setdefault(a, []).append(b)
    loops = []
    seen = set()
    for start in list(step):
        if start in seen:
            continue
        loop, cur = [start], start
        while True:
            seen.add(cur)
            nxt = [w for w in step.get(cur, []) if w not in seen] or \
                step.get(cur, [])
            if not nxt:
                break
            cur = nxt[0]
            if cur == start:
                break
            loop.append(cur)
            if len(loop) > len(step):
                break
        loops.append(loop)
    if len(loops) < 2:
        return np.zeros(0, dtype=np.int64)

    def area(loop):
        P = X[np.asarray(loop)]
        return abs(0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1)
                                      - np.roll(P[:, 0], -1) * P[:, 1])))

    outer = max(range(len(loops)), key=lambda i: area(loops[i]))
    kept = np.flatnonzero(~drop)
    cen = X[cells[kept]].mean(axis=1)
    inside = np.zeros(len(kept), dtype=bool)
    for i, loop in enumerate(loops):
        if i == outer or len(loop) < 3:
            continue
        P = X[np.asarray(loop)]
        inside |= np.array([_inside_polygon(P, c) for c in cen])
    return kept[inside].astype(np.int64)


def _manifold_subset_2d(tris, kept):
    """Shrink a triangle subset until its boundary is a set of simple loops.

    A clipped subset can pinch at a vertex — two fans of kept triangles
    meeting only there (a bow-tie), or a triangle hanging by one vertex —
    and such a vertex carries four boundary edges, which no closed-loop
    walk can pass. Every triangle at such a vertex is dropped, and the
    check repeats until each vertex carries zero or two boundary edges.
    Returns the new mask; the input is untouched.
    """
    from collections import Counter
    kept = kept.copy()
    for _round in range(50):
        sub = tris[kept]
        if not len(sub):
            return kept
        use = Counter()
        for a, b, c in sub:
            for e in ((a, b), (b, c), (c, a)):
                use[(min(e), max(e))] += 1
        at_v = Counter()
        for (a, b), n in use.items():
            if n == 1:
                at_v[a] += 1
                at_v[b] += 1
        bad = {v for v, n in at_v.items() if n != 2}
        if not bad:
            return kept
        hit = np.array([any(int(v) in bad for v in t) for t in tris])
        kept &= ~hit
    raise RuntimeError("the clipped band did not become manifold")


def _gmsh_fill_plain_3d(shell_xyz, shell_tris, h):
    """Tetrahedralise inside the shell with NOTHING embedded — the removal
    fill. The shell is a discrete surface carrying its triangulation
    verbatim; the interior comes back at the background scale ``h``, which
    is what returns a removed object's region to the surrounding mesh.

    Returns ``(points, tets, moved, n_shell)``, nodes shell-first.
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("uw_removal_fill")
        n_shell = len(shell_xyz)
        shell_tag = gmsh.model.addDiscreteEntity(2)
        gmsh.model.mesh.addNodes(2, shell_tag, list(range(1, n_shell + 1)),
                                 shell_xyz.reshape(-1).tolist())
        gmsh.model.mesh.addElementsByType(
            shell_tag, 2, [], (shell_tris + 1).reshape(-1).tolist())
        loop = gmsh.model.geo.addSurfaceLoop([shell_tag])
        vol = gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
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
        et, _ei, en = gmsh.model.mesh.getElements(3, vol)
        tets = None
        for t, nodes in zip(et, en):
            if t == 4:
                tets = np.array([renum[int(x)] for x in nodes],
                                dtype=np.int64).reshape(-1, 4)
        if tets is None:
            raise RuntimeError("gmsh produced no tetrahedra for the removal")
        moved = sum(1 for t in range(1, n_shell + 1)
                    if not np.array_equal(points[renum[t]],
                                          shell_xyz[t - 1]))
        return points, tets, moved, n_shell
    finally:
        gmsh.finalize()


def _labelled_face_soup(dm, names_values):
    """The (points, facets) soup of every labelled facet, ALLGATHERED.

    A removal must mark against the object's global geometry whatever the
    current distribution — the object was embedded rank-locally, but a
    checkpoint reload or a later redistribution may have scattered it. The
    soup is small (an object's skin) and duplicates across strata are
    harmless to a distance query. Facets are triangles in 3-D and edges in
    2-D; the soup's connectivity is trivial (one facet per row).
    """
    dim = dm.getDimension()
    nvf = dim                       # vertices per facet
    vS, vE = dm.getDepthStratum(0)
    fS, fE = dm.getHeightStratum(1)
    X = _coords(dm)[: vE - vS]
    local = []
    for name, value in names_values:
        if not dm.hasLabel(name):
            continue
        label = dm.getLabel(name)
        if label.getStratumSize(int(value)) == 0:
            continue
        for p in label.getStratumIS(int(value)).getIndices():
            p = int(p)
            if not (fS <= p < fE):
                continue
            verts = [int(q) - vS for q in dm.getTransitiveClosure(p)[0]
                     if vS <= int(q) < vE]
            local.append(X[verts])
    comm = uw.mpi.comm
    gathered = comm.allgather(np.array(local).reshape(-1, nvf, dim))
    soup = np.vstack([g for g in gathered if len(g)]) \
        if any(len(g) for g in gathered) else np.empty((0, nvf, dim))
    pts = soup.reshape(-1, dim)
    facets = np.arange(len(pts), dtype=np.int64).reshape(-1, nvf)
    return pts, facets


def _locked_edges_excluding(dm, exclude):
    """The 2-D interface-edge flags, with named (label, value) pairs unset.

    A removal must not hold its own object's edges against itself — the
    2-D form of the 3-D readers' ``exclude=``. An edge carried by BOTH an
    excluded pair and another interface label is unlocked here; the removal
    gate on that other label's counts is what catches a genuine overlap.
    """
    locked = reconnect._interface_edges(dm).copy()
    pStart, _pEnd = dm.getChart()
    eS, eE = dm.getDepthStratum(1)
    for name, value in exclude:
        if not dm.hasLabel(name):
            continue
        lab = dm.getLabel(name)
        if lab.getStratumSize(int(value)) == 0:
            continue
        for p in lab.getStratumIS(int(value)).getIndices():
            p = int(p)
            if eS <= p < eE:
                locked[p - pStart] = False
    return locked


def _interface_vertices_and_cells_excluding(dm, n_vertices, n_cells,
                                            exclude):
    """:func:`_interface_vertices_and_cells`, minus the excluded labels."""
    pStart, _pEnd = dm.getChart()
    vS, _vE = dm.getDepthStratum(0)
    cS, _cE = dm.getHeightStratum(0)
    locked = _locked_edges_excluding(dm, exclude)
    vertices = np.zeros(n_vertices, dtype=bool)
    cells = np.zeros(n_cells, dtype=bool)
    for e in _interior_interface_facets(dm, locked, pStart):
        for v in dm.getCone(e):
            vertices[int(v) - vS] = True
        for c in dm.getSupport(e):
            cells[int(c) - cS] = True
    return vertices, cells


def _interface_facet_counts_excluding(dm, exclude):
    """:func:`_interface_facet_counts`, minus the excluded labels."""
    pStart, _pEnd = dm.getChart()
    exclude = set((n, int(v)) for n, v in exclude)
    locked = _locked_edges_excluding(dm, exclude)
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
            if (name, int(val)) in exclude:
                continue
            if label.getStratumSize(int(val)) == 0:
                continue
            held = [p for p in label.getStratumIS(int(val)).getIndices()
                    if int(p) in interior]
            if held:
                counts[(name, int(val))] = len(held)
    return counts


def _remove_embedded_2d(dm, label, label_value, clearance, verbose):
    """The removal one dimension down: seed from the labels, ring carve,
    PLAIN 2-D fill at background scale, same gates. Serial and parallel
    through the same gather-first mechanism."""
    comm = uw.mpi.comm
    skin_label = label + "_skin"
    removed_pairs = ((label, int(label_value)),
                     (skin_label, int(label_value)))

    soup_pts, soup_segs = _labelled_face_soup(dm, removed_pairs)
    n_soup = int(comm.allreduce(len(soup_segs), op=MPI.MAX))
    if n_soup == 0:
        raise ValueError(
            f"nothing is embedded under ({label!r}, {label_value}); "
            "there is nothing to remove.")
    seg_pairs = [(int(a), int(b)) for a, b in soup_segs]

    vS, vE = dm.getDepthStratum(0)
    pStart, pEnd = dm.getChart()
    X = _coords(dm)[: vE - vS]
    cells = _cells_anticlockwise(dm, X)
    h_vertex, _hc = _vertex_h_3d(dm, cells, len(X))
    d_skin = _segments_distance(X, soup_pts, seg_pairs)
    reach_v = clearance * h_vertex

    mark = np.zeros(pEnd - pStart, dtype=np.int32)
    mark[np.flatnonzero(d_skin < reach_v + 1.0 * h_vertex)
         + vS - pStart] = 1
    cS0, cE0 = dm.getHeightStratum(0)
    if dm.hasLabel(label):
        lab = dm.getLabel(label)
        if lab.getStratumSize(int(label_value)) > 0:
            for p in lab.getStratumIS(int(label_value)).getIndices():
                p = int(p)
                if cS0 <= p < cE0:
                    for q in dm.getTransitiveClosure(p)[0]:
                        if vS <= int(q) < vE:
                            mark[int(q) - pStart] = 1

    area_before = np.array([float(cell_areas(dm).sum())
                            if len(cells) else 0.0])
    comm.Allreduce(MPI.IN_PLACE, area_before, op=MPI.SUM)

    dm_work, moved = _gather_region(dm, mark, verbose=verbose)
    if moved:
        vS, vE = dm_work.getDepthStratum(0)
        pStart, pEnd = dm_work.getChart()
        X = _coords(dm_work)[: vE - vS]
        cells = _cells_anticlockwise(dm_work, X)
        h_vertex, _hc = _vertex_h_3d(dm_work, cells, len(X))
        d_skin = _segments_distance(X, soup_pts, seg_pairs)
        reach_v = clearance * h_vertex

    shared = _shared_point_flags(dm_work).astype(bool)
    on_wall = _true_wall_vertex_mask(dm_work, len(X))
    held_v, held_c = _interface_vertices_and_cells_excluding(
        dm_work, len(X), len(cells), removed_pairs)
    held_counts = _interface_facet_counts_excluding(dm_work, removed_pairs)

    cS, _cE = dm_work.getHeightStratum(0)
    seed = np.zeros(len(cells), dtype=bool)
    if dm_work.hasLabel(label):
        lab = dm_work.getLabel(label)
        if lab.getStratumSize(int(label_value)) > 0:
            for p in lab.getStratumIS(int(label_value)).getIndices():
                p = int(p)
                if cS <= p < cS + len(cells):
                    seed[p - cS] = True

    n_region = int((d_skin < reach_v).sum() + seed.sum())
    owners = np.asarray(comm.allgather(n_region))
    target = int(np.argmax(owners))

    failure = None
    surgery = None
    if comm.rank == target:
        try:
            beside_held = np.zeros(len(X), dtype=bool)
            beside_held[cells[held_c].ravel()] = True
            protected = on_wall | held_v | beside_held
            victim = (d_skin < reach_v) & ~protected
            drop = victim[cells].any(axis=1) | seed
            drop &= ~held_c
            need = victim[cells].any(axis=1) | seed
            if (need & held_c).any():
                raise RuntimeError(
                    "the removal's cavity needs a cell that belongs to a "
                    "DIFFERENT embedded surface; remove that one first or "
                    "lower `clearance`.")
            drop |= need
            if not drop.any():
                raise ValueError("nothing to remove meets this rank's cells")

            ring, drop = _ring_growing(cells, drop, held_c)
            if on_wall[np.asarray(ring)].any():
                raise RuntimeError(
                    "the removal's cavity reached the domain wall; raise "
                    "`clearance` margins or remove serially.")
            if victim[np.asarray(ring)].any():
                raise RuntimeError(
                    "a deleted vertex is on the cavity boundary")

            referenced = np.zeros(len(X), dtype=bool)
            if (~drop).any():
                referenced[cells[~drop].ravel()] = True
            orphan = ~referenced & ~victim
            if orphan[on_wall].any():
                raise RuntimeError(
                    "the removal would strand a domain-wall vertex")
            victim |= orphan

            gap_tris, extra = _gmsh_fill_2d(X, ring, None)
            placed = np.asarray(extra, dtype=float).reshape(-1, 2)

            def mixed(v):
                return int(v) if v < len(X) else -(int(v) - len(X) + 1)

            made = [tuple(mixed(v) for v in t) for t in gap_tris]

            touched = set()
            for c in np.flatnonzero(drop):
                for q in dm_work.getTransitiveClosure(int(c) + cS)[0]:
                    touched.add(int(q))
            if any(shared[q - pStart] for q in touched):
                raise RuntimeError(
                    "remove_embedded internal: the gathered region touches "
                    "a shared point; the gather mask under-reached.")
            surgery = (np.flatnonzero(victim), np.flatnonzero(drop), made,
                       placed)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    failures = comm.allgather(failure)
    real = [f for f in failures if f]
    if real:
        raise RuntimeError(
            f"remove_embedded failed on the surgery rank: {real[0]}")

    if comm.rank == target:
        victims_arr, drop_arr, made, placed = surgery
    else:
        victims_arr = np.empty(0, dtype=np.int64)
        drop_arr = np.empty(0, dtype=np.int64)
        made = []
        placed = np.empty((0, 2), dtype=float)

    new_dm, _pm, _pp = _rebuild_sewn(dm_work, drop_arr, victims_arr, made,
                                     placed)

    for name, value in removed_pairs:
        left = 0
        if new_dm.hasLabel(name):
            left = new_dm.getLabel(name).getStratumSize(int(value))
        left = int(comm.allreduce(left, op=MPI.SUM))
        if left:
            raise RuntimeError(
                f"{left} point(s) still carry ({name!r}, {value}) after "
                "the removal; the carve missed part of the object.")

    areas = cell_areas(new_dm)
    stats = np.array([float(areas.sum()) if len(areas) else 0.0,
                      float((areas <= 0.0).sum()) if len(areas) else 0.0,
                      float(len(victims_arr)), float(len(drop_arr)),
                      float(len(made))])
    comm.Allreduce(MPI.IN_PLACE, stats, op=MPI.SUM)
    if stats[1]:
        raise RuntimeError(
            f"{int(stats[1])} cell(s) of the result are inverted")
    if abs(stats[0] - area_before[0]) > 1e-9 * area_before[0]:
        raise RuntimeError(
            f"the removal changed the domain area: {area_before[0]:.12f} "
            f"-> {stats[0]:.12f}")

    after = _interface_facet_counts_excluding(new_dm, removed_pairs)
    breach = None
    for key, before in held_counts.items():
        if after.get(key, 0) != before:
            breach = (f"removing {label!r} would leave the surface "
                      f"{key[0]!r} with {after.get(key, 0)} facets instead "
                      f"of {before}.")
    breaches = comm.allgather(breach)
    real = [b for b in breaches if b]
    if real:
        raise RuntimeError(real[0])

    _validity_and_orientation_gates(new_dm, comm)

    info = {"n_removed_cells": int(stats[3]),
            "n_removed_vertices": int(stats[2]),
            "n_filled_cells": int(stats[4])}
    if verbose:
        uw.pprint(f"[remove_embedded {label!r}] removed "
                  f"{info['n_removed_cells']} cells, refilled with "
                  f"{info['n_filled_cells']}")
    return new_dm, info


def remove_embedded(dm, label, label_value=1, clearance=0.6, verbose=False):
    """Delete an embedded surface or zone; refill its region at background h.

    The inverse of the placements, and the other half of the lifecycle: a
    fault that has done its work is carved out — the labelled CELLS of a
    thin volume, or the region around the labelled FACES of a placed sheet —
    and the cavity is refilled with a plain gmsh fill at the surrounding
    mesh's own scale, so the region returns to background resolution. The
    object's labels vanish with their points (asserted, globally); every
    OTHER embedded surface must come through intact, gated exactly as a
    placement gates it. Nothing is redistributed: the same gather-first
    surgery as the placements, so the rest of the mesh never moves.

    A zone placed as ``(label, value)`` also removes its companion
    ``(label + "_skin", value)``.

    Parameters
    ----------
    dm : PETSc.DMPlex
        A 3-D simplex mesh, serial or distributed. **Not modified.**
    label, label_value : str, int
        The embedded object's label. Cells under it mean a thin volume;
        faces alone mean a placed sheet.
    clearance : float
        The collar around the object's skin also cleared, as a multiple of
        local ``h`` — what lets the fill erase the layer-scale grading.
    verbose : bool
        Report the counts.

    Returns
    -------
    cleared : PETSc.DMPlex
        A new mesh with the object gone and the region refilled.
    info : dict
        ``n_removed_cells``, ``n_removed_vertices``, ``n_filled_cells``,
        ``min_volume``.

    Raises
    ------
    ValueError
        If the label holds nothing, collectively.
    NotImplementedError
        In 2-D (arrives with the 2-D parallel work).
    RuntimeError
        Carve/fill refusals, always collective — including a cavity that
        would need cells held for a DIFFERENT embedded surface.
    """
    if dm.getDimension() == 2:
        return _remove_embedded_2d(dm, label, label_value, clearance,
                                   verbose)
    if dm.getDimension() != 3:
        raise NotImplementedError(
            f"remove_embedded takes a 2-D or 3-D simplex mesh; this mesh "
            f"is {dm.getDimension()}-D.")

    comm = uw.mpi.comm
    skin_label = label + "_skin"
    removed_pairs = ((label, int(label_value)),
                     (skin_label, int(label_value)))

    # The object's global geometry, and which cells are its own.
    soup_pts, soup_tris = _labelled_face_soup(dm, removed_pairs)
    n_soup = int(comm.allreduce(len(soup_tris), op=MPI.MAX))
    if n_soup == 0:
        raise ValueError(
            f"nothing is embedded under ({label!r}, {label_value}); "
            "there is nothing to remove.")

    vS, vE = dm.getDepthStratum(0)
    pStart, pEnd = dm.getChart()
    X = _coords(dm)[: vE - vS]
    cells = _tet_vertices(dm)
    h_vertex, _h_cell = _vertex_h_3d(dm, cells, len(X))
    d_skin = _sheet_distance(X, soup_pts, soup_tris)
    reach_v = clearance * h_vertex
    mark = np.zeros(pEnd - pStart, dtype=np.int32)
    mark[np.flatnonzero(d_skin < reach_v + 1.0 * h_vertex)
         + vS - pStart] = 1
    # The object's own vertices must gather with it, however fat the zone.
    cS0, _cE0 = dm.getHeightStratum(0)
    if dm.hasLabel(label):
        lab = dm.getLabel(label)
        if lab.getStratumSize(int(label_value)) > 0:
            for p in lab.getStratumIS(int(label_value)).getIndices():
                p = int(p)
                if cS0 <= p < _cE0:
                    for q in dm.getTransitiveClosure(p)[0]:
                        if vS <= int(q) < vE:
                            mark[int(q) - pStart] = 1

    volume_before = np.array([_owned_cell_volume(dm)], dtype=float)
    comm.Allreduce(MPI.IN_PLACE, volume_before, op=MPI.SUM)

    dm_work, moved = _gather_region(dm, mark, verbose=verbose)
    if moved:
        vS, vE = dm_work.getDepthStratum(0)
        pStart, pEnd = dm_work.getChart()
        X = _coords(dm_work)[: vE - vS]
        cells = _tet_vertices(dm_work)
        h_vertex, _h_cell = _vertex_h_3d(dm_work, cells, len(X))
        d_skin = _sheet_distance(X, soup_pts, soup_tris)
        reach_v = clearance * h_vertex

    on_wall = _true_wall_vertex_mask(dm_work, len(X))
    shared = _shared_point_flags(dm_work).astype(bool)

    cS, _cE = dm_work.getHeightStratum(0)
    interface = _interface_faces_3d(dm_work, exclude=removed_pairs)
    held_cells = set()
    for f in interface:
        for c in dm_work.getSupport(f):
            held_cells.add(int(c) - cS)
    held_counts = _interior_face_counts_3d(dm_work, exclude=removed_pairs)

    seed = np.zeros(len(cells), dtype=bool)
    if dm_work.hasLabel(label):
        lab = dm_work.getLabel(label)
        if lab.getStratumSize(int(label_value)) > 0:
            for p in lab.getStratumIS(int(label_value)).getIndices():
                p = int(p)
                if cS <= p < cS + len(cells):
                    seed[p - cS] = True

    from underworld3.utilities.edge_split import cell_diameters
    h_cell_local = cell_diameters(dm_work) if len(cells) else np.zeros(0)
    h_mean = np.array([float(h_cell_local.sum()), float(len(cells))])
    comm.Allreduce(MPI.IN_PLACE, h_mean, op=MPI.SUM)
    h = float(h_mean[0] / h_mean[1])
    # The centroid-drop rule exists to catch cells STRADDLING an object
    # about to be embedded; a removal embeds nothing, so the rule would
    # only widen the cavity for no purpose — measured: it pushed a sheet
    # removal at z = 0.3 into the domain wall that the sheet's own
    # placement cleared comfortably.
    reach_c = np.zeros(len(cells))

    n_region = int((d_skin < reach_v).sum() + seed.sum())
    owners = np.asarray(comm.allgather(n_region))
    target = int(np.argmax(owners))

    failure = None
    victims = drop_ids = None
    fill = shell_vert_ids = None
    if comm.rank == target:
        try:
            victims, drop_ids, shell, _cap = _carve_around_volume_3d(
                dm_work, X, cells, soup_pts, soup_tris, reach_v, reach_c,
                held_cells, on_wall, shared, seed_drop=seed)
            touched = set()
            for c in drop_ids:
                for q in dm_work.getTransitiveClosure(int(c) + cS)[0]:
                    touched.add(int(q))
            if any(shared[q - pStart] for q in touched):
                raise RuntimeError(
                    "remove_embedded internal: the gathered region touches "
                    "a shared point; the gather mask under-reached.")
            shell_vert_ids = sorted({v for _f, verts in shell
                                     for v in verts})
            local = {v: i for i, v in enumerate(shell_vert_ids)}
            shell_xyz = X[shell_vert_ids]
            shell_tris = np.array([[local[v] for v in verts]
                                   for _f, verts in shell], dtype=np.int64)
            fill = _gmsh_fill_plain_3d(shell_xyz, shell_tris, h)
            _p, _t, moved_nodes, _n = fill
            if moved_nodes:
                raise RuntimeError(
                    f"the removal fill moved {moved_nodes} constrained "
                    "node(s); the cavity cannot be sewn back.")
        # Exception, not just RuntimeError/ValueError: a raw gmsh error
        # (e.g. a PLC intersection) is a plain Exception, and an
        # uncaught raise on the surgery rank is a HANG for its peers —
        # every failure must become a collective refusal.
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    failures = comm.allgather(failure)
    real = [f for f in failures if f]
    if real:
        raise RuntimeError(
            f"remove_embedded failed on the surgery rank: {real[0]}")

    if comm.rank == target:
        fill_pts, fill_tets, _m, n_shell = fill
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

    new, point_map, _placed_new = _rebuild_sewn(
        dm_work, drop_arr, victims_arr, made, placed)

    # ------------------------------------------------------- global gates
    for name, value in removed_pairs:
        left = 0
        if new.hasLabel(name):
            left = new.getLabel(name).getStratumSize(int(value))
        left = int(comm.allreduce(left, op=MPI.SUM))
        if left:
            raise RuntimeError(
                f"{left} point(s) still carry ({name!r}, {value}) after the "
                "removal; the carve missed part of the object.")

    volume_after = np.array([_owned_cell_volume(new)], dtype=float)
    comm.Allreduce(MPI.IN_PLACE, volume_after, op=MPI.SUM)
    if abs(volume_after[0] - volume_before[0]) > 1e-9 * volume_before[0]:
        raise RuntimeError(
            f"the removal changed the domain volume: "
            f"{volume_before[0]:.12f} -> {volume_after[0]:.12f}")

    # The surgery must CONSERVE the domain's topology, not assume it: a
    # box is a ball (Euler 1) but a spherical shell is S^2 x I (Euler 2).
    owned = np.asarray(_owned_stratum_counts(dm), dtype=np.int64)
    comm.Allreduce(MPI.IN_PLACE, owned, op=MPI.SUM)
    euler_before = int(owned[0] - owned[1] + owned[2] - owned[3])
    owned = np.asarray(_owned_stratum_counts(new), dtype=np.int64)
    comm.Allreduce(MPI.IN_PLACE, owned, op=MPI.SUM)
    nv_g, ne_g, nf_g, nc_g = (int(x) for x in owned)
    if nv_g - ne_g + nf_g - nc_g != euler_before:
        raise RuntimeError(
            f"the removal changed the global Euler number: "
            f"{euler_before} -> {nv_g - ne_g + nf_g - nc_g}")

    after = _interior_face_counts_3d(new, exclude=removed_pairs)
    for key, before in held_counts.items():
        now = after.get(key, 0)
        if now != before:
            raise RuntimeError(
                f"removing {label!r} would leave the surface {key[0]!r} "
                f"with {now} interior faces instead of {before}.")

    _validity_and_orientation_gates(new, comm)

    min_vol = np.array([_owned_min_cell_volume(new)], dtype=float)
    comm.Allreduce(MPI.IN_PLACE, min_vol, op=MPI.MIN)

    counts = np.array([len(drop_arr), len(victims_arr), len(made)],
                      dtype=np.int64)
    comm.Allreduce(MPI.IN_PLACE, counts, op=MPI.SUM)
    info = {"n_removed_cells": int(counts[0]),
            "n_removed_vertices": int(counts[1]),
            "n_filled_cells": int(counts[2]),
            "min_volume": float(min_vol[0])}
    if verbose:
        uw.pprint(f"[remove_embedded {label!r}] removed "
                  f"{info['n_removed_cells']} cells / "
                  f"{info['n_removed_vertices']} vertices, refilled with "
                  f"{info['n_filled_cells']}")
    return new, info


ZONE_LABEL = "uw_zone"


def _place_thin_volume_2d(dm, polylines, width, label, label_value,
                          clearance, size, assembly, verbose, mesher=None,
                          seams="gather", ligament=None, grading=0.35):
    """The ribbon: the identical construction one dimension down.

    Serial AND parallel through the same gather-first mechanism as the 3-D
    volume: the assembly is meshed once (rank 0) and broadcast, the region
    gathers to one rank, the carve and the holes fill run there, every rank
    rebuilds collectively.

    The specify-long contract holds as it does in 3-D: polylines may run
    past the domain, the assembly is clipped to the box, and a clipped edge
    left in a wall line is the ribbon's OUTCROP BAND — boundary edges
    carrying both the skin label and the wall's labels. The 2-D cap is two
    straight wall segments (the 3-D wall annulus one dimension down), so the
    outcrop fill is one simply-connected polygon: the cavity ring with its
    wall span replaced by the skin's interior chain.
    """
    comm = uw.mpi.comm

    # The domain's own boundary, collectively — the clip target and the
    # complex the outcrop TRACE is identified against.
    dom_verts, dom_edges = _domain_boundary_facets(dm)
    domain_loops = [dom_verts[loop] for loop in
                    _skin_loops([(int(a), int(b)) for a, b in dom_edges],
                                what="the domain boundary")]

    failure = None
    payload = None
    if comm.rank == 0:
        try:
            if mesher == "ladder":
                asm_pts, asm_tris, cad_area = _occ_ladder_assembly_2d(
                    polylines, width, size, assembly, domain=domain_loops)
            elif mesher == "network":
                # fuse + embedded spines: each polyline's interior points
                # (the end points sit ON the caps) become mesh vertices;
                # junctions between touching ribbons are free
                spines = [np.asarray(P, dtype=float)[1:-1]
                          for P in polylines]
                asm_pts, asm_tris, cad_area = _occ_assembly_2d(
                    polylines, width, size, assembly, domain=domain_loops,
                    embed=spines)
            else:
                asm_pts, asm_tris, cad_area = _occ_assembly_2d(
                    polylines, width, size, assembly, domain=domain_loops)
            P = asm_pts[asm_tris]
            twice = ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                     - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
            mesh_area = float(np.abs(twice).sum() / 2.0)
            if abs(mesh_area - cad_area) > 1e-9 * cad_area:
                raise RuntimeError(
                    f"the ribbon assembly meshed to area {mesh_area:.12e} "
                    f"against CAD {cad_area:.12e}; the layer mesh does not "
                    "fill its own outlines.")
            asm_pts, asm_tris = _collapse_boundary_imprints(
                asm_pts, asm_tris, domain_loops, 0.1 * size)
            payload = (asm_pts, asm_tris)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    failures = comm.allgather(failure)
    real = [f for f in failures if f]
    if real:
        raise RuntimeError(f"place_thin_volume assembly failed: {real[0]}")
    asm_pts, asm_tris = comm.bcast(payload, root=0)

    _skin_xyz, skin_local, skin_node_ids = _assembly_skin(asm_pts, asm_tris)
    skin_edges = [(int(skin_node_ids[a]), int(skin_node_ids[b]))
                  for a, b in skin_local]
    loops_asm = _skin_loops(skin_edges)

    # The outcrop band: the skin's trace on the domain boundary. The
    # outcropping loop splits into band + interior chain; loops away from
    # the boundary stay holes of the fill.
    skin_edge_arr = np.asarray(skin_edges, dtype=np.int64)
    _int_idx, band_idx = _split_skin_trace(asm_pts, skin_edge_arr,
                                           dom_verts, dom_edges)
    outcropping = bool(len(band_idx))
    chain_asm = None
    hole_loops = loops_asm
    band_pairs = set()
    if outcropping:
        _refuse_multiple_bands(skin_edge_arr[band_idx])
        band_pairs = {frozenset((int(a), int(b)))
                      for a, b in skin_edge_arr[band_idx]}
        chain_asm, hole_loops = _outcrop_chain_2d(loops_asm, band_pairs)

    vS, vE = dm.getDepthStratum(0)
    pStart, pEnd = dm.getChart()
    X = _coords(dm)[: vE - vS]
    cells = _cells_anticlockwise(dm, X)
    h_vertex, _hc = _vertex_h_3d(dm, cells, len(X))
    d_skin = _segments_distance(X, asm_pts, skin_edges)
    reach_v = np.maximum(clearance * h_vertex, 0.6 * width)

    area_before = np.array([float(cell_areas(dm).sum())
                            if len(cells) else 0.0])
    comm.Allreduce(MPI.IN_PLACE, area_before, op=MPI.SUM)

    mark = np.zeros(pEnd - pStart, dtype=np.int32)
    mark[np.flatnonzero(d_skin < reach_v + 1.0 * h_vertex)
         + vS - pStart] = 1
    if seams in ("ligament", "conform"):
        dm_work, moved = dm, 0          # no gather: each rank carves its own
    else:
        dm_work, moved = _gather_region(dm, mark, verbose=verbose)
    if moved:
        vS, vE = dm_work.getDepthStratum(0)
        pStart, pEnd = dm_work.getChart()
        X = _coords(dm_work)[: vE - vS]
        cells = _cells_anticlockwise(dm_work, X)
        h_vertex, _hc = _vertex_h_3d(dm_work, cells, len(X))
        d_skin = _segments_distance(X, asm_pts, skin_edges)
        reach_v = np.maximum(clearance * h_vertex, 0.6 * width)

    shared = _shared_point_flags(dm_work).astype(bool)
    on_wall = _true_wall_vertex_mask(dm_work, len(X))
    held_v, held_c = _interface_vertices_and_cells(dm_work, len(X),
                                                   len(cells))
    held_counts = _interface_facet_counts(dm_work)

    n_region = int((d_skin < reach_v).sum())
    owners = np.asarray(comm.allgather(n_region))
    if owners.sum() == 0:
        raise ValueError("the thin volume meets no cell of this mesh")
    target = int(np.argmax(owners))
    # Who carves: the gather's one target, or — with seam ligaments — every
    # rank holding part of the zone, concurrently, each its own cavities.
    do_surgery = (n_region > 0) if seams == "ligament" \
        else True if seams == "conform" else (comm.rank == target)

    # The seam ligament (#670): the band butts up to the partition seam in
    # elements and nothing the surgery makes crosses it. The rebuild's one
    # contract is that no shared vertex is deleted and no placed vertex is
    # shared, so the shared vertices (and, with ``ligament``, those within
    # ligament/2 of the seam) are protected and everything else at the
    # seam is carved like any other cell: the cavity ring runs along the
    # seam's own edges and the fill attaches to them. The assembly is
    # clipped clear of the seam by a fraction of a band cell, and the fill
    # strip left at the seam is the LIGAMENT — labelled as zone, for the
    # weak rheology to bridge.
    seam_prot_v = np.zeros(len(X), dtype=bool)
    shared_v = shared[vS - pStart: vE - pStart]
    if seams == "ligament" and comm.size > 1:
        seam_prot_v = shared_v.copy()
        if ligament is not None and shared_v.any():
            from scipy.spatial import cKDTree as _KDT_seam
            seam_prot_v |= (_KDT_seam(X[shared_v]).query(X)[0]
                            < 0.5 * float(ligament))
    # The seam-conforming placement (#670): the band is meshed THROUGH the
    # seam. Nothing on the seam is protected — a seam vertex within the
    # band's reach is deleted on both ranks (victims are synchronised
    # below) — and every band cell is made by the rank whose cavity holds
    # its centroid, so the boundary between the two ranks' band cells is a
    # chain of band edges: the new seam inside the band. Each rank's fill
    # runs against its own ring with the crossed seam span replaced by a
    # connector to the band's skin, the skin arc on its side, and a
    # connector back — the outcrop splice with the seam as the wall. The
    # band vertices both sides use become shared points of the rebuild.
    conform = seams == "conform" and comm.size > 1
    seam_edge_set = set()
    if conform:
        # the seam survives wherever the band does not reach it: only a
        # seam vertex within the band's own margin is deleted (on both
        # sides), never one the carve's clearance would have taken, so a
        # seam that runs beside the band keeps its edges and the fills on
        # either side keep their common boundary
        seam_prot_v = shared_v & ~(d_skin < 0.6 * width)
        eS0, eE0 = dm_work.getDepthStratum(1)
        for e in range(eS0, eE0):
            if shared[e - pStart]:
                a, b = (int(q) - vS for q in dm_work.getCone(e))
                seam_edge_set.add((min(a, b), max(a, b)))
    comp_of_tri = _assembly_components(asm_tris)
    loop_comp = [int(comp_of_tri[np.flatnonzero(
        (asm_tris == int(loop[0])).any(axis=1))[0]]) for loop in loops_asm]

    failure = None
    surgery = None
    victim = np.zeros(len(X), dtype=bool)
    drop = np.zeros(len(cells), dtype=bool)
    drop_wanted = drop.copy()
    deletable_wall = np.zeros(len(X), dtype=bool)
    near_touched = None
    if do_surgery:
        try:
            beside_held = np.zeros(len(X), dtype=bool)
            beside_held[cells[held_c].ravel()] = True
            if outcropping:
                comp_loops = [_compress_collinear_loop(L)
                              for L in domain_loops]
                deletable_wall, near_touched = _outcrop_frame_2d(
                    comp_loops, asm_pts, band_pairs, X, on_wall)
            protected = ((on_wall & ~deletable_wall) | held_v | beside_held
                         | seam_prot_v)
            victim = (d_skin < reach_v) & ~protected

            drop = victim[cells].any(axis=1)
            cen = X[cells].mean(axis=1)
            reach_c = np.maximum(clearance * h_vertex[cells].min(axis=1),
                                 0.6 * width)
            drop |= _segments_distance(cen, asm_pts, skin_edges) < reach_c
            drop_wanted = drop.copy()
            drop &= ~held_c
            need = victim[cells].any(axis=1)
            if (need & held_c).any():
                raise RuntimeError(
                    "the ribbon's cavity needs a cell that belongs to a "
                    "surface already embedded. Zones and surfaces must be "
                    "separated by at least a cell.")
            drop |= need
            if not drop.any() and seams == "gather":
                raise ValueError("the thin volume meets no cell of this mesh")
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    failures = comm.allgather(failure)
    real = [f for f in failures if f]
    if real:
        raise RuntimeError(
            f"place_thin_volume failed on the surgery rank: {real[0]}")

    # Seam-conforming: the victims on the seam are one decision on both
    # sides (their reach is SF-reconciled already; this makes it a gate
    # rather than an assumption), and every band cell — and every band
    # vertex, for the arc choice — has ONE owner, the rank whose dropped
    # cells hold it. COLLECTIVE.
    tri_owner = np.full(len(asm_tris), comm.rank, dtype=np.int32)
    vtx_owner = np.full(len(asm_pts), comm.rank, dtype=np.int32)
    cen_tri = asm_pts[asm_tris].mean(axis=1)
    if conform:
        flags = np.zeros(pEnd - pStart, dtype=np.int32)
        flags[vS - pStart + np.flatnonzero(victim)] = 1
        flags = _propagate_vertex(dm_work, flags, MPI.MAX, np.maximum)
        victim = flags[vS - pStart: vE - pStart].astype(bool)
        drop |= victim[cells].any(axis=1)
        drop &= ~held_c
        mine_c = _inside_mesh(X, cells[drop], cen_tri) if drop.any() \
            else np.zeros(len(cen_tri), dtype=bool)
        mine_v = _inside_mesh(X, cells[drop], asm_pts) if drop.any() \
            else np.zeros(len(asm_pts), dtype=bool)
        tri_owner = np.where(mine_c, comm.rank, comm.size).astype(np.int32)
        vtx_owner = np.where(mine_v, comm.rank, comm.size).astype(np.int32)
        # a skin edge bounds the fill of the rank whose cavity holds the
        # point just outside it (outside = away from its band cell)
        skin_arr = np.asarray(skin_edges, dtype=np.int64)
        tri_of_edge = {}
        for t, tri in enumerate(asm_tris):
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                tri_of_edge[(min(int(a), int(b)), max(int(a), int(b)))] = t
        outward = np.empty((len(skin_arr), 2))
        for k, (a, b) in enumerate(skin_arr):
            tri = asm_tris[tri_of_edge[(min(int(a), int(b)),
                                        max(int(a), int(b)))]]
            third = [int(v) for v in tri if int(v) not in (int(a), int(b))][0]
            mid = 0.5 * (asm_pts[a] + asm_pts[b])
            nrm = mid - asm_pts[third]
            nrm /= np.linalg.norm(nrm) + 1e-300
            outward[k] = mid + 0.05 * size * nrm
        mine_e = _inside_mesh(X, cells[drop], outward) if drop.any() \
            else np.zeros(len(skin_arr), dtype=bool)
        edge_owner = np.where(mine_e, comm.rank, comm.size).astype(np.int32)
        comm.Allreduce(MPI.IN_PLACE, tri_owner, op=MPI.MIN)
        comm.Allreduce(MPI.IN_PLACE, vtx_owner, op=MPI.MIN)
        comm.Allreduce(MPI.IN_PLACE, edge_owner, op=MPI.MIN)
        if (edge_owner >= comm.size).any():
            raise RuntimeError(
                f"place_thin_volume: {int((edge_owner >= comm.size).sum())} "
                "skin edge(s) face no rank's cavity; the reach did not cover "
                "the band across the seam.")
        if (tri_owner >= comm.size).any():
            raise RuntimeError(
                f"place_thin_volume: {int((tri_owner >= comm.size).sum())} "
                "assembly cell(s) lie in no rank's cavity; the reach did not "
                "cover the band across the seam.")

    failure = None
    if do_surgery:
        try:
            crossings = []
            comps = []
            if drop.any():
                # Several cavities on one rank are ordinary (a network
                # whose zones sit apart, a graded base whose cleared cells
                # fall into more than one hole): each is carved and
                # filled on its own. This lifts the "one simple hole"
                # refusal the fine S-fault rig hit on a graded base.
                comps, drop = _ring_growing_multi(cells, drop, held_c,
                                                  allow_multiple=True, X=X)
            boundary_pairs = None
            if outcropping:
                boundary_pairs = {frozenset((a, b))
                                  for _e, a, b in _boundary_edges(dm_work)}

            # Per cavity: the assembly cells it holds (all of them under
            # the gather; under ligaments, those wholly inside it and
            # clear of the seam-side ring by most of a band cell — the
            # clipped edge must leave the fill room), that subset's skin,
            # its outcrop chain if the band reaches the wall there, and
            # the ring spliced accordingly.
            carved = []
            if conform:
                # cavities with nothing of the band and no seam are left
                # alone; the rest are one fill-boundary graph
                rings_c = []
                for ring, comp in comps:
                    has_seam = any((min(ring[i], ring[(i + 1) % len(ring)]),
                                    max(ring[i], ring[(i + 1) % len(ring)]))
                                   in seam_edge_set for i in range(len(ring)))
                    inside = _inside_mesh(X, cells[comp], cen_tri)
                    if not has_seam and not (inside
                                             & (tri_owner == comm.rank)).any():
                        drop[comp] = False
                        continue
                    rings_c.append(ring)
                mine_skin = [(int(a), int(b)) for (a, b), o
                             in zip(skin_edges, edge_owner) if o == comm.rank]
                fills = _conform_fill_loops_2d(
                    rings_c, X, victim, seam_edge_set, asm_pts, loops_asm,
                    mine_skin, len(X)) if rings_c else []
                kept_c = tri_owner == comm.rank
                crossings = [None] * sum(
                    1 for r in rings_c for i in range(len(r))
                    if (min(r[i], r[(i + 1) % len(r)]),
                        max(r[i], r[(i + 1) % len(r)])) in seam_edge_set)
                for outer, holes in fills:
                    holes_k = [[int(v) - len(X) for v in h] for h in holes]
                    old_ring = np.asarray([v for v in outer if v < len(X)])
                    if len(old_ring) and victim[old_ring].any():
                        raise RuntimeError(
                            "a deleted vertex is on the cavity boundary")
                    if len(old_ring) and on_wall[old_ring].any():
                        raise RuntimeError(
                            "the ribbon's cavity reached the domain wall; "
                            "the volume must be interior, with clearance "
                            "to spare")
                    carved.append((outer, kept_c, [], holes_k, None, []))
                comps = []                  # handled
            for ring, comp in comps:
                # the clip margin applies along every ring edge on the
                # seam (both ends protected) and every one whose outside
                # cell the cavity WANTED but a shrink restored
                edge_cell = {}
                for ci in np.flatnonzero(~drop & drop_wanted):
                    v0, v1, v2 = (int(v) for v in cells[ci])
                    for a, b in ((v0, v1), (v1, v2), (v2, v0)):
                        edge_cell[(min(a, b), max(a, b))] = ci
                seam_edges = [(a, b) for a, b in zip(ring, ring[1:] + ring[:1])
                              if (min(a, b), max(a, b)) in edge_cell
                              or (seam_prot_v[a] and seam_prot_v[b])]
                # the assembly cells THIS cavity holds. One cavity with no
                # seam holds the whole assembly (an outcrop vertex snapped
                # to a curved wall sits just outside the straight base
                # cells, so a point test would lose it); several cavities,
                # or a seam, take the cells whose vertices lie inside.
                if len(comps) == 1 and not seam_edges:
                    kept = np.ones(len(asm_tris), dtype=bool)
                elif not seam_edges:
                    # several cavities, no seam: a band cell belongs to the
                    # cavity holding its CENTROID (a vertex can sit on a
                    # ring edge to roundoff and fail a point test; a
                    # centroid cannot, and the reach covers the band)
                    kept = _inside_mesh(X, cells[comp], cen_tri)
                else:
                    inside = _inside_mesh(X, cells[comp], asm_pts)
                    kept = inside[asm_tris].all(axis=1)
                if seam_edges and kept.any():
                    d_seam = _segments_distance(asm_pts, X, seam_edges)
                    kept &= ~(d_seam[asm_tris] < 0.4 * size).any(axis=1)
                if not kept.all():
                    kept = _manifold_subset_2d(asm_tris, kept)
                if not kept.any():
                    drop[comp] = False      # nothing to embed: leave it
                    continue
                sub = asm_tris[kept]
                _sk, sk_local, sk_ids = _assembly_skin(asm_pts, sub)
                sk_edges = [(int(sk_ids[a]), int(sk_ids[b]))
                            for a, b in sk_local]
                loops_k = _skin_loops(sk_edges)
                chain_k, holes_k = None, loops_k
                removed_wall = []
                if outcropping:
                    pairs_k = {frozenset(e) for e in sk_edges} & band_pairs
                    if pairs_k and len(pairs_k) != len(band_pairs):
                        raise RuntimeError(
                            "a seam ligament crosses the outcrop band; the "
                            "outcrop must lie within one rank's cavity. Use "
                            "seams='gather' for this layout.")
                    if pairs_k:
                        chain_k, holes_k = _outcrop_chain_2d(loops_k,
                                                             band_pairs)
                if chain_k is None:
                    if on_wall[np.asarray([v for v in ring
                                           if v < len(X)])].any():
                        raise RuntimeError(
                            "the ribbon's cavity reached the domain wall; "
                            "the volume must be interior, with clearance "
                            "to spare")
                else:
                    chain_ids = [len(X) + int(v) for v in chain_k]
                    chain_ends = (asm_pts[chain_k[0]], asm_pts[chain_k[-1]])
                    ring, removed_wall = _outcrop_ring_splice(
                        ring, X, boundary_pairs, chain_ids, chain_ends)
                    off_band = [v for v in ring if v < len(X)
                                and on_wall[v] and not near_touched[v]]
                    if off_band:
                        raise RuntimeError(
                            "the ribbon's cavity reached the domain "
                            "boundary away from the outcrop band; only a "
                            "one-band outcrop is built.")
                old_ring = np.asarray([v for v in ring if v < len(X)])
                if victim[old_ring].any():
                    raise RuntimeError(
                        "a deleted vertex is on the cavity boundary")
                carved.append((ring, kept, sk_edges, holes_k, chain_k,
                               removed_wall))

            referenced = np.zeros(len(X), dtype=bool)
            if (~drop).any():
                referenced[cells[~drop].ravel()] = True
            # a ring vertex is referenced by the fill; on the seam it may
            # have no kept cell of this rank at all (its fan is one-sided
            # and wholly dropped), and it must survive
            for ring, _k, _sk, _h, _c, _w in carved:
                referenced[[v for v in ring if v < len(X)]] = True
            # a cavity left uncarved (nothing of the band inside it) keeps
            # its cells, so its would-be victims stay
            victim &= ~referenced
            orphan = ~referenced & ~victim
            if (orphan & on_wall & ~deletable_wall).any():
                raise RuntimeError(
                    "the cavity would strand a domain-wall vertex; the "
                    "volume must be interior, with clearance to spare")
            victim |= orphan

            # Labels the deleted wall span carried, read PER EDGE before
            # the rebuild forgets them — a span across a domain corner
            # changes label mid-way (Top on one side, Right on the other),
            # so each new wall edge must take the labels of the old
            # segment it lies on, not a set common to the whole span.
            outcrop = None
            for ring, kept, sk_edges, holes_k, chain_k, removed_wall in carved:
                if not removed_wall:
                    continue
                span_labels = []
                span_segments = []
                names = [dm_work.getLabelName(i)
                         for i in range(dm_work.getNumLabels())
                         if dm_work.getLabelName(i)
                         not in reconnect._TOPOLOGY_LABELS]
                for a, b in removed_wall:
                    p = int(dm_work.getFullJoin([int(a) + vS,
                                                 int(b) + vS])[0])
                    pairs_p = []
                    for name in names:
                        val = dm_work.getLabel(name).getValue(p)
                        if val >= 0:
                            pairs_p.append((name, int(val)))
                    span_labels.append(pairs_p)
                    span_segments.append((X[int(a)].copy(),
                                          X[int(b)].copy()))
                # The splice ends: the surviving corner vertices and the
                # chain ends they meet — the chain is the ring's tail, in
                # the orientation the splice chose.
                outcrop = (span_labels, span_segments,
                           int(removed_wall[0][0]),
                           int(removed_wall[-1][1]),
                           int(ring[-len(chain_k)]) - len(X),
                           int(ring[-1]) - len(X))

            kept_all = np.zeros(len(asm_tris), dtype=bool)
            ring_band = []
            for ring, kept, _sk, holes_k, _c, _w in carved:
                kept_all |= kept
                ring_band += [int(v) - len(X) for v in ring if v >= len(X)]
                ring_band += [int(v) for h in holes_k for v in h]
            used = np.unique(np.concatenate([
                asm_tris[kept_all].ravel() if kept_all.any()
                else np.zeros(0, dtype=np.int64),
                np.asarray(ring_band, dtype=np.int64)]))
            asm_row = -np.ones(len(asm_pts), dtype=np.int64)
            asm_row[used] = np.arange(len(used))
            placed_parts = [asm_pts[used]]
            n_rows = len(used)
            made = []
            Xall = np.vstack([X, asm_pts])
            for ring, kept, sk_edges, holes_k, chain_k, _w in carved:
                holes = [[len(X) + int(v) for v in loop] for loop in holes_k]
                # GRADED fill: the annulus between the assembly's skin and
                # the cavity ring is meshed from the skin's own size out to
                # the ring's edge length, interpolated by relative distance.
                # Without this the fill inherits the skin segmentation
                # throughout, and where several ribbons sit within a cavity
                # of each other the merged cavity fills at band resolution
                # end to end (measured on the S-fault rig: the fill cost as
                # many cells as the bands — the #629 "fill shell was the
                # fat" finding on the network path).
                size_of = None
                ring_arr = np.asarray(ring, dtype=int)
                if conform and (ring_arr >= len(X)).any():
                    # the crossing band sits in the OUTER ring, not in a
                    # hole: grade from every band vertex (holes and the
                    # ring's band arcs) out to the ring's BASE vertices
                    band_ids = np.unique(np.concatenate(
                        [ring_arr[ring_arr >= len(X)] - len(X)]
                        + [np.asarray(l, dtype=int) for l in holes_k]))
                    holes_for_size = [band_ids]
                    ring_for_size = ring_arr[ring_arr < len(X)]
                else:
                    holes_for_size = holes_k
                    ring_for_size = ring_arr
                if holes_for_size:
                    from scipy.spatial import cKDTree as _KDT
                    _skin = asm_pts[np.unique(np.concatenate(
                        [np.asarray(l, dtype=int) for l in holes_for_size]))]
                    _ring_pts = Xall[ring_for_size]
                    _rl = np.linalg.norm(np.diff(
                        np.vstack([_ring_pts, _ring_pts[:1]]), axis=0),
                        axis=1)
                    _h_ring, _h_skin = float(np.median(_rl)), float(size)
                    if _h_ring > 1.2 * _h_skin:
                        _kd_s, _kd_r = _KDT(_skin), _KDT(_ring_pts)

                        def size_of(x, y, _s=_h_skin, _h=_h_ring,
                                    _ks=_kd_s, _kr=_kd_r, _p=float(grading)):
                            q = np.array([[x, y]])
                            ds = float(_ks.query(q)[0][0])
                            dr = float(_kr.query(q)[0][0])
                            # relative distance across the fill, raised to
                            # the grading power: 1 is linear, below 1 coarsens
                            # faster leaving the band
                            t = ds / (ds + dr + 1e-30)
                            return _s + (_h - _s) * t ** _p
                try:
                    gap_tris, extra = _gmsh_fill_2d(Xall, ring, None,
                                                    holes=holes,
                                                    size_of=size_of)
                except Exception as exc:
                    # the fill's inputs, for inspection: a refused fill
                    # is a geometry question, and the ring is the answer
                    dump = f"place_fill_failure_rank{comm.rank}.npz"
                    np.savez(dump, Xall=Xall, ring=np.asarray(ring),
                             holes=np.asarray(
                                 [len(h) for h in holes] + sum(
                                     ([int(v) for v in h] for h in holes),
                                     [])),
                             seams=str(seams))
                    raise RuntimeError(
                        f"{type(exc).__name__}: {exc} [fill inputs saved "
                        f"to {dump}]") from exc
                n_all = len(Xall)
                offset = n_rows

                def mixed(v, _off=offset, _n=n_all):
                    v = int(v)
                    if v < len(X):
                        return v
                    if v < _n:
                        return -(int(asm_row[v - len(X)]) + 1)
                    return -(_off + (v - _n) + 1)

                made += [tuple(mixed(v) for v in t) for t in gap_tris]
                if len(extra):
                    placed_parts.append(np.asarray(extra, dtype=float)
                                        .reshape(-1, 2))
                    n_rows += len(extra)
            made += [tuple(-(int(asm_row[v]) + 1) for v in t)
                     for t in asm_tris[kept_all]]
            placed = np.vstack(placed_parts) if n_rows else \
                np.empty((0, 2), dtype=float)

            # The rebuild's contract, gated here rather than assumed: no
            # shared vertex is deleted (the gather's mask reached, or the
            # seam's vertices were protected).
            if not conform and any(shared[int(v) + vS - pStart]
                                   for v in np.flatnonzero(victim)):
                raise RuntimeError(
                    "place_thin_volume internal: the surgery would delete a "
                    "shared vertex; the gather mask under-reached.")
            sk_all = [e for _r, _k, sk_edges, _h, _c, _w in carved
                      for e in sk_edges]
            if conform:
                # every skin edge both of whose vertices this rank places is
                # an edge of this rank's mesh (a band cell's, or the arc's)
                used_set = set(int(v) for v in used)
                sk_all = [(a, b) for a, b in skin_edges
                          if a in used_set and b in used_set]
            surgery = (np.flatnonzero(victim), np.flatnonzero(drop), made,
                       placed, outcrop, kept_all, asm_row, sk_all,
                       len(crossings))
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    failures = comm.allgather(failure)
    real = [f for f in failures if f]
    if real:
        raise RuntimeError(
            f"place_thin_volume failed on the surgery rank: {real[0]}")

    n_cross_local = 0
    if surgery is not None:
        (victims_arr, drop_arr, made, placed, outcrop, kept_all, asm_row,
         sk_all, n_cross_local) = surgery
    else:
        victims_arr = np.empty(0, dtype=np.int64)
        drop_arr = np.empty(0, dtype=np.int64)
        made = []
        placed = np.empty((0, 2), dtype=float)
        outcrop = None
        kept_all = np.zeros(len(asm_tris), dtype=bool)
        asm_row = -np.ones(len(asm_pts), dtype=np.int64)
        sk_all = []

    # Every assembly cell is embedded on exactly one rank or, under the
    # ligament, on none — the ligament cells; a cell kept twice is a
    # defect of the clipping and is refused here.
    kept_count = kept_all.astype(np.int32)
    comm.Allreduce(MPI.IN_PLACE, kept_count, op=MPI.SUM)
    if (kept_count > 1).any():
        raise RuntimeError(
            "place_thin_volume internal: an assembly cell was embedded on "
            f"{int(kept_count.max())} ranks; the seam clipping overlapped.")
    n_kept = int((kept_count == 1).sum())
    lig_tris = np.flatnonzero(kept_count == 0)
    if len(lig_tris) and seams != "ligament":
        raise RuntimeError(
            f"place_thin_volume internal: {len(lig_tris)} assembly cells "
            "were not embedded.")

    # Seam-conforming: the band vertices more than one rank places are
    # shared points, owned by the lowest rank; the others become leaves of
    # the rebuild's vertex star-forest. COLLECTIVE.
    shared_rows = []
    if conform:
        mine = {int(v): int(asm_row[v]) for v in np.flatnonzero(asm_row >= 0)}
        every = comm.allgather(mine)
        for vid, row in mine.items():
            holders = [r for r, d in enumerate(every) if vid in d]
            if len(holders) > 1 and holders[0] != comm.rank:
                shared_rows.append((row, holders[0], every[holders[0]][vid]))
    n_skin_expected = len(skin_edges) if conform \
        else int(comm.allreduce(len(sk_all), op=MPI.SUM))

    new_dm, point_map, placed_points = _rebuild_sewn(
        dm_work, drop_arr, victims_arr, made, placed, shared_rows=shared_rows)

    skin_label = label + "_skin"
    trace_label = label + "_trace"
    lig_label = label + "_ligament"
    for name in (label, skin_label, trace_label):
        if not new_dm.hasLabel(name):
            new_dm.createLabel(name)
    n_zone_local = 0
    n_skin_local = 0
    if surgery is not None:
        out_label = new_dm.getLabel(label)
        out_skin = new_dm.getLabel(skin_label)
        for t in asm_tris[kept_all]:
            joined = new_dm.getFullJoin(
                [int(placed_points[int(asm_row[int(v)])]) for v in t])
            if len(joined) != 1:
                failure = ("an assembly cell is not a cell of the sewn "
                           "mesh; the embed lost the layer.")
                break
            out_label.setValue(int(joined[0]), int(label_value))
            n_zone_local += 1
        else:
            leaf = set()
            if conform:
                try:
                    _n, il, _ir = new_dm.getPointSF().getGraph()
                    leaf = set(int(q) for q in il) if il is not None else set()
                except (ValueError, TypeError):
                    leaf = set()
            for a, b in sk_all:
                joined = new_dm.getFullJoin(
                    [int(placed_points[int(asm_row[a])]),
                     int(placed_points[int(asm_row[b])])])
                if len(joined) != 1:
                    failure = ("a skin edge is not an edge of the sewn "
                               "mesh; the gap was not sewn onto the layer.")
                    break
                out_skin.setValue(int(joined[0]), int(label_value))
                if int(joined[0]) not in leaf:
                    n_skin_local += 1
    failures = comm.allgather(failure)
    real = [f for f in failures if f]
    if real:
        raise RuntimeError(real[0])

    # The ligament cells: base cells the clipped-away assembly cells
    # cover (centroid inside one). Zone AND ligament labelled, so the band
    # mask sees them and the split's edge labelling can avoid them.
    n_lig_local = 0
    if len(lig_tris):
        if not new_dm.hasLabel(lig_label):
            new_dm.createLabel(lig_label)
        out_lig = new_dm.getLabel(lig_label)
        out_label = new_dm.getLabel(label)
        cS_n, cE_n = new_dm.getHeightStratum(0)
        ids, cen = _cell_centroids_of(new_dm, np.ones(cE_n - cS_n, dtype=bool))
        covered = _inside_mesh(asm_pts, asm_tris[lig_tris], cen)
        for c in ids[covered]:
            out_label.setValue(int(c) + cS_n, int(label_value))
            out_lig.setValue(int(c) + cS_n, int(label_value))
            n_lig_local += 1

    # An outcropping ribbon's new wall coverage — the two splice segments
    # and the band itself — is relabelled EXPLICITLY with what the deleted
    # wall span carried, full closures included.
    n_wall_local = 0
    if outcropping:
        has_outcrop = np.asarray(comm.allgather(outcrop is not None))
        if int(has_outcrop.sum()) != 1:
            raise RuntimeError(
                f"place_thin_volume internal: the outcrop was carved on "
                f"{int(has_outcrop.sum())} ranks.")
        root = int(np.argmax(has_outcrop))
        label_names = comm.bcast(
            sorted({name for pairs_p in outcrop[0] for name, _v in pairs_p})
            if comm.rank == root else None, root=root)
        for name in label_names:
            if not new_dm.hasLabel(name):
                new_dm.createLabel(name)
        if comm.rank == root:
            (span_lab, span_seg, corner_l, corner_r,
             a_first, a_last) = outcrop
            wall_ids = [[int(point_map[corner_l + vS - pStart]),
                         int(placed_points[int(asm_row[a_first])])],
                        [int(placed_points[int(asm_row[a_last])]),
                         int(point_map[corner_r + vS - pStart])]]
            wall_ids += [[int(placed_points[int(asm_row[a])]),
                          int(placed_points[int(asm_row[b])])]
                         for a, b in (tuple(p) for p in band_pairs)]
            mids = [0.5 * (X[corner_l] + asm_pts[a_first]),
                    0.5 * (asm_pts[a_last] + X[corner_r])]
            mids += [0.5 * (asm_pts[a] + asm_pts[b])
                     for a, b in (tuple(p) for p in band_pairs)]

            def span_labels_at(m):
                best, at = np.inf, 0
                for k2, (A, B) in enumerate(span_seg):
                    e = B - A
                    u = np.clip(float((m - A) @ e) / float(e @ e),
                                0.0, 1.0)
                    d = float(np.linalg.norm(m - (A + u * e)))
                    if d < best:
                        best, at = d, k2
                return span_lab[at]

            # The first two entries are the splice segments (the 2-D cap);
            # the rest are the band — the TRACE, the intersection itself,
            # labelled as such so the model can form whatever unions it
            # needs (never a partition of the wall into named halves).
            # Each edge takes the labels of the removed segment it lies
            # on, so a band across a domain corner restores Top on one
            # side and Right on the other.
            out_trace = new_dm.getLabel(trace_label)
            for k, ids in enumerate(wall_ids):
                joined = new_dm.getFullJoin(ids)
                if len(joined) != 1:
                    failure = ("an outcrop wall edge is not an edge of the "
                               "sewn mesh; the splice or band was not "
                               "sewn.")
                    break
                for q in new_dm.getTransitiveClosure(int(joined[0]))[0]:
                    for name, val in span_labels_at(mids[k]):
                        new_dm.getLabel(name).setValue(int(q), int(val))
                    if k >= 2:
                        out_trace.setValue(int(q), int(label_value))
                n_wall_local += 1
        failures = comm.allgather(failure)
        real = [f for f in failures if f]
        if real:
            raise RuntimeError(real[0])
        n_wall = np.array([n_wall_local], dtype=np.int64)
        comm.Allreduce(MPI.IN_PLACE, n_wall, op=MPI.SUM)
        if int(n_wall[0]) != 2 + len(band_pairs):
            raise RuntimeError(
                f"{int(n_wall[0])} outcrop wall edges relabelled for "
                f"{2 + len(band_pairs)} given.")

    counts = np.array([n_zone_local, n_skin_local, len(victims_arr),
                       len(placed), n_lig_local], dtype=np.int64)
    comm.Allreduce(MPI.IN_PLACE, counts, op=MPI.SUM)
    n_zone, n_skin, n_removed, n_placed, n_lig = (int(x) for x in counts)
    if n_zone != n_kept:
        raise RuntimeError(f"{n_zone} zone cells labelled for "
                           f"{n_kept} assembly cells embedded.")
    if n_skin != n_skin_expected:
        raise RuntimeError(f"{n_skin} skin edges labelled for "
                           f"{n_skin_expected} given.")

    areas = cell_areas(new_dm)
    stats = np.array([float(areas.sum()) if len(areas) else 0.0,
                      float((areas <= 0.0).sum()) if len(areas) else 0.0])
    comm.Allreduce(MPI.IN_PLACE, stats, op=MPI.SUM)
    if stats[1]:
        raise RuntimeError(
            f"{int(stats[1])} cell(s) of the result are inverted")
    if abs(stats[0] - area_before[0]) > 1e-9 * area_before[0]:
        raise RuntimeError(
            f"the placement changed the domain area: "
            f"{area_before[0]:.12f} -> {stats[0]:.12f}")

    after = _interface_facet_counts(new_dm)
    breach = None
    for key, before in held_counts.items():
        now = after.get(key, 0)
        if now < before or (now != before
                            and key != (skin_label, int(label_value))):
            breach = (f"placing {label!r} would leave the surface "
                      f"{key[0]!r} with {now} facets instead of {before}.")
    breaches = comm.allgather(breach)
    real = [b for b in breaches if b]
    if real:
        raise RuntimeError(real[0])

    _validity_and_orientation_gates(new_dm, comm)

    mins = np.array([float(areas.min()) if len(areas) else np.inf,
                     float(min_angles(new_dm).min())
                     if len(areas) else np.inf])
    comm.Allreduce(MPI.IN_PLACE, mins, op=MPI.MIN)
    n_ranks_carving = int(comm.allreduce(int(surgery is not None),
                                         op=MPI.SUM))
    info = {"n_zone_cells": n_zone, "n_skin_faces": n_skin,
            "n_placed": n_placed, "n_removed": n_removed,
            "n_trace_facets": int(len(band_idx)),
            "min_area": float(mins[0]), "min_angle": float(mins[1]),
            "n_gathered": int(moved), "seams": seams,
            "n_ligament_cells": n_lig, "n_ligament_tris": int(len(lig_tris)),
            "n_surgery_ranks": n_ranks_carving,
            "n_seam_crossings": int(comm.allreduce(n_cross_local, op=MPI.SUM))}
    if verbose:
        uw.pprint(f"[place_thin_volume {label!r}] {n_zone} zone cells, "
                  f"{n_skin} skin edges; placed {n_placed} vertices, "
                  f"removed {n_removed}; min angle "
                  f"{info['min_angle']:.2f} deg"
                  + (f"; {n_lig} ligament cells stand in for "
                     f"{len(lig_tris)} clipped band cells"
                     if len(lig_tris) else ""))
    return new_dm, info



def place_thin_volume(dm, patches, width, label=ZONE_LABEL, label_value=1,
                      clearance=0.7, size=None, *, assembly="fuse",
                      mesher=None, embed=None, verbose=False,
                      seams="gather", ligament=None, grading=0.35):
    """Embed a THIN VOLUME of the given width around each patch, junctions free.

    The finite-width fault representation: each planar patch is thickened by
    ``±width/2``, the thickened volumes of the whole network are resolved
    against one another in OCC — a junction becomes ordinary cells of the
    union, no geometric treatment, the rheology decides — the assembly is
    meshed standalone at layer scale (sub-``h`` widths are the
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
        ribbons. Patches may cross — that is the point — and may run PAST
        the domain: the assembly is clipped against the mesh's own
        boundary, and a clipped face left ON the boundary (an edge, in
        2-D) becomes the zone's OUTCROP BAND, carrying the skin label, the
        wall's labels and the ``<label>_trace`` label. The boundary need
        not be axis-aligned, and the band may cross a domain crease (a box
        edge). Patches must not touch a surface already embedded.
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
    assembly : {"fuse", "fragment"}, keyword-only
        How overlapping zones are resolved in CAD before meshing. Keyword-only
        so that inserting it ahead of ``verbose`` cannot rebind a positional
        ``verbose`` from an existing caller. ``"fuse"``
        (the default) returns the union as one region with no internal seam;
        ``"fragment"`` keeps each overlap piece as its own region, so the
        mesh conforms to the boundaries of the overlap. The zone mesh carries
        one label either way — a cell's fault properties come from the
        :class:`Surface` objects, not from the piece it was meshed in — so
        ``"fragment"`` is worth asking for only when those internal
        boundaries are themselves of interest. Two zones converging at a
        shallow angle make the overlap a spike, and its fragmented tip meshes
        to arbitrarily bad angles; the fused union has no such tip.
    mesher : {None, "ladder", "network"}, keyword-only
        How the band itself is meshed. ``None`` (default) is the CAD-built
        band (frontal fill). ``"network"`` (2-D) is the CAD-built band of
        a whole NETWORK of polylines in one call — ribbons fused (touching
        strands, junctions free) with every spine EMBEDDED so cuts walk
        exact vertices at any resolution; the one path for kissing joins,
        shared-band stepovers and plain strands alike. ``"ladder"`` is
        the STRUCTURED band with an
        exact mid-surface vertex sheet — the mandatory choice when the
        band's spine/mid-surface is to be cut/split (#595: a cut through a
        remeshed band snaps rail vertices), and the choice that makes
        placed level pairs NEST (a 2:1 sub-sampled ladder shares every
        vertex with the fine one — the composed-hierarchy economics,
        #629). In 2-D it takes one straight polyline (transfinite, three
        nodes across; :func:`_occ_ladder_assembly_2d`); in 3-D it takes
        one ``(grid, normals)`` pair of ``(nu, nv, 3)`` arrays — the
        sheet's own discretisation, offset ``±width/2`` into two prism
        layers and split to tets, no remesh
        (:func:`_ladder_assembly_3d`). Ladder bands must lie inside the
        domain.
    embed : sequence of array_like, keyword-only, 3-D ``mesher="network"``
        Planar polygons FRAGMENTED INTO the fused band as conforming
        interior surfaces — the mid-surfaces a split walks, at any width
        (the 2-D network mesher's embedded spines, one dimension up).
        Typically the un-expanded fault patches while ``patches`` carry
        their margin-expanded bands. Interior assemblies only (no
        outcrop clip yet). ``info["embedded_nodes"]`` returns one
        ``(m, 3)`` node-coordinate array per entry — the point set
        :func:`_label_mid_surface` labels a placed fault from.
    verbose : bool
        Report the counts.
    seams : {"gather", "ligament", "conform"}, keyword-only
        How a zone that straddles a partition seam is placed. ``"gather"``
        (the default) moves the zone's region onto one rank and carves
        there. ``"conform"`` (2-D) meshes the band THROUGH the seam with
        nothing moved: every band cell is made by the rank whose cavity
        holds its centroid, so the seam inside the band runs along band
        edges; each rank fills its own side, with the crossed seam span
        replaced by connectors to the skin and the skin on its side; the
        band vertices both sides use become shared points of the rebuild.
        The band keeps its own resolution everywhere (#670).
        ``"ligament"`` (2-D) butts the band up to the seam instead: the
        assembly is clipped a fraction of a cell short and the strip of
        fill left there — the LIGAMENT, labelled ``<label>`` and
        ``<label>_ligament`` — is band material at the base's resolution
        for the weak rheology to bridge.
    ligament : float or None, keyword-only
        With ``seams="ligament"``, widen the ligament: cells within
        ``ligament/2`` of the seam are also left uncarved. ``None`` keeps
        only the seam cells themselves (one layer on each side).
    grading : float, keyword-only
        How the fill's cell size grows from the band's own spacing at the
        skin to the base's at the cavity ring, as a power of the relative
        distance across the fill: 1 is linear; below 1 coarsens faster
        leaving the band (2-D). The default 0.35 is measured on the
        S-fault rig at fine width: 29% fewer cells than linear, the
        weak plane's answer unchanged to 0.5%, worst angle 17.2 against
        17.7 degrees.

    Returns
    -------
    placed : PETSc.DMPlex
        A new mesh with the assembly's cells embedded verbatim.
    info : dict
        Global counts: ``n_zone_cells``, ``n_skin_faces``, ``n_placed``
        (vertices added), ``n_removed`` (vertices deleted), ``min_volume``;
        with ``embed``, also ``embedded_nodes``.

    Raises
    ------
    NotImplementedError
        When the zone meets the boundary in more than one band, or the
        band runs along a domain crease with no cavity bowl beyond it.
    RuntimeError, ValueError
        Carve/fill refusals, always collective.
    """
    width = float(width)
    if width <= 0.0:
        raise ValueError("width must be positive")
    size = 0.9 * width if size is None else float(size)
    if assembly not in ("fuse", "fragment"):
        raise ValueError(
            f"assembly must be 'fuse' or 'fragment', not {assembly!r}")
    if mesher not in (None, "ladder", "network"):
        raise ValueError(
            f"mesher must be None, 'ladder' or 'network', not {mesher!r}")
    if seams not in ("gather", "ligament", "conform"):
        raise ValueError(
            f"seams must be 'gather', 'ligament' or 'conform', not {seams!r}")
    if ligament is not None and seams != "ligament":
        raise ValueError("ligament= belongs to seams='ligament'")
    if seams != "gather" and dm.getDimension() != 2:
        raise NotImplementedError(
            f"seams={seams!r} is built for the 2-D ribbon; the 3-D band "
            "still gathers (seams='gather').")
    if embed is not None and mesher != "network":
        raise ValueError(
            "embed= belongs to mesher='network' (mid-surfaces fragmented "
            "into the fused band); any other mesher would silently "
            "ignore it.")

    if dm.getDimension() == 2:
        return _place_thin_volume_2d(dm, patches, width, label, label_value,
                                     clearance, size, assembly, verbose,
                                     mesher=mesher, seams=seams,
                                     ligament=ligament, grading=grading)
    if mesher == "ladder":
        if len(patches) != 1:
            raise ValueError("the 3-D ladder takes exactly one patch")
        p0 = patches[0]
        if isinstance(p0, (tuple, list)) and len(p0) == 2:
            grid, grid_normals = p0
        else:
            grid = np.asarray(p0, dtype=float)
            if grid.ndim != 3 or grid.shape[-1] != 3:
                raise ValueError(
                    "the 3-D ladder patch is an (nu, nv, 3) sheet grid, or "
                    "a (grid, normals) pair of such arrays — the sheet's "
                    "own structured discretisation. (For nested levels, "
                    "subsample grid AND normals from the fine level's, or "
                    "use place_fault_ribbon, which does.)")
            grid_normals = _grid_normals(grid)
        patches = [(grid, grid_normals)]
    if dm.getDimension() != 3:
        raise NotImplementedError(
            f"place_thin_volume takes a 2-D or 3-D simplex mesh; this mesh "
            f"is {dm.getDimension()}-D.")

    comm = uw.mpi.comm

    # The specify-long contract: patches may extend PAST the domain; the
    # assembly is clipped against the mesh's own boundary, and a clipped
    # face left ON the boundary becomes the zone's OUTCROP BAND — on any
    # boundary, axis-aligned or not.
    dom_verts, dom_tris = _domain_boundary_facets(dm)

    # ------------------------------------------ the assembly, once, shared
    failure = None
    payload = None
    if comm.rank == 0:
        try:
            embedded_nodes = None
            if mesher == "ladder":
                # The extruded band: no CAD, no remesh — the sheet's own
                # triangulation offset to prisms. Interior bands only
                # (no domain clip); a protruding ladder fails the carve.
                asm_pts, asm_tets = _ladder_assembly_3d(
                    patches[0][0], patches[0][1], width)
            elif mesher == "network":
                # fuse + embedded mid-surfaces (interior assemblies only):
                # the fault surfaces become conforming faces of the band,
                # so the split walks them at any width
                asm_pts, asm_tets, _cad_vol, embedded = _occ_assembly_3d(
                    patches, width, size, domain=None, assembly=assembly,
                    embed=embed)
                if embedded is not None:
                    # coordinates, not indices: the imprint collapse below
                    # may renumber, but interior points do not move
                    embedded_nodes = [
                        np.asarray(asm_pts[np.unique(t)], dtype=float)
                        for t in embedded]
            else:
                # The meshed-vs-CAD volume gate runs inside the assembly
                # builder, before its boundary snap.
                asm_pts, asm_tets, _cad_vol, _no_embed = _occ_assembly_3d(
                    patches, width, size, domain=(dom_verts, dom_tris),
                    assembly=assembly)
            asm_pts, asm_tets = _collapse_boundary_imprints_3d(
                asm_pts, asm_tets, dom_verts, dom_tris, 0.1 * size)
            payload = (asm_pts, asm_tets, embedded_nodes)
        # Exception, not just RuntimeError/ValueError: a raw gmsh error
        # (e.g. a PLC intersection) is a plain Exception, and an
        # uncaught raise on the surgery rank is a HANG for its peers —
        # every failure must become a collective refusal.
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    failures = comm.allgather(failure)
    real = [f for f in failures if f]
    if real:
        raise RuntimeError(f"place_thin_volume assembly failed: {real[0]}")
    asm_pts, asm_tets, embedded_nodes = comm.bcast(payload, root=0)
    skin_xyz, skin_tris, skin_node_ids = _assembly_skin(asm_pts, asm_tets)

    # The outcrop band: the skin's trace on the domain boundary.
    interior_idx, band_idx = _split_skin_trace(skin_xyz, skin_tris,
                                               dom_verts, dom_tris)
    outcropping = bool(len(band_idx))
    band_outline = None
    if outcropping:
        _refuse_multiple_bands(skin_tris[band_idx])
        from collections import Counter as _Counter
        band_edge = _Counter()
        for t in skin_tris[band_idx]:
            a, b, c = sorted(int(v) for v in t)
            for e in ((a, b), (a, c), (b, c)):
                band_edge[e] += 1
        band_outline = _single_loop(
            [e for e, k in band_edge.items() if k == 1],
            "zone's outcrop band outline")
    skin_tris_fill = (skin_tris[interior_idx] if outcropping
                      else skin_tris)

    # -------------------------------------------------- mark, then gather
    vS, vE = dm.getDepthStratum(0)
    pStart, pEnd = dm.getChart()
    X = _coords(dm)[: vE - vS]
    cells = _tet_vertices(dm)
    h_vertex, _h_cell = _vertex_h_3d(dm, cells, len(X))
    reach_v = np.maximum(clearance * h_vertex, 0.6 * width)
    d_skin = _sheet_distance_within(X, skin_xyz, skin_tris,
                                    reach_v + 1.0 * h_vertex)
    # One region per connected component of the assembly (#670): each
    # zone's shell goes to its own rank, and a zone whose shell is already
    # interior to a rank moves nothing. The outcrop and ladder paths carry
    # single-rank machinery (the bowl, the cap, the extrusion) and keep one
    # region.
    comp_of_tet = _assembly_components(asm_tets)
    n_comp = int(comp_of_tet.max())
    single = outcropping or mesher == "ladder" or n_comp == 1
    mark = np.zeros(pEnd - pStart, dtype=np.int32)
    if single:
        mark[np.flatnonzero(d_skin < reach_v + 1.0 * h_vertex)
             + vS - pStart] = 1
    else:
        for k in range(1, n_comp + 1):
            xyz_k, tris_k, _ids_k = _assembly_skin(
                asm_pts, asm_tets[comp_of_tet == k])
            d_k = _sheet_distance_within(X, xyz_k, tris_k,
                                         reach_v + 1.0 * h_vertex)
            near = np.flatnonzero(d_k < reach_v + 1.0 * h_vertex)
            mark[near + vS - pStart] = np.maximum(
                mark[near + vS - pStart], k)

    volume_before = np.array([_owned_cell_volume(dm)], dtype=float)
    comm.Allreduce(MPI.IN_PLACE, volume_before, op=MPI.SUM)

    dm_work, moved, n_moved, owner, canon = _gather_regions(
        dm, mark, verbose=verbose)
    if n_moved:
        vS, vE = dm_work.getDepthStratum(0)
        pStart, pEnd = dm_work.getChart()
        X = _coords(dm_work)[: vE - vS]
        cells = _tet_vertices(dm_work)
        h_vertex, _h_cell = _vertex_h_3d(dm_work, cells, len(X))
        reach_v = np.maximum(clearance * h_vertex, 0.6 * width)
        d_skin = _sheet_distance_within(X, skin_xyz, skin_tris,
                                        reach_v + 1.0 * h_vertex)

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

    mine_regions = {r for r, rk in owner.items() if rk == comm.rank}
    my_comps = [k for k in range(1, n_comp + 1)
                if canon[k if not single else 1] in mine_regions]
    mine = bool(my_comps)
    target = owner[canon[1]]          # the root of the outcrop broadcasts
    if mine and not single:
        # this rank's share of the assembly, compacted: its components'
        # cells, their nodes, and the skin of that subset (components are
        # face-disjoint, so the subset's skin is its components' skins)
        sel = np.isin(comp_of_tet, my_comps)
        used = np.unique(asm_tets[sel])
        remap = np.full(len(asm_pts), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        asm_pts_m = asm_pts[used]
        asm_tets_m = remap[asm_tets[sel]]
        skin_xyz_m, skin_tris_m, skin_node_ids_m = _assembly_skin(
            asm_pts_m, asm_tets_m)
        skin_tris_fill_m = skin_tris_m
    elif mine:
        asm_pts_m, asm_tets_m = asm_pts, asm_tets
        skin_xyz_m, skin_tris_m, skin_node_ids_m = (
            skin_xyz, skin_tris, skin_node_ids)
        skin_tris_fill_m = skin_tris_fill
    else:
        asm_pts_m = np.empty((0, 3), dtype=float)
        asm_tets_m = np.empty((0, 4), dtype=np.int64)
        skin_xyz_m = np.empty((0, 3), dtype=float)
        skin_tris_m = skin_tris_fill_m = np.empty((0, 3), dtype=np.int64)
        skin_node_ids_m = np.empty(0, dtype=np.int64)

    failure = None
    victims = drop_ids = None
    fill = shell_vert_ids = None
    if mine:
        try:
            deletable = near = None
            if outcropping:
                dom_region, dom_planes = _coplanar_regions(dom_verts,
                                                           dom_tris)
                deletable, near, _regions = _outcrop_frame_3d(
                    X, on_wall, dom_verts, dom_tris, dom_region,
                    skin_xyz_m, skin_tris_m[band_idx])
            victims, drop_ids, shell, cap_faces = _carve_around_volume_3d(
                dm_work, X, cells, skin_xyz_m, skin_tris_m, reach_v, reach_c,
                held_cells, on_wall, shared, open_deletable=deletable,
                open_near=near)
            touched = set()
            for c in drop_ids:
                for q in dm_work.getTransitiveClosure(int(c) + cS)[0]:
                    touched.add(int(q))
            if any(shared[q - pStart] for q in touched):
                raise RuntimeError(
                    "place_thin_volume internal: the gathered region touches "
                    "a shared point; the gather mask under-reached.")

            shell_vert_ids = sorted(
                {v for _f, verts in shell for v in verts}
                | {v for _f, verts in cap_faces for v in verts})
            local = {v: i for i, v in enumerate(shell_vert_ids)}
            shell_xyz = X[shell_vert_ids]
            shell_tris = np.array([[local[v] for v in verts]
                                   for _f, verts in shell], dtype=np.int64)

            cap_payload = None
            removed_wall = []
            if cap_faces:
                # Labels each removed wall face carried, read PER FACE
                # before the rebuild forgets them — the bowl can span
                # several walls (a band across a box edge, a curved
                # boundary's facets), so each new wall triangle takes the
                # labels of the old face it lies on, not a set common to
                # the whole bowl. The 2-D per-segment rule one level up.
                names = [dm_work.getLabelName(i)
                         for i in range(dm_work.getNumLabels())
                         if dm_work.getLabelName(i)
                         not in reconnect._TOPOLOGY_LABELS]
                for f, verts in cap_faces:
                    pairs_f = []
                    for name in names:
                        val = dm_work.getLabel(name).getValue(int(f))
                        if val >= 0:
                            pairs_f.append((name, int(val)))
                    removed_wall.append((X[np.asarray(verts)].copy(),
                                         pairs_f))

                cap_tris_mesh = np.array([verts for _f, verts in cap_faces],
                                         dtype=np.int64)
                d_cap, at_cap = _nearest_facet(
                    X[cap_tris_mesh].mean(axis=1), dom_verts, dom_tris)
                if (d_cap > 1e-9).any():
                    raise RuntimeError(
                        "an outcrop bowl wall face lies off the gathered "
                        "boundary complex; the wall mask and the complex "
                        "disagree")
                _d_band, at_band = _nearest_facet(
                    skin_xyz_m[skin_tris_m[band_idx]].mean(axis=1),
                    dom_verts, dom_tris)
                alive = np.ones(len(X), dtype=bool)
                alive[np.asarray(victims, dtype=np.int64)] = False
                cap_nodes, hole_nodes, cap_extra, cap_tris = \
                    _outcrop_collar_3d(
                        X, alive, cap_tris_mesh, dom_region[at_cap],
                        dom_planes, skin_xyz_m, skin_tris_m[band_idx],
                        dom_region[at_band], band_outline)
                cap_payload = {
                    "rim_shell_local": [local[v] for v in cap_nodes],
                    "hole_skin_local": list(hole_nodes),
                    "tris": cap_tris,
                    "extra_xyz": cap_extra,
                }

            fill = _gmsh_fill_annulus_3d(shell_xyz, shell_tris, skin_xyz_m,
                                         skin_tris_fill_m, size_out=h,
                                         size_in=size, cap=cap_payload)
            (_pts, _tets, moved_nodes, skin_out, _n_shell,
             cap_out) = fill
            if moved_nodes:
                raise RuntimeError(
                    f"the gap fill moved {moved_nodes} constrained node(s); "
                    "the cavity cannot be sewn back.")
            if skin_out != len(skin_tris_fill_m):
                raise RuntimeError(
                    f"the gap fill remeshed the skin ({skin_out} triangles "
                    f"for {len(skin_tris_fill_m)} given).")
            if cap_payload is not None and (
                    cap_out is None
                    or len(cap_out) != len(cap_payload["tris"])):
                raise RuntimeError(
                    "the gap fill remeshed the outcrop cap.")
        # Exception, not just RuntimeError/ValueError: a raw gmsh error
        # (e.g. a PLC intersection) is a plain Exception, and an
        # uncaught raise on the surgery rank is a HANG for its peers —
        # every failure must become a collective refusal.
        except Exception as exc:
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
    if mine:
        fill_pts, fill_tets, _m, _s, n_shell, cap_out = fill
        n_skin = len(skin_xyz_m)
        skin_row = np.asarray(skin_node_ids_m, dtype=np.int64)
        gap_new = fill_pts[n_shell + n_skin:]

        def gap_code(v):
            if v < n_shell:
                return int(shell_vert_ids[v])
            if v < n_shell + n_skin:
                return -(int(skin_row[v - n_shell]) + 1)
            return -(len(asm_pts_m) + (int(v) - n_shell - n_skin) + 1)

        made = np.array(
            [[gap_code(int(v)) for v in tet] for tet in fill_tets]
            + [[-(int(v) + 1) for v in tet] for tet in asm_tets_m],
            dtype=np.int64)
        placed = np.vstack([asm_pts_m, gap_new])
        victims_arr = np.asarray(victims, dtype=np.int64)
        drop_arr = np.asarray(drop_ids, dtype=np.int64)
    else:
        made = np.empty((0, 4), dtype=np.int64)
        placed = np.empty((0, 3), dtype=float)
        victims_arr = np.empty(0, dtype=np.int64)
        drop_arr = np.empty(0, dtype=np.int64)

    new, point_map, placed_new = _rebuild_sewn(
        dm_work, drop_arr, victims_arr, made, placed)

    # Label the zone's cells and the skin's faces, by joining vertex tuples.
    # Two labels, deliberately: a stratum holding cells is a volume label and
    # is invisible to the interface machinery, so the skin faces — which a
    # later placement must hold clear of — go under their own name.
    skin_label = label + "_skin"
    trace_label = label + "_trace"
    for name in (label, skin_label, trace_label):
        if not new.hasLabel(name):
            new.createLabel(name)
    n_cells_local = 0
    n_skin_local = 0
    if mine:
        out_label = new.getLabel(label)
        out_skin = new.getLabel(skin_label)
        for tet in asm_tets_m:
            joined = new.getFullJoin([int(placed_new[int(v)]) for v in tet])
            if len(joined) != 1:
                failure = ("an assembly cell is not a cell of the sewn mesh; "
                           "the embed lost the layer.")
                break
            out_label.setValue(int(joined[0]), int(label_value))
            n_cells_local += 1
        else:
            for tri in skin_tris_m:
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

    # An outcropping zone's new wall coverage — the cap's collar AND the
    # band (the zone's own face on the wall) — is relabelled EXPLICITLY
    # with what the replaced wall faces carried, full closures included.
    # Each new triangle takes the labels of the removed face it lies on
    # (the 2-D per-segment rule), so a bowl spanning several walls
    # restores each wall's own labels.
    pairs = comm.bcast(
        sorted({p for _tri, pairs_f in removed_wall for p in pairs_f})
        if mine else None, root=target)
    n_wall_expect = comm.bcast(
        ((len(cap_out) if cap_out is not None else 0) + len(band_idx))
        if mine else 0, root=target)
    for name, val in (pairs or []):
        if not new.hasLabel(name):
            new.createLabel(name)
    n_wall_local = 0
    if mine and outcropping:
        n_shell_ids = np.asarray(shell_vert_ids, dtype=np.int64)
        n_skin = len(skin_xyz_m)

        def new_id(v):
            if v < n_shell:
                old_pt = (int(n_shell_ids[v])
                          + dm_work.getDepthStratum(0)[0])
                return int(point_map[old_pt - pStart])
            if v < n_shell + n_skin:
                return int(placed_new[int(skin_row[v - n_shell])])
            return int(placed_new[len(asm_pts_m) + (v - n_shell - n_skin)])

        wall_tris = ([[new_id(int(v)) for v in t] for t in cap_out]
                     if cap_out is not None else [])
        n_cap_tris = len(wall_tris)
        wall_tris += [[int(placed_new[int(skin_row[int(v)])]) for v in t]
                      for t in skin_tris_m[band_idx]]
        centres = np.array(
            [fill_pts[np.asarray(t)].mean(axis=0) for t in cap_out]
            if cap_out is not None else np.zeros((0, 3)))
        if len(band_idx):
            band_cen = skin_xyz_m[skin_tris_m[band_idx]].mean(axis=1)
            centres = (np.vstack([centres, band_cen]) if len(centres)
                       else band_cen)
        old_pts = np.vstack([tri for tri, _p in removed_wall])
        old_tris = np.arange(3 * len(removed_wall)).reshape(-1, 3)
        d_old, at_old = _nearest_facet(centres, old_pts, old_tris)
        # A raise here is rank-local — a hang at np>=2 — so every
        # refusal in this block goes through the collective failure.
        if (d_old > 1e-9).any():
            failure = ("an outcrop wall triangle lies on no removed wall "
                       "face; the collar and the bowl disagree")
        else:
            # The cap's collar first, then the band — the TRACE, the
            # intersection itself, labelled as such so the model can form
            # whatever unions it needs (never a partition of the wall
            # into named pieces).
            out_trace = new.getLabel(trace_label)
            for k, ids in enumerate(wall_tris):
                joined = new.getFullJoin(ids)
                if len(joined) != 1:
                    failure = ("an outcrop wall triangle is not a face of "
                               "the sewn mesh; the cap or band was not "
                               "sewn.")
                    break
                for q in new.getTransitiveClosure(int(joined[0]))[0]:
                    for name, val in removed_wall[int(at_old[k])][1]:
                        new.getLabel(name).setValue(int(q), int(val))
                    if k >= n_cap_tris:
                        out_trace.setValue(int(q), int(label_value))
                n_wall_local += 1
    failures = comm.allgather(failure)
    real = [f for f in failures if f]
    if real:
        raise RuntimeError(real[0])
    n_wall = int(comm.allreduce(n_wall_local, op=MPI.SUM))
    if n_wall != n_wall_expect:
        raise RuntimeError(
            f"{n_wall} outcrop wall faces relabelled for {n_wall_expect} "
            "given.")

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

    # The surgery must CONSERVE the domain's topology, not assume it: a
    # box is a ball (Euler 1) but a spherical shell is S^2 x I (Euler 2),
    # and the general boundary clip makes such domains reachable.
    owned = np.asarray(_owned_stratum_counts(dm), dtype=np.int64)
    comm.Allreduce(MPI.IN_PLACE, owned, op=MPI.SUM)
    euler_before = int(owned[0] - owned[1] + owned[2] - owned[3])
    owned = np.asarray(_owned_stratum_counts(new), dtype=np.int64)
    comm.Allreduce(MPI.IN_PLACE, owned, op=MPI.SUM)
    nv_g, ne_g, nf_g, nc_g = (int(x) for x in owned)
    if nv_g - ne_g + nf_g - nc_g != euler_before:
        raise RuntimeError(
            f"the placement changed the global Euler number: "
            f"{euler_before} -> {nv_g - ne_g + nf_g - nc_g}")

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

    _validity_and_orientation_gates(new, comm)

    min_vol = np.array([_owned_min_cell_volume(new)], dtype=float)
    comm.Allreduce(MPI.IN_PLACE, min_vol, op=MPI.MIN)

    info = {"n_zone_cells": n_zone, "n_skin_faces": n_skin_faces,
            "n_placed": n_placed, "n_removed": n_removed,
            "n_trace_facets": int(len(band_idx)),
            "min_volume": float(min_vol[0])}
    if embedded_nodes is not None:
        info["embedded_nodes"] = embedded_nodes
    if verbose:
        uw.pprint(f"[place_thin_volume {label!r}] {info['n_zone_cells']} "
                  f"zone cells, {info['n_skin_faces']} skin faces; placed "
                  f"{info['n_placed']} vertices, removed "
                  f"{info['n_removed']}")
    info["n_gathered"] = int(moved)   # the regions' size, star and layer (#670)
    info["n_moved"] = int(n_moved)    # cells that changed rank for it
    info["n_regions"] = len(set(canon.values()))
    if embedded_nodes is not None:
        # which region each embedded mid-surface's zone became: the split
        # redistributes per region too (#670), so it must not gather what
        # the placement kept apart
        node_comp = np.zeros(len(asm_pts), dtype=np.int32)
        for k in range(1, n_comp + 1):
            node_comp[np.unique(asm_tets[comp_of_tet == k])] = k
        region_of_comp = {k: (canon[k] if not single else canon[1])
                          for k in range(1, n_comp + 1)}
        info["embedded_regions"] = []
        for pts in embedded_nodes:
            hit = np.flatnonzero(
                np.all(np.isclose(asm_pts[:, None, :], pts[None, :1, :],
                                  atol=1e-12), axis=2))
            k = int(node_comp[hit[0]]) if hit.size else 1
            info["embedded_regions"].append(region_of_comp.get(k, 1))
    return new, info


def _cell_volumes_signed6(dm):
    """Signed 6*volume (3-D) or signed 2*area (2-D) per cell — the sign is
    the point: abs()-based measures are structurally blind to inversion."""
    X = _coords(dm)
    vS, vE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    dim = dm.getDimension()
    cells = np.array([[int(p) - vS for p in dm.getTransitiveClosure(c)[0]
                       if vS <= p < vE] for c in range(cS, cE)],
                     dtype=np.int64).reshape(cE - cS, dim + 1)
    P = X[cells]
    if dim == 3:
        return np.einsum("ij,ij->i",
                         np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]),
                         P[:, 3] - P[:, 0])
    return ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
            - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))


def _owned_cell_volume(dm):
    """Total volume of this rank's cells. Cells are never shared, so the
    allreduced sum is the domain volume."""
    return float(np.abs(_cell_volumes_signed6(dm)).sum() / 6.0)


def _owned_min_cell_volume(dm):
    v = np.abs(_cell_volumes_signed6(dm)) / 6.0
    return float(v.min()) if len(v) else np.inf


def _extend_grid(grid, rings):
    """Continue a sheet grid ``rings`` rows/columns outward, linearly.

    The tip-margin builder for :func:`place_fault_ribbon`: the user's
    grid IS the fault and must be honoured exactly, so the resolved
    band around it is grown from INVENTED surround — each grid line
    continued along its own end tangent at the local spacing, one ring
    at a time (corners fill consistently because the column pass also
    extends the freshly added rows). Deliberately linear: the data ends
    where it ends, and a tangent continuation adds no curvature the
    structural model never asserted."""
    G = np.asarray(grid, dtype=float)
    for _ in range(int(rings)):
        G = np.concatenate([(2.0 * G[0] - G[1])[None], G,
                            (2.0 * G[-1] - G[-2])[None]], axis=0)
        G = np.concatenate([(2.0 * G[:, 0] - G[:, 1])[:, None], G,
                            (2.0 * G[:, -1] - G[:, -2])[:, None]], axis=1)
    return G


def _label_mid_surface(dm, spine_points, label, value):
    """Label the ladder band's mid-surface faces by coordinate selection.

    The ladder's mid-surface is a real vertex sheet, so a fault face is
    simply an interior face whose three vertices all lie on the (inset)
    spine point set — no plane test, so a curved sheet needs nothing
    special. All-rim erosion follows (#626): the splitter refuses a face
    with no interior vertex, so faces whose every vertex is on the
    selection's rim are dropped until none remain.
    """
    from collections import Counter

    keep = {tuple(q) for q in
            np.asarray(spine_points, dtype=float).reshape(-1, 3)
            .round(9).tolist()}
    X = _coords(dm)
    fS, fE = dm.getHeightStratum(1)
    vS, vE = dm.getDepthStratum(0)
    sel = {}
    for f in range(fS, fE):
        if dm.getSupportSize(f) != 2:
            continue
        verts = [int(q) - vS for q in dm.getTransitiveClosure(f)[0]
                 if vS <= int(q) < vE]
        if all(tuple(X[v].round(9)) in keep for v in verts):
            sel[f] = tuple(verts)
    while True:
        edge_use = Counter()
        for verts in sel.values():
            a, b, c = sorted(verts)
            for e in ((a, b), (a, c), (b, c)):
                edge_use[e] += 1
        rim_v = set()
        for (a, b), k in edge_use.items():
            if k == 1:
                rim_v.update((a, b))
        bad = [f for f, verts in sel.items()
               if all(v in rim_v for v in verts)]
        if not bad:
            break
        for f in bad:
            del sel[f]
    # Gather-first placement: the placed band lives on ONE rank, so the
    # other ranks legitimately select no face. The refusal is judged on
    # the GLOBAL count and raised collectively — a rank-local raise on
    # an empty selection is a hang for the peers.
    comm = dm.getComm().tompi4py()
    n_global = int(comm.allreduce(len(sel)))
    if n_global == 0:
        raise ValueError(
            f"no {label!r} faces survive on the mid-surface: the inset "
            f"leaves too small an interior (enlarge the sheet grid or "
            f"reduce inset_rings).")
    dm.createLabel(label)
    lbl = dm.getLabel(label)
    for f in sel:
        lbl.setValue(f, int(value))
    return n_global


def place_fault_ribbon(base_mesh, sheet, width, *, normals=None,
                       label="Fault", label_value=41, band_label="Band",
                       band_value=71, margin_rings=2, mid_level=True,
                       clearance=0.3, split=True, verbose=False):
    """A split-ready fault ribbon from the fault surface's OWN mesh (#629).

    The conceptually simple fault-prep path: the user supplies the fault
    surface as a structured grid of points — their own discretisation of
    it, e.g. sampled from a structural model — and this function
    thickens it into the resolved band (no remeshing,
    :func:`_ladder_assembly_3d`), embeds it in the mesh, labels and
    splits the mid-surface into a frictionless-ready fault, and builds
    the matching UNSPLIT intermediate level for the multigrid hierarchy
    — all from ONE parametrisation, so every level nests
    vertex-for-vertex by construction.

    **The grid IS the fault, honoured exactly.** The tip rule (a split
    must never reach the band's rim, and its tip needs band-resolution
    material on every side) is satisfied by EXTRAPOLATION, not by
    shrinking the request: the band is built on the sheet continued
    ``margin_rings`` rings outward along its own end tangents
    (:func:`_extend_grid` — linear, no invented curvature), and the
    slip surface is labelled on precisely the points supplied. The
    intact extrapolated frame around the fault is the 3-D analogue of
    the junction ruling's intact gap. (One machinery caveat: the
    splitter refuses a face with no interior vertex, so a corner
    TRIANGLE of the slip patch — all three vertices on the patch's own
    perimeter — is eroded; on a structured sheet that is at most a few
    half-cells at the corners, reported via ``n_slit_faces``.)

    Parameters
    ----------
    base_mesh : uw.discretisation.Mesh
        The background mesh (3-D simplex). Not modified.
    sheet : array_like, (nu, nv, 3)
        THE FAULT surface as a structured point grid — curved is fine;
        the sheet (plus its extrapolated margin) must lie inside the
        domain. With ``mid_level=True``, ``nu`` and ``nv`` must be odd
        so the 2:1 subsample of the band grid exists.
    width : float
        Ribbon thickness (the mesh-bridging width of a split-node model:
        a resolution parameter, nothing rheological — see the design
        note's w ruling).
    normals : array_like, (nu, nv, 3), optional
        Per-vertex sheet normals. Default: derived from the grid
        (central differences). Extended over the margin by the same
        tangent continuation as the points; the intermediate level
        always subsamples the SAME field, which is what keeps the
        levels nested.
    label, label_value : str, int
        The fault label carried by the mid-surface faces; after the
        split this is the boundary for ``add_fault_bc``.
    band_label, band_value : str, int
        The label carried by the ribbon's cells (the ``fac_zone`` key if
        volumetric rheology is later painted into the band; under pure
        contact it is reporting only).
    margin_rings : int
        Tip margin: the band extends this many grid rings BEYOND the
        fault on every side, built by extrapolation. Must be >= 1 (the
        tip rule; 2 is the measured default).
    mid_level : bool
        Also build the band at 2:1 (UNSPLIT) on the base mesh — the
        bridge level of the composed hierarchy.
    clearance : float
        Carve clearance, in local-h units. The default 0.3 is the
        measured production choice: the ``0.6 * width`` floor governs,
        keeping the unstructured fill shell thin (at the default 0.7 the
        shell was the majority of the fine level's nodes).
    split : bool
        Split the fault (:func:`~underworld3.utilities.fault_split.split_fault`).
        ``False`` returns the labelled, unsplit mesh (e.g. for painted
        weak/TI models on the same geometry).
    verbose : bool
        Report counts.

    Returns
    -------
    mesh : uw.discretisation.Mesh
        The fault-resolving mesh, split (or labelled) and ready for
        ``add_fault_bc(..., boundary=label)``.
    mid : uw.discretisation.Mesh or None
        The unsplit 2:1 band level (``mid_level=False`` gives None).
    info : dict
        ``n_slit_faces``, ``n_cells``, ``n_mid_cells``, ``spacing``,
        and ``footprint`` — the FAULT-footprint cell mask (band cells
        nearest a USER grid point, not the extrapolated margin): what
        a painted rheology or ``fac_zone`` key should use.

    Examples
    --------
    >>> mesh, mid, info = place_fault_ribbon(base, sheet_grid, width=0.06)
    >>> stokes = uw.systems.Stokes(mesh, ...)
    >>> stokes.add_fault_bc(0, boundary="Fault")
    >>> set_custom_fmg(stokes, base._coarse_level_meshes()[:-1] + [base, mid],
    ...                field_id=0)     # pure contact: no fac_zone
    """
    from enum import Enum

    from underworld3 import discretisation
    from underworld3.utilities.fault_split import split_fault

    G = np.asarray(sheet, dtype=float)
    if G.ndim != 3 or G.shape[2] != 3 or min(G.shape[:2]) < 3:
        raise ValueError(
            f"sheet must be an (nu, nv, 3) grid with nu, nv >= 3; got "
            f"shape {G.shape}")
    nu, nv = G.shape[:2]
    if mid_level and (nu % 2 == 0 or nv % 2 == 0):
        raise ValueError(
            f"mid_level=True needs ODD grid dimensions so the 2:1 "
            f"subsample of the band grid exists; got {nu} x {nv}.")
    m = int(margin_rings)
    if m < 1:
        raise ValueError(
            "margin_rings must be >= 1: the split cannot reach the band "
            "rim (the tip rule), and the margin comes from extrapolated "
            "surround — the requested fault is honoured either way.")
    N = (_grid_normals(G) if normals is None
         else np.asarray(normals, dtype=float))
    if N.shape != G.shape:
        raise ValueError(f"normals must match the sheet shape {G.shape}")
    # The BAND grid: the fault continued m rings outward; normals extended
    # by the same tangent continuation, then renormalised.
    Gb = _extend_grid(G, m)
    Nb = _extend_grid(N, m)
    Nb = Nb / np.linalg.norm(Nb, axis=2)[..., None]
    spacing = float(np.mean([
        np.linalg.norm(np.diff(G, axis=0), axis=2).mean(),
        np.linalg.norm(np.diff(G, axis=1), axis=2).mean()]))

    dm_fine, info_f = place_thin_volume(
        base_mesh.dm, [(Gb, Nb)], width, label=band_label,
        label_value=band_value, clearance=clearance, size=spacing,
        mesher="ladder", verbose=verbose)
    n_slit = _label_mid_surface(dm_fine, G, label, label_value)

    members = {b.name: b.value for b in base_mesh.boundaries}
    members[label] = int(label_value)
    boundaries = Enum("boundaries", members)
    mesh = discretisation.Mesh(
        dm_fine, simplex=True, qdegree=base_mesh.qdegree,
        coordinate_system_type=base_mesh.CoordinateSystem.coordinate_type,
        boundaries=boundaries, verbose=False)
    if split:
        mesh = split_fault(mesh, label)

    mid = None
    n_mid = 0
    if mid_level:
        # 2:1 subsample of the BAND grid (odd user dims + even 2m stay
        # odd), same extended parametrisation -> the levels nest.
        dm_mid, _info_m = place_thin_volume(
            base_mesh.dm, [(Gb[::2, ::2], Nb[::2, ::2])], width,
            label=band_label, label_value=band_value, clearance=clearance,
            size=2.0 * spacing, mesher="ladder", verbose=verbose)
        mid = discretisation.Mesh(
            dm_mid, simplex=True, qdegree=base_mesh.qdegree,
            coordinate_system_type=base_mesh.CoordinateSystem.coordinate_type,
            boundaries=base_mesh.boundaries, verbose=False)
        n_mid = mid.dm.getHeightStratum(0)[1]

    # The FAULT FOOTPRINT mask (the honoured-paint rule): band cells
    # whose nearest extended-grid node is a USER grid point — what a
    # volumetric rheology or fac_zone key should use, never the whole
    # band (the margin is extrapolated surround).
    nu_b, nv_b = Gb.shape[:2]
    ii, jj = np.meshgrid(np.arange(nu_b), np.arange(nv_b), indexing="ij")
    is_user = ((ii >= m) & (ii < nu_b - m)
               & (jj >= m) & (jj < nv_b - m)).ravel()
    footprint = _footprint_from_samples(
        mesh.dm, mesh.cells_labelled(band_label, band_value),
        Gb.reshape(-1, 3), is_user)

    info = {"n_slit_faces": int(n_slit),
            "n_cells": int(mesh.dm.getHeightStratum(0)[1]),
            "n_mid_cells": int(n_mid),
            "spacing": spacing,
            "footprint": footprint}
    if verbose:
        import underworld3 as _uw
        _uw.pprint(f"[place_fault_ribbon {label!r}] {info['n_cells']} cells "
                   f"({info['n_slit_faces']} fault faces"
                   f"{', split' if split else ''})"
                   + (f", mid level {n_mid} cells" if mid_level else ""))
    return mesh, mid, info


def place_fault_ribbon_2d(base_mesh, traces, width, *, margin_rings=2,
                          band_label="Band", band_value=71,
                          clearance=0.3, split=True, mesher="ladder",
                          spines=None, verbose=False, seams="gather",
                          ligament=None, grading=0.35):
    """Split-ready 2-D fault ribbons from the traces' OWN sampling (#629).

    The 2-D production fault-prep path, honouring the same contract set
    as :func:`place_fault_ribbon`: each supplied polyline IS a fault —
    its points, sampled by the user at fault scale, become the band's
    spine vertices verbatim (curved is fine; the ladder rails are mitred
    offsets) — the tip margin is EXTRAPOLATED ``margin_rings`` points
    along the end tangents (never confiscated), and the cut consumes the
    spine's own vertices (#595: nothing snaps). Multiple traces are
    placed sequentially — a fault network of stop-short strands per the
    junction ruling (the intact gap is the linkage; strands must not
    touch).

    Parameters
    ----------
    base_mesh : uw.discretisation.Mesh
        The background mesh (2-D simplex). Not modified.
    traces : list of (label, polyline)
        Each polyline an ``(n, 2)`` array — THE FAULT, sampled at the
        rung scale the band should have (≈2 elements across ``width``
        keeps prism aspect near 1). After the split each ``label`` is a
        boundary for ``add_fault_bc``.
    width : float
        Band thickness (split-node models: a resolution parameter).
    margin_rings : int or sequence of (int, int)
        The band extends this many points beyond each fault end, by
        linear tangent continuation. Must be >= 1 (the tip rule). One
        count for every end, or a ``(start, end)`` pair per trace.
    band_label, band_value : str, int
        Cell label of the band. With ``mesher="ladder"`` trace ``k`` gets
        ``band_value + k`` so per-fault zones stay distinguishable
        (``mesh.cells_labelled(band_label)`` unions them); with
        ``mesher="network"`` the fused band is ONE region carrying
        ``band_value``. Either way ``info["band"]`` is the union mask and
        ``info["footprints"]`` the per-fault ones.
    clearance : float
        Carve clearance (the measured thin-shell default 0.3).
    spines : list of (label, polyline), optional
        The polylines to PLACE, when they differ from the traces to cut:
        a collinear abutting pair must share one spine (two ribbons
        overlapping along one line interleave their vertices into
        slivers), with the cut stopping at each piece's own ends. Every
        trace vertex must be a vertex of some spine. Default: the
        traces themselves.
    mesher : {"ladder", "network"}
        How the band is meshed. ``"ladder"`` (default) places each trace
        SEQUENTIALLY as a structured band: bands may not touch, and level
        pairs NEST (a 2:1 sub-sampled ladder shares every vertex — the
        composed-hierarchy economics, #629). ``"network"`` places the
        whole set in ONE call: the ribbons are fused in CAD so strands may
        touch — a kissing junction, a shared stepover band — and every
        spine is embedded, so the cuts still walk exact vertices at any
        resolution. Choose ``"network"`` whenever the traces come within a
        band width of one another; choose ``"ladder"`` when the placed
        levels must nest.
    split : bool
        Cut + split each trace (``mesh.add_fault``). ``False`` returns
        the placed, unlabelled-fault mesh for painted (volumetric)
        models on identical geometry — remember the paint stops at the
        fault FOOTPRINT, not the band.
    seams : {"gather", "ligament"}
        How a band crossing a partition seam is placed (#670).
        ``"gather"`` moves the band's region onto one rank and carves
        there; the split then runs with serial topology. ``"ligament"``
        carves on every rank, clips the band one cell short of each seam
        and leaves the base cells there as the LIGAMENT: the trace is
        not cut through it, so the split runs rank-local with the
        weak-plane rheology bridging the seam
        (:meth:`FaultNetwork.apply` paints it on the ligament cells).
        Nothing is redistributed. See
        :func:`place_thin_volume` for the mechanism.
    ligament : float or None
        With ``seams="ligament"``, widen the ligament to the cells within
        ``ligament/2`` of the seam.
    verbose : bool
        Report counts.

    Returns
    -------
    mesh : uw.discretisation.Mesh
        The fault-resolving mesh (split when ``split=True``).
    info : dict
        ``n_cells``, ``spacing`` / ``n_rungs`` (per trace), the
        ``mesher`` used, the ``extended`` spines (user samples plus the
        margin) and the band ``width``; ``footprints`` — per-label
        FAULT-footprint cell masks (the honoured-paint rule): what
        painted rheology / ``fac_zone`` keys should use, never the whole
        band; and ``band`` — the union band mask, for a band-confined
        yield or a structural patch key.

    Notes
    -----
    No intermediate (2:1) level is built: the measured tail of choice at
    rig proportions is ``[L0, L1] + finest`` (the no-mid economics). A
    nested mid band, when wanted, subsamples the same extended
    parametrisation — ``(S[::2], reach[::2])`` patches — by hand.
    """
    from underworld3 import discretisation

    if spines is None:
        spines = [(label, np.asarray(P, dtype=float)) for label, P in traces]
    if np.ndim(margin_rings) == 0:
        margin_rings = [(int(margin_rings), int(margin_rings))] * len(spines)
    margin_rings = [(int(a), int(b)) for a, b in margin_rings]
    if len(margin_rings) != len(spines) or min(min(m) for m in margin_rings) < 1:
        raise ValueError(
            "margin_rings must be >= 1 at every end of every trace: the "
            "split cannot reach the band rim (the tip rule); the margin is "
            "extrapolated surround.")
    if mesher not in ("ladder", "network"):
        raise ValueError(
            f"mesher must be 'ladder' or 'network', not {mesher!r}")
    labels = [label for label, _P in traces]
    if len(set(labels)) != len(labels):
        raise ValueError(
            f"trace labels must be unique (each becomes a boundary); got "
            f"{labels}")
    if spines is None:
        spines = [(label, np.asarray(P, dtype=float)) for label, P in traces]
    dm = base_mesh.dm
    spacing_all, rungs_all, extended = [], [], []
    for label, P in spines:
        P = np.asarray(P, dtype=float)
        if P.ndim != 2 or P.shape[1] != 2 or len(P) < 3:
            raise ValueError(
                f"trace {label!r}: expected an (n, 2) polyline with "
                f"n >= 3, got shape {P.shape}")
        extended.append(_extend_polyline_2d(P, margin_rings[len(extended)]))
        spacing_all.append(
            float(np.linalg.norm(np.diff(P, axis=0), axis=1).mean()))
        rungs_all.append(len(P))

    if mesher == "network":
        # ONE placement call for the whole network: the ribbons are fused
        # in CAD, so touching strands and shared bands are ordinary cells
        # of the union, and every spine is EMBEDDED, so the cut below
        # walks its own vertices (#595). The fused band is one region and
        # therefore carries one label value.
        dm, _info = place_thin_volume(
            dm, extended, width, label=band_label, label_value=band_value,
            clearance=clearance, size=float(np.mean(spacing_all)),
            mesher="network", verbose=verbose, seams=seams,
            ligament=ligament, grading=grading)
    else:
        if seams != "gather":
            raise ValueError(
                "seams='ligament' needs mesher='network' (the spines must "
                "be embedded for the label-only cut)")
        for k, S in enumerate(extended):
            # the ladder places strands one after another, and a later
            # cavity may not take a cell of an earlier band's skin: the
            # linear fill keeps that room between close strands (the
            # harder grading is the network mesher's, one placement)
            dm, _info = place_thin_volume(
                dm, [(S, _mitred_reach_2d(S))], width, label=band_label,
                label_value=band_value + k, clearance=clearance,
                size=spacing_all[k], mesher="ladder", verbose=verbose,
                grading=1.0)

    mesh = discretisation.Mesh(
        dm, simplex=True, qdegree=base_mesh.qdegree,
        coordinate_system_type=base_mesh.CoordinateSystem.coordinate_type,
        boundaries=base_mesh.boundaries, verbose=False)
    # The placed mesh OWNS the base's multigrid tail (the coarse levels do
    # not need the fault): every solver on it drives FMG automatically,
    # and the band label is its FAC patch key for volumetric rheologies.
    from underworld3.utilities.custom_mg import adopt_hierarchy
    band_all = np.zeros(int(mesh.dm.getHeightStratum(0)[1]), dtype=bool)
    for k in range(len(spines)):
        band_all |= mesh.cells_labelled(band_label, band_value + k)
    adopt_hierarchy(mesh, base_mesh, fac_zone=band_all)
    if split:
        # ONE network call — cut all, then split all (chained add_fault
        # calls do not compose: each split re-derives the pairing records
        # and drops the earlier fault's).
        # Under the seam ligament the spines are already the mesh's edges
        # and the ligament is deliberately uncut: label, do not cut.
        mesh = mesh.add_fault([(label, np.asarray(P, dtype=float))
                               for label, P in traces],
                              cut=(seams == "gather"),
                              exclude=(band_label + "_ligament"
                                       if seams != "gather" else None))
        mesh._custom_mg_fac_zone = None     # a split fault needs no patch

    # Per-strand FAULT FOOTPRINT masks (the honoured-paint rule): band
    # cells whose nearest extended sample is one of THAT trace's own
    # vertices. This is the mask a volumetric rheology (or a fac_zone
    # key) should use — never the whole band, whose margin is
    # extrapolated surround (and, on a shared spine, whose gap edges
    # belong to no trace). A band cell belongs to the trace whose vertex
    # is nearest to it, read off the CONCATENATED spine samples.
    band = np.zeros_like(band_all)
    for k in range(len(spines)):
        band |= mesh.cells_labelled(band_label, band_value + k)
    S_all = np.vstack(extended)
    scale = 1e-9 * float(np.mean(spacing_all))
    footprints = {}
    for label, P in traces:
        P = np.asarray(P, dtype=float)
        d = S_all[:, None, :] - P[None, :, :]
        is_user = (np.einsum("ijk,ijk->ij", d, d).min(axis=1) < scale ** 2)
        if is_user.sum() != len(P):
            raise ValueError(
                f"trace {label!r}: {int(is_user.sum())} of its {len(P)} "
                f"vertices lie on a spine; every trace vertex must be a "
                f"spine vertex")
        footprints[label] = _footprint_from_samples(
            mesh.dm, band, S_all, is_user)

    # The seam ligament's cells (empty under the gather): the band material
    # the fault is NOT cut through, for apply() to paint the weak plane on.
    ligament_mask = mesh.cells_labelled(band_label + "_ligament")
    info = {"n_cells": int(mesh.dm.getHeightStratum(0)[1]),
            "spacing": spacing_all, "n_rungs": rungs_all,
            "footprints": footprints, "band": band, "mesher": mesher,
            "extended": extended, "width": float(width),
            "margin_rings": margin_rings,
            "spines": [label for label, _P in spines],
            "seams": seams, "ligament": ligament_mask,
            "n_ligament_cells": int(_info.get("n_ligament_cells", 0))
            if mesher == "network" else 0}
    if verbose:
        import underworld3 as _uw
        _uw.pprint(f"[place_fault_ribbon_2d] {info['n_cells']} cells, "
                   f"{len(traces)} fault(s)"
                   f"{' (split)' if split else ''}")
    return mesh, info
