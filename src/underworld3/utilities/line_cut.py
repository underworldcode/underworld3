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

**What the slivers actually cost.** Measured with GAMG on a Poisson solve, which
reads the operator and so is sensitive to element shape (the geometric hierarchy
is deliberately not — it sat at 2-3 V-cycles across every mesh here and cannot
discriminate). On a 5,432-cell box, CG iterations to ``rtol=1e-10``:

===========  ===========  ==========
``snap_frac``  min angle    CG iters
===========  ===========  ==========
uncut          43.7 deg     20
0.00            0.6 deg     32
0.05            2.7 deg     28
0.10            6.7 deg     23
0.20           11.3 deg     21
===========  ===========  ==========

So cutting without snapping costs 60 % more iterations, and snapping buys it back.
A Lawson flip pass (:func:`~underworld3.utilities.reconnect.flip_to_reduce_max_angle`,
which locks the cut automatically because :data:`CUT_LABEL` is an edge label)
helps less than snapping does — 32 to 29 at ``snap_frac=0``, 28 to 25 at 0.05 —
so raising the snap fraction is the better lever, and repair is a second-order
touch-up rather than a requirement.

Scope
-----
Two dimensions, and lines that cross the mesh from boundary to boundary. A line
*ending* inside the mesh (a fault tip) leaves a triangle the line enters but does
not leave, which bisects without cutting; that is refused rather than silently
mis-meshed, as are triangles crossed three times.
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


def _resolve_snapping(dm, X, ends, lines, snap_frac):
    """Which vertices to move onto the line, and where the crossings then land.

    A crossing at parameter ``t`` on an edge sits ``t`` of the way along it, so
    ``t < snap_frac`` means the split would carve off a sliver whose short side is
    that fraction of an edge. In that case the edge's endpoint is moved onto the
    line instead and the edge is left whole.

    Snapping a vertex changes the crossings on every edge that touches it, which
    can bring a further crossing close to a vertex, so this repeats until the set
    settles. It settles quickly — each round only ever adds vertices, and the mesh
    is finite — but the loop is capped rather than trusted.

    The chosen set is reconciled over the point star-forest EVERY round. The
    decision "this crossing is too close to that end" is read off one edge, and a
    rank holding only one side of a shared vertex can decide differently from its
    neighbour. The vertex then moves on one rank and not the other, the two ranks
    disagree about which edges are crossed, and the caller's split loop never
    empties its crossing set — measured, at np=3, as a cut that converged at
    snap_frac=0 and never converged at snap_frac=0.1.
    """
    on_line = np.zeros(len(X), dtype=bool)
    for _ in range(10):
        X_snapped = X.copy()
        if on_line.any():
            X_snapped[on_line] = _project_onto_lines(X[on_line], lines)

        t, multiply_crossed = _crossing_parameters(X_snapped, ends, lines, on_line)
        near = np.isfinite(t) & ((t < snap_frac) | (t > 1.0 - snap_frac))

        # The end the crossing is nearest is the one that would form the sliver.
        rows = np.flatnonzero(near)
        pick = ends[rows, np.where(t[rows] < 0.5, 0, 1)]
        proposed = on_line.copy()
        proposed[pick] = True

        # COLLECTIVE, so every rank must reach it — including one that proposes
        # nothing. An early `if not near.any(): return` here deadlocked at np=3:
        # the rank owning no part of the line walked out while its peers waited
        # in the reduce.
        vS, _vE = dm.getDepthStratum(0)
        pStart, pEnd = dm.getChart()
        flag = np.zeros(pEnd - pStart, dtype=np.int32)
        flag[np.flatnonzero(proposed) + vS - pStart] = 1
        _sf_logical_or(dm, flag)
        proposed = flag[np.arange(len(X)) + vS - pStart] == 1

        # Settled is a GLOBAL property: one rank still moving means another round
        # for everyone, or the reconcile above goes unmatched.
        settled = uw.mpi.comm.allreduce(
            int(np.array_equal(proposed, on_line)), op=MPI.MIN)
        if settled:
            return on_line, X_snapped, t, multiply_crossed
        on_line = proposed

    raise RuntimeError(
        "snapping did not settle in 10 rounds; snap_frac is large enough that "
        "moving one vertex keeps dragging the next crossing into tolerance.")


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


def _child_vertex_of(parent, child, positions):
    """Child vertex nearest each given position, insisting the match is exact.

    Parent vertices keep their coordinates through the transform and inserted
    vertices land on their parent edge's midpoint, so position identifies both.
    petsc4py does not expose ``DMPlexTransformGetTargetPoint``, so this is the
    available route, and matching on geometry keeps it independent of the
    transform's internal point numbering.
    """
    Xc = _coords(child)
    vS, vE = child.getDepthStratum(0)
    tree = uw.kdtree.KDTree(np.ascontiguousarray(Xc[: vE - vS]))
    idx, dist_sqr, found = tree.find_closest_point(np.ascontiguousarray(positions))

    scale = np.ptp(_coords(parent), axis=0).max()
    bad = np.flatnonzero(~np.asarray(found).ravel()
                         | (np.asarray(dist_sqr).ravel() > (1e-9 * scale) ** 2))
    if len(bad):
        raise RuntimeError(
            f"{len(bad)} expected vertex position(s) have no child vertex; the "
            "transform did not place points where this routine assumes it does.")
    return np.asarray(idx, dtype=np.int64).ravel()


def _set_coordinates(dm, indices, values):
    """Move a set of vertices. The coordinate vector is written whole."""
    vec = dm.getCoordinatesLocal()
    arr = np.asarray(vec.array).reshape(-1, dm.getCoordinateDim()).copy()
    arr[indices] = values
    new = vec.duplicate()
    new.array[:] = arr.reshape(-1)
    dm.setCoordinatesLocal(new)


def _label_cut_edges(dm, lines, tol, name, value):
    """Mark the edges lying along the cut.

    An edge is on the cut when both its endpoints and its midpoint lie on a line.
    The midpoint test is what distinguishes the cut from a chord: where a polyline
    turns, two vertices on different segments can be joined by an edge that is not
    part of the line at all.
    """
    X = _coords(dm)
    on = _distance_to_lines(X, lines) < tol
    eS, eE = dm.getDepthStratum(1)
    ends = _edge_vertices(dm)
    mid_on = _distance_to_lines(0.5 * (X[ends[:, 0]] + X[ends[:, 1]]), lines) < tol
    keep = on[ends[:, 0]] & on[ends[:, 1]] & mid_on

    if not dm.hasLabel(name):
        dm.createLabel(name)
    label = dm.getLabel(name)
    label.setDefaultValue(0)
    for e in np.flatnonzero(keep) + eS:
        label.setValue(int(e), int(value))
    del eE
    return int(keep.sum())


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


def cut_along_lines(dm, lines, snap_frac=0.10, label=CUT_LABEL, label_value=1):
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
        ``n_split`` edges split, ``n_snapped`` vertices moved onto a line,
        ``n_cut_edges`` edges labelled, ``min_area`` and ``min_angle`` of the
        result.

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
    >>> cut, info = cut_along_lines(mesh.dm, [np.array([[0.5, -0.1], [0.5, 1.1]])])
    >>> info["n_cut_edges"] == info["n_split"] + info["n_snapped"] - 1
    True
    """
    if dm.getDimension() != 2:
        raise ValueError(
            f"cut_along_lines is 2-D; this mesh is {dm.getDimension()}-D. Cutting "
            "tetrahedra along a surface is a different pattern problem.")

    from underworld3.utilities import _nvb_transform  # noqa: F401  (registers the type)

    X = _coords(dm)
    ends = _edge_vertices(dm)

    on_line, X_snapped, t, multiply_crossed = _resolve_snapping(
        dm, X, ends, lines, snap_frac)
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
    # must not take a different branch from one that does.
    totals = np.array([len(crossed), int(on_line.sum())], dtype=np.int64)
    n_crossed_total, n_snapped = uw.mpi.comm.allreduce(totals, op=MPI.SUM)
    if n_crossed_total == 0 and n_snapped == 0:
        raise ValueError("no mesh edge is crossed by any line: nothing to cut.")

    # Apply the snapping to a WORKING COPY. The caller's mesh is never touched, so
    # a line can be moved and re-cut against the same fixed base.
    work = dm.clone()
    if on_line.any():
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
        scale = np.ptp(X_now, axis=0).max()
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
        if len(chosen):
            ce = ends_now[chosen - eS_now]
            tc = t_now[chosen - eS_now][:, None]
            midpoints = 0.5 * (X_now[ce[:, 0]] + X_now[ce[:, 1]])
            targets = _child_vertex_of(cut, child, midpoints)
            _set_coordinates(child, targets,
                             (1.0 - tc) * X_now[ce[:, 0]] + tc * X_now[ce[:, 1]])
        cut = child
    else:
        raise RuntimeError(
            "the cut did not converge in 12 passes; every pass must split at "
            "least one edge and remove it from the crossing set.")

    areas = cell_areas(cut)
    if (areas <= 0.0).any():
        raise RuntimeError(
            f"snapping inverted {int((areas <= 0).sum())} cell(s); snap_frac="
            f"{snap_frac} is too large for this mesh.")

    n_cut_edges = _label_cut_edges(cut, lines, 1e-9 * np.ptp(X, axis=0).max(),
                                   label, label_value)
    return cut, {
        "n_split": n_split,
        "n_snapped": n_snapped,
        "n_cut_edges": n_cut_edges,
        "min_area": float(areas.min()),
        "min_angle": float(min_angles(cut).min()),
    }


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
