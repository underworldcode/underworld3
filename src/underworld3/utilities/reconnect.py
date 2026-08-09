"""Reconnection: repair the element shapes a refinement pass leaves behind (2-D).

Refinement engines choose *where* to put a new vertex; they do not get to choose
how the surrounding cells reconnect. :mod:`underworld3.utilities.edge_split`
splits an edge in every cell incident on it, so a cell that nominated that edge
gains two well-shaped children while a cell dragged along — split at an edge it
did not nominate — gains a thin one. Reconnection is the missing third operation
of the classical refine / swap / smooth triple: UW3 has refine (:meth:`Mesh.adapt`)
and smooth (:meth:`Mesh.relax`), and this is swap.

Two passes live here. :func:`flip_to_reduce_max_angle` changes *connectivity* and
keeps every point; :func:`remove_vertices` deletes a point and retriangulates the
hole. They fix different damage and they compose — in one order only. See
`Deleting a vertex`_.

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

.. _Deleting a vertex:

Deleting a vertex
-----------------
A conforming cut (:mod:`underworld3.utilities.line_cut`) has only two primitives:
**snap** a vertex onto the surface, or **split** an edge the surface crosses.
Every sliver it leaves follows from that — a crossing falling near a vertex must
either drag the vertex to it or carve a thin cell beside it, and tightening the
snap tolerance only trades one for the other. **Delete** is the missing third. It
dissolves the case rather than trading it, and it is the only one of the three
that *removes* work instead of adding it.

Measured on a box fault cut into an adapted mesh, counting cells whose smallest
angle is under 15 degrees:

===========================  =====  ========  ==========
pass                         cells  < 15 deg  max angle
===========================  =====  ========  ==========
the cut                       4548        60      148.45
flip                          4548        18      133.27
flip, then delete             4306         4      133.27
delete, then flip             4076        60      138.81
===========================  =====  ========  ==========

Two things to read off that. The order is **not** symmetric: flipping first and
deleting second removes 242 cells *and* takes the sliver count from 60 to 4,
while deleting first leaves it at 60. Deletion retriangulates a cavity from the
point set it is given, so running it on connectivity the flip pass has not yet
cleaned up spends its independent set on cavities that a flip would have fixed
for free — and the cavity, once ear-clipped, no longer presents the quad the flip
pass was looking for. Flip is cheap and reversible; delete is neither. Flip
first. A second round of each then finds nothing, so the pair converges.

The second is that the acceptance test needs both shape measures, unlike the flip
pass. Gating on the largest angle alone — correct for flipping, since the P1
interpolation bound depends on it — let the minimum angle fall from 10.80 to
10.23 degrees and *raised* the sliver count from 60 to 61, because a needle has
one tiny angle and two close to 90 and never registers as obtuse. Hence
``gate="both"``.

Where deletion is *wanted* is not decided here: :func:`remove_vertices` takes a
candidate list, and refuses whatever it cannot improve. Offered every vertex of a
clean mesh it does nothing, which is the behaviour a pass that removes degrees of
freedom has to have.

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

Deletion freezes the same seam, for a stronger reason: it compacts the point
chart, so unlike a flip it cannot hand the star-forest across verbatim. Every
point after a deleted one shifts, and each leaf's *remote* index is a number only
its owner holds. :func:`rebuild_cavities` renumbers locally and
broadcasts the new numbering root-to-leaf **once** to close that gap. One
exchange of bookkeeping over the existing partition — no cell changes rank and
nothing is redistributed, which is the property an external remesher costs us.
Freezing the seam is then what keeps the *leaf set* itself unchanged, so the
forest need only be renumbered and never rebuilt. Measured cost at np=2..4 on a
cut mesh: 113-115 deletions against 121 serial.

Status
------
2-D only, both passes. A deleted vertex's cavity is a polygon in 2-D and a
polyhedron in 3-D, and ear clipping does not generalise to one — though cavity
insertion is standard practice in 3-D meshing where ad-hoc cutting of tets along
a surface is not, so this is the operation that generalises *better* than the cut
it replaces.

In 3-D no single flip suffices either: the operator set has to become
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

#: Labels that partition the CELLS but carry no material meaning, so an edge
#: between two cells holding different values of one is not an interface.
#:
#: ``uwnvb_refedge`` is the newest-vertex transform's per-triangle *slot*: which
#: of a cell's edges is its refinement edge. It takes values 0/1/2 across any
#: NVB-adapted mesh, which :func:`_cell_regions` read as three material regions —
#: 2230/2184/134 cells on one measured fault mesh — and duly locked every edge
#: between them. That silently disabled repair over most of ANY adapted mesh: of
#: the edges around a sliver in a cut mesh, 113 were declined as a "region
#: interface" against 54 genuinely locked on the fault.
#:
#: This is the same trap :func:`_interface_edges` documents for ``Elements``, one
#: level along. ``Elements`` does not trip :func:`_cell_regions` only because it
#: is uniform; a bookkeeping label that VARIES does.
_BOOKKEEPING_LABELS = ("uwnvb_refedge",)

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


def _largest_cosine(triangles):
    """The largest cosine of any interior angle across ``triangles``.

    The companion of :func:`_smallest_cosine`, and a monotone stand-in for "the
    smallest angle" by the same argument. The two measure different failures and
    a pass that watches only one is blind to the other: an obtuse cell has a
    cosine near ``-1`` and a *needle* — one tiny angle and two near right angles
    — has one near ``+1`` while its largest angle stays close to 90 degrees, so
    it slips past a maximum-angle test untouched.
    """
    worst = -1.0
    for P in triangles:
        for i in range(3):
            u = P[(i + 1) % 3] - P[i]
            v = P[(i + 2) % 3] - P[i]
            denom = np.hypot(u[0], u[1]) * np.hypot(v[0], v[1])
            if denom == 0.0:
                return 1.0
            worst = max(worst, float((u[0] * v[0] + u[1] * v[1]) / denom))
    return worst


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


def _interface_edges(dm):
    """Chart-indexed flags for the **edges** of an interface label.

    A labelled interior edge is an interface — a named boundary, or a registered
    surface — and must never be flipped, nor be dissolved by deleting one of its
    end points, since that is what protects a fault or a material boundary from
    being reconnected across.

    Two whole strata are ignored, and both exclusions are load-bearing rather
    than fastidious.

    A label value carried by a **cell** is a *volume* and not an interface.
    ``Elements`` labels every cell of a gmsh mesh, and the ``uwnvb_bisect``
    transform propagates a parent's labels to its children, so after refinement
    every new *interior edge* carries ``Elements`` as well. Treating any labelled
    point as an interface therefore locked 81 % of the interior edges of a plain
    refined box, and repair quietly did almost nothing on every real UW3 mesh —
    while hand-built fixtures, which have no such label, kept working.

    **Vertices** are ignored because in 2-D an interface is a *curve*, so it is
    identified by the edges that make it up; a vertex on one always carries
    interface edges too, and reading the vertices adds nothing but noise. It adds
    a great deal of noise: ``Null_Boundary`` marks **every vertex of every UW3
    mesh** with the reserved value 666 — the sentinel a natural boundary
    condition attaches to when it applies to no boundary — and ``UW_Boundaries``
    re-packs every per-boundary stratum, sentinel included, into one stacked
    label. Between them every vertex in the chart is labelled. That costs the
    flip pass nothing, since it asks only about edges, but a vertex-level test
    built on it refuses every candidate it is ever offered. Measured: 1114 of
    1114 on a cut mesh, before a single one reached a shape test.

    Over-locking is safe in the sense that it cannot corrupt a mesh, but not in
    the sense that matters: it disables the feature silently. Third instance of
    the same trap, after ``Elements`` here and ``uwnvb_refedge`` in
    :data:`_BOOKKEEPING_LABELS`. A label's *presence* on a point says nothing
    about whether it means a material interface.

    A region *join* is handled separately, by :func:`_cell_regions`, which
    compares the two cells rather than reading the edge.
    """
    pStart, pEnd = dm.getChart()
    cS, cE = dm.getHeightStratum(0)
    eS, eE = dm.getDepthStratum(1)
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
            flag[idx[(idx >= eS) & (idx < eE)] - pStart] = True
    return flag


def _cell_regions(dm):
    """Per-cell label signature, or ``None`` when every cell carries the same one.

    An edge between two cells with different region values is a material
    interface even when the edge itself is unlabelled, so the signature is what
    lets those edges be locked. Built from label strata rather than a per-cell
    query, which would be one PETSc call per cell per label.

    :data:`_BOOKKEEPING_LABELS` is excluded: a label may partition the cells for
    reasons that have nothing to do with material, and treating one of those as a
    region locks most of the mesh against repair without saying so.
    """
    cS, cE = dm.getHeightStratum(0)
    names = [dm.getLabelName(i) for i in range(dm.getNumLabels())
             if dm.getLabelName(i) not in _TOPOLOGY_LABELS
             and dm.getLabelName(i) not in _BOOKKEEPING_LABELS]
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

def _write_coordinates(new, cdim, vertex_range, values):
    """Give ``new`` a local vertex coordinate section holding ``values``.

    ``vertex_range`` is the new mesh's vertex stratum and ``values`` is one row
    per vertex of it, in point order — the source's own rows for a rebuild that
    preserves the numbering, the survivors' rows for one that compacts the
    chart, and the survivors followed by the newly placed points for one that
    grows it. Written through a section rather than ``setCoordinates`` because
    this is purely local data and the latter wants a global vector.
    """
    vS, vE = vertex_range
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
    coords.array[:] = np.asarray(values, dtype=float).reshape(-1)
    new.setCoordinatesLocal(coords)


def _copy_labels(new, dm, point_map=None):
    """Copy every non-topology label across, by point id.

    ``point_map`` is chart-indexed and may be ``None`` for an unchanged chart.
    A point mapped to a negative entry has been deleted and its label value goes
    with it. No coordinate matching is involved anywhere, which is the whole
    reason both rebuilds work in terms of a point map.
    """
    pStart, _pEnd = dm.getChart()
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
                q = int(p) if point_map is None else int(point_map[int(p)
                                                                  - pStart])
                if q >= 0:
                    target.setValue(q, int(val))


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
    _write_coordinates(new, dm.getCoordinateDim(), (vS, vE), _coords(dm)[:vE - vS])
    _copy_labels(new, dm)

    # The star-forest transfers verbatim: every rank preserves its numbering, so
    # the remote point numbers it carries are still the right ones.
    if uw.mpi.size > 1:
        _install_point_sf(new, dm.getPointSF())
    return new


def rebuild_cavities(dm, victims, drop_cells, new_cells, placed_coords=()):
    """Build a fresh plex with cells replaced by a retriangulation of their cavity.

    The one rebuild that changes the point chart, in both directions: vertices
    may be **deleted** (a repair pass dissolving a bad point) and vertices may be
    **placed** (an embedded surface asserting its own points). A caller doing
    only one of the two passes an empty list for the other.

    Parameters
    ----------
    dm : PETSc.DMPlex
        Source mesh. Not modified.
    victims : sequence of int
        Vertex points to delete.
    drop_cells : sequence of int
        Cell points to delete — every cell of the cavity being retriangulated.
    new_cells : sequence of tuple
        Replacement cells as vertex triples. A **non-negative** entry is a point
        number in the source mesh; a **negative** entry ``-(k + 1)`` is row ``k``
        of ``placed_coords``, a vertex this call creates. Winding is fixed here,
        so the caller need not order the triples.
    placed_coords : array_like
        ``(n, cdim)`` coordinates of the vertices being added, or empty.

    Returns
    -------
    new : PETSc.DMPlex
        The rebuilt mesh, on a re-numbered chart.
    point_map : numpy.ndarray
        Chart-indexed source point -> new point, ``-1`` for a deleted point.
    placed_points : numpy.ndarray
        New point number of each row of ``placed_coords``.

    Notes
    -----
    This is :func:`rebuild_with_cones` with its one restriction lifted. A flip
    adds and removes no points, so that function can preserve the numbering and
    hand the star-forest across verbatim. A change of chart cannot: every point
    after a deleted one shifts, and a placed one has no source number at all. So
    the numbering is rebuilt, and with it the edges — which are not given, but
    **derived from the new cells**, since an edge of the retriangulated cavity
    may be an old edge that survived or a chord the retriangulation invented and
    there is no way to tell them apart except by looking.

    Ordering is by source point number for everything that survives, then by
    caller order for the placed vertices, and by vertex tuple for the new cells,
    so the result is a function of the input topology and not of the order the
    caller happened to accumulate it in.

    The parallel cost is one exchange. Renumbering the star-forest's *local*
    indices is local knowledge, but the *remote* index of each leaf is the
    owner's new number for that point, which only the owner knows — so the new
    numbering is broadcast root-to-leaf once and read back off the leaves. That
    is bookkeeping over the existing partition; no cell moves rank, and nothing
    is redistributed.

    .. warning::

       A **placed** vertex has no remote number anywhere, so the one-exchange
       renumbering does not cover it: placing points across a partition seam
       needs the leaf set itself extended, which is the chart-EXPANSION rebuild
       and is not this function. Deleting is safe in parallel because
       :func:`remove_vertices` never offers a shared point; placing is currently
       refused before it gets here.

    The caller is responsible for never deleting a point that the star-forest
    touches (see :func:`remove_vertices`), which is what lets the leaf set carry
    across unchanged rather than having to be recomputed.
    """
    pStart, pEnd = dm.getChart()
    cS, cE = dm.getHeightStratum(0)
    vS, vE = dm.getDepthStratum(0)
    eS, eE = dm.getDepthStratum(1)
    cdim = dm.getCoordinateDim()

    dead_v = np.zeros(vE - vS, dtype=bool)
    dead_v[np.asarray(victims, dtype=np.int64) - vS] = True
    dead_c = np.zeros(cE - cS, dtype=bool)
    dead_c[np.asarray(drop_cells, dtype=np.int64) - cS] = True

    surv_v = np.flatnonzero(~dead_v) + vS
    surv_c = np.flatnonzero(~dead_c) + cS
    placed = np.asarray(placed_coords, dtype=float).reshape(-1, cdim)

    # Cells, in the source numbering, as vertex triples: survivors keep their
    # relative order, the replacements follow sorted by vertex tuple. Every
    # triple is turned anticlockwise here, because the cone orientations below
    # are derived from the traversal and a clockwise cell would be wired to a
    # negative volume without raising.
    X = _coords(dm)

    def at(v):
        """Coordinates of a mixed index: a source point, or a placed row."""
        return placed[-int(v) - 1] if v < 0 else X[int(v) - vS]

    def anticlockwise(tri):
        a, b, c = (int(v) for v in tri)
        if _orient2d(at(a), at(b), at(c)) < 0:
            return (a, c, b)
        return (a, b, c)

    kept = []
    for c in surv_c:
        closure = np.asarray(dm.getTransitiveClosure(int(c))[0], dtype=np.int64)
        kept.append(anticlockwise([p for p in closure if vS <= p < vE]))
    made = sorted((anticlockwise(tri) for tri in new_cells),
                  key=lambda tri: tuple(sorted(tri)))
    cells = kept + made

    # Edges are whatever the cells ask for. An old edge is reused when its pair
    # is still wanted, which keeps its labels; the rest are new.
    wanted = set()
    for tri in cells:
        a, b, c = tri
        wanted.update(((a, b) if a < b else (b, a),
                       (b, c) if b < c else (c, b),
                       (c, a) if c < a else (a, c)))
    surv_e, pair_of = [], {}
    for e in range(eS, eE):
        a, b = (int(v) for v in dm.getCone(e))
        pair = (a, b) if a < b else (b, a)
        if pair in wanted:
            surv_e.append(e)
            pair_of[e] = pair
    edges = [pair_of[e] for e in surv_e] + sorted(wanted
                                                  - {pair_of[e]
                                                     for e in surv_e})

    # Strata keep the source's relative order; only their sizes change. The
    # placed vertices sit at the END of the vertex stratum, after the survivors,
    # so a source vertex's new number depends only on how many vertices before
    # it died and not on how many were placed.
    sizes = {"c": len(cells), "v": len(surv_v) + len(placed), "e": len(edges)}
    offset, next_point = {}, pStart
    for _start, key in sorted(((cS, "c"), (vS, "v"), (eS, "e"))):
        offset[key] = next_point
        next_point += sizes[key]

    point_map = np.full(pEnd - pStart, -1, dtype=np.int64)
    point_map[surv_c - pStart] = offset["c"] + np.arange(len(surv_c))
    point_map[surv_v - pStart] = offset["v"] + np.arange(len(surv_v))
    point_map[np.asarray(surv_e, dtype=np.int64) - pStart] = (
        offset["e"] + np.arange(len(surv_e)))
    placed_points = offset["v"] + len(surv_v) + np.arange(len(placed))

    def v_new(v):
        return (int(placed_points[-int(v) - 1]) if v < 0
                else int(point_map[int(v) - pStart]))

    new = PETSc.DMPlex().create(comm=dm.comm)
    new.setDimension(dm.getDimension())
    new.setChart(pStart, next_point)
    for i in range(len(cells)):
        new.setConeSize(offset["c"] + i, 3)
    for i in range(len(edges)):
        new.setConeSize(offset["e"] + i, 2)
    new.setUp()

    edge_of, first_of = {}, {}
    for i, (a, b) in enumerate(edges):
        e = offset["e"] + i
        new.setCone(e, [v_new(a), v_new(b)])
        edge_of[(a, b)] = e
        first_of[e] = a

    for i, (v0, v1, v2) in enumerate(cells):
        cone, orientation = [], []
        for x, y in ((v0, v1), (v1, v2), (v2, v0)):
            e = edge_of[(x, y) if x < y else (y, x)]
            cone.append(e)
            orientation.append(0 if first_of[e] == x else -1)
        new.setCone(offset["c"] + i, cone)
        new.setConeOrientation(offset["c"] + i, orientation)

    new.symmetrize()
    new.stratify()

    _write_coordinates(new, cdim, (offset["v"], offset["v"] + sizes["v"]),
                       np.vstack([X[surv_v - vS], placed.reshape(-1, cdim)]))
    _copy_labels(new, dm, point_map)

    if uw.mpi.size > 1:
        _rebuild_point_sf(new, dm, point_map, next_point - pStart)
    return new, point_map, placed_points


def _install_point_sf(new, new_sf):
    """Install a rebuilt star-forest on the plex AND its coordinate DM.

    The coordinate DM is created when the rebuilt chart's coordinates
    are written — BEFORE any point SF exists — so it snapshots an empty
    one. The solve never notices (field sections are created later,
    from the plex SF), but a parallel HDF5 save then writes every
    shared vertex as owned on every rank, and the serial reload of such
    a checkpoint dies in ``coordinatesLoad`` (measured: a split mesh
    written at np = 4 carried owned+shared = 7121 vertex rows where the
    true owned count is 6334).
    """
    new.setPointSF(new_sf)
    new.getCoordinateDM().setPointSF(new_sf)


def _install_point_sf(new, new_sf):
    """Install a rebuilt star-forest on the plex AND its coordinate DM.

    The coordinate DM is created when the rebuilt chart's coordinates
    are written — BEFORE any point SF exists — so it snapshots an empty
    one. The solve never notices (field sections are created later,
    from the plex SF), but a parallel HDF5 save then writes every
    shared vertex as owned on every rank, and the serial reload of such
    a checkpoint dies in ``coordinatesLoad`` (measured: a split mesh
    written at np = 4 carried owned+shared = 7121 vertex rows where the
    true owned count is 6334).
    """
    new.setPointSF(new_sf)
    new.getCoordinateDM().setPointSF(new_sf)


def _rebuild_point_sf(new, dm, point_map, nroots):
    """Carry the point star-forest onto a renumbered chart, in one exchange.

    A leaf's remote entry names the *owner's* local index for the shared point,
    so renumbering it needs a number this rank does not hold. Broadcasting the
    new numbering root-to-leaf over the star-forest that is being replaced
    delivers exactly that, one value per leaf.

    The leaf set itself is unchanged: :func:`remove_vertices` never deletes a
    shared point, and the split-fault rebuilds never duplicate one (a
    seam-touching fault is redistributed onto one rank before splitting), so
    every leaf still exists and only its number has moved.
    """
    pStart, pEnd = dm.getChart()
    sf = dm.getPointSF()
    try:
        _nroots, ilocal, iremote = sf.getGraph()
    except (ValueError, TypeError):
        return                            # unpopulated: nothing is shared

    root_new = np.ascontiguousarray(point_map, dtype=np.int32)
    leaf_new = np.full(pEnd - pStart, -1, dtype=np.int32)
    # COLLECTIVE, and reached on every rank of a parallel run: a rank sharing
    # nothing still has to participate or its peers block.
    sf.bcastBegin(MPI.INT32_T, root_new, leaf_new, MPI.REPLACE)
    sf.bcastEnd(MPI.INT32_T, root_new, leaf_new, MPI.REPLACE)

    # petsc4py will not narrow an index array for us — the graph must arrive as
    # PETSc's own integer type or `setGraph` raises an unsafe-cast TypeError.
    new_sf = PETSc.SF().create(comm=dm.comm)
    if ilocal is None or not len(ilocal):
        new_sf.setGraph(nroots, np.zeros(0, dtype=PETSc.IntType),
                        np.zeros(0, dtype=PETSc.IntType))
        _install_point_sf(new, new_sf)
        return

    leaves = np.asarray(ilocal, dtype=np.int64)
    local = point_map[leaves - pStart]
    remote_index = leaf_new[leaves - pStart]
    if (local < 0).any() or (remote_index < 0).any():
        raise RuntimeError(
            "reconnect: a shared point was deleted. The removal pass must "
            "freeze the seam; see remove_vertices.")

    remote = np.empty((len(leaves), 2), dtype=PETSc.IntType)
    remote[:, 0] = np.asarray(iremote).reshape(-1, 2)[:, 0]
    remote[:, 1] = remote_index
    new_sf.setGraph(nroots, local.astype(PETSc.IntType), remote.reshape(-1))
    _install_point_sf(new, new_sf)


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
        old = ((Xp, Xa, Xb), (Xa, Xq, Xb))
        new = ((Xp, Xa, Xq), (Xp, Xq, Xb))
        before = _smallest_cosine(old)
        after = _smallest_cosine(new)
        if after <= before + _MIN_GAIN:
            continue                     # no shape gain worth the flip
        # ... and it may not buy that gain by making a NEEDLE. The objective
        # stays the maximum angle, for the Babuska-Aziz reason above; this is
        # only a floor under the other end. Without it the pass is free to
        # trade an obtuse cell for a thin one, whose largest angle is
        # unremarkable and so never registers here — measured on a cut graded
        # mesh, flipping alone took the smallest angle in the mesh DOWN, which
        # is what a caller composing this with a size field will not expect.
        if _largest_cosine(new) > _largest_cosine(old) + _MIN_GAIN:
            continue
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
        candidates = _flippable(dm, X, verts, frozen, _interface_edges(dm),
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


# ------------------------------------------------------------- the removal pass

def _link_ring(cells_of_v, v, verts, cS):
    """The victim's link as a closed anticlockwise ring, or ``None``.

    Each incident cell contributes the directed edge of its link that the cell
    traverses, so following them from any start walks the ring once. Anything
    other than a single closed walk visiting every incident cell — a boundary
    vertex, a non-manifold fan, a vertex reached twice — returns ``None`` and the
    victim is declined. That is the cheapest available test for "the link is a
    simple polygon", which is the one thing the ear-clip below assumes.
    """
    step = {}
    for c in cells_of_v:
        tri = list(verts[c - cS])
        i = tri.index(v)
        step[tri[(i + 1) % 3]] = tri[(i + 2) % 3]
    if len(step) != len(cells_of_v):
        return None
    start = min(step)
    ring, cur = [start], step[start]
    while cur != start:
        if cur not in step or len(ring) > len(step):
            return None
        ring.append(cur)
        cur = step[cur]
    return ring if len(ring) == len(step) else None


def _ear_clip(ring, X, vS):
    """Triangulate a simple polygon, choosing each ear by shape.

    The ear taken is the one whose triangle has the smallest largest angle —
    the same objective the flip pass optimises, and for the same reason, that
    the P1 interpolation bound depends on the maximum angle and not the minimum
    (Babuska-Aziz). Ties break on the ear tip's coordinates.

    Neither the choice nor the order depends on point numbers or on traversal
    order, only on geometry, so two ranks holding the same polygon would produce
    the same triangulation. Nothing here relies on that yet — the removal pass
    freezes the seam — but a pass that did not have the property could never
    have the restriction lifted.

    Returns ``None`` if no ear can be cut, which is what a polygon that is not
    simple looks like from the inside.
    """
    poly = list(ring)
    out = []
    while len(poly) > 3:
        n = len(poly)
        best, best_key = None, None
        for i in range(n):
            a, b, c = poly[(i - 1) % n], poly[i], poly[(i + 1) % n]
            Xa, Xb, Xc = X[a - vS], X[b - vS], X[c - vS]
            if _orient2d(Xa, Xb, Xc) <= 0:
                continue                      # reflex, or unresolved
            # An ear may not contain another vertex of the polygon. The test is
            # inclusive, so a vertex lying ON the candidate ear's long side
            # blocks it: that is exactly the chord which would run along a
            # straight run of the link and leave a vertex stranded inside an
            # edge, and it is how a cut whose flank passes through the link
            # survives this pass intact.
            if any(_orient2d(Xa, Xb, X[p - vS]) >= 0
                   and _orient2d(Xb, Xc, X[p - vS]) >= 0
                   and _orient2d(Xc, Xa, X[p - vS]) >= 0
                   for p in poly if p not in (a, b, c)):
                continue
            key = (-_smallest_cosine(((Xa, Xb, Xc),)), Xb[0], Xb[1])
            if best_key is None or key < best_key:
                best, best_key = (i, a, b, c), key
        if best is None:
            return None
        i, a, b, c = best
        out.append((a, b, c))
        poly.pop(i)
    return out + [tuple(poly)]


def _removable(dm, X, verts, frozen, locked, regions, candidates, gate):
    """Vertices worth deleting, as ``(gain, tie, victim, cells, triangles)``.

    A candidate is declined unless every one of these holds:

    * it is **not shared**, and no cell of its link is — the seam is frozen for
      the same reason the flip pass freezes it, and one level more strictly,
      since a deletion changes the chart and not merely a few cones;
    * no incident edge carries an **interface label**, which is what keeps a cut
      or a named boundary intact. Testing the *edges* rather than the vertex is
      what makes this work at all, in both directions:
      :func:`~underworld3.utilities.line_cut.cut_along_lines` labels the cut's
      edges and not its vertices, so a vertex test would delete a vertex out of
      the middle of a fault and leave a gap in it — while every vertex of every
      UW3 mesh carries a sentinel label, so a vertex test would equally refuse
      every candidate. See :func:`_interface_edges`;
    * every incident edge has **two cells**, so the vertex is interior. A
      boundary vertex would change the domain, and its link is not closed
      anyway;
    * its link cells all lie in the **same region**, so a material join is not
      dissolved;
    * the retriangulation **improves the shape** of the cells it replaces, in
      the sense ``gate`` names.

    The last is a refusal, not a policy: it stops the pass making a mesh worse,
    but it does not decide where deletion is *wanted*. That is the caller's job,
    and it matters, because unlike a flip a deletion removes a degree of freedom
    — run over every vertex it would coarsen wherever the mesh happens to be
    slightly ill-shaped. The intended candidate set is the vertices a conforming
    cut had to distort, which is where the damage is.
    """
    cS, cE = dm.getHeightStratum(0)
    vS, _vE = dm.getDepthStratum(0)
    eS, eE = dm.getDepthStratum(1)
    pStart, _pEnd = dm.getChart()

    out = []
    for v in candidates:
        v = int(v)
        star = np.asarray(dm.getTransitiveClosure(v, useCone=False)[0],
                          dtype=np.int64)
        cells = [int(p) for p in star if cS <= p < cE]
        edges = [int(p) for p in star if eS <= p < eE]
        if any(locked[e - pStart] for e in edges):
            continue
        if any(len(dm.getSupport(e)) != 2 for e in edges):
            continue                          # boundary vertex
        if any(frozen[c - cS] for c in cells):
            continue                          # seam
        if regions is not None and not all(
                np.array_equal(regions[c - cS], regions[cells[0] - cS])
                for c in cells):
            continue

        ring = _link_ring(cells, v, verts, cS)
        if ring is None:
            continue
        tris = _ear_clip(ring, X, vS)
        if tris is None:
            continue

        old = [X[np.asarray(verts[c - cS]) - vS] for c in cells]
        new = [X[np.asarray(t) - vS] for t in tris]
        obtuse = _smallest_cosine(new) - _smallest_cosine(old)
        needle = _largest_cosine(old) - _largest_cosine(new)
        if gate in ("obtuse", "both") and obtuse <= -_MIN_GAIN:
            continue
        if gate in ("needle", "both") and needle <= -_MIN_GAIN:
            continue
        gain = {"obtuse": obtuse, "needle": needle,
                "both": min(obtuse, needle)}[gate]
        if gain <= _MIN_GAIN:
            continue
        out.append((gain, tuple(X[v - vS]), v, cells, tris))

    # Best gain first, as in the flip pass, so that when two candidates share a
    # cell the pass keeps the better of the two rather than whichever the loop
    # reached first. The coordinate tie-break keeps the order geometric.
    out.sort(key=lambda row: (-row[0], row[1]))
    return out


def remove_vertices(dm, candidates, max_passes=3, gate="both"):
    """Delete vertices and retriangulate their links, leaving the seam alone.

    The third mesh primitive. A conforming cut has only two — move a vertex onto
    the surface, or split an edge the surface crosses — and every sliver it
    leaves follows from that: a crossing that falls near a vertex must either
    drag the vertex to it or carve a thin cell beside it. Deleting the vertex
    dissolves the case instead of trading it, and it is the only one of the three
    that *removes* work rather than adding it.

    Parameters
    ----------
    dm : PETSc.DMPlex
        A 2-D simplex mesh. Not modified.
    candidates : sequence of int
        Vertex points offered for deletion. Which vertices to offer is a policy
        decision and deliberately not made here — see :func:`_removable`.
    max_passes : int
        Cap on passes. Each pass deletes an independent set, so a candidate
        beaten to a shared cell needs another pass to be reconsidered.
    gate : {"both", "obtuse", "needle"}
        Which shape measure a deletion must improve, and may never degrade:
        the cavity's largest angle (``obtuse``), its smallest (``needle``), or
        both. See the notes.

    Returns
    -------
    reduced : PETSc.DMPlex
        A new mesh, or ``dm`` itself if nothing was deleted.
    n_removed : int
        Vertices deleted across all ranks.

    Notes
    -----
    Deletions are applied an independent set at a time — no two victims sharing
    a cell — because two overlapping links would each be retriangulated from a
    stale reading of the other's result.

    The candidate list is carried between passes through the point map the
    rebuild returns, so a caller chooses its vertices once against the mesh it
    was handed rather than having to re-derive them against a renumbered chart.
    """
    if dm.getDimension() != 2:
        raise NotImplementedError(
            "reconnect.remove_vertices is 2-D only. The cavity of a deleted "
            "vertex is a polygon in 2-D and a polyhedron in 3-D, and ear "
            "clipping does not generalise to one.")

    cand = np.unique(np.asarray(candidates, dtype=np.int64))
    total = 0
    for _pass in range(max_passes):
        pStart, _pEnd = dm.getChart()
        cS, _cE = dm.getHeightStratum(0)
        X = _coords(dm)
        verts, frozen = _cell_vertices_and_seam(dm, X, _shared_points(dm))
        plans = _removable(dm, X, verts, frozen, _interface_edges(dm),
                           _cell_regions(dm), cand, gate)

        claimed, victims, drop, made = set(), [], [], []
        for _gain, _tie, v, cells, tris in plans:
            if any(c in claimed for c in cells):
                continue
            claimed.update(cells)
            victims.append(v)
            drop.extend(cells)
            made.extend(tris)

        # COLLECTIVE, and reached on every rank: one with nothing to delete
        # still has to vote or its peers block waiting for it.
        n = uw.mpi.comm.allreduce(len(victims), op=MPI.SUM)
        if n == 0:
            break
        dm, point_map, _placed = rebuild_cavities(dm, victims, drop, made)
        cand = point_map[cand - pStart]
        cand = cand[cand >= 0]
        total += n

    return dm, total
