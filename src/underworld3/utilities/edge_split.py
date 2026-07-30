"""Longest-edge refinement without a conforming closure.

An alternative refinement engine for :meth:`Mesh.adapt`. Where newest-vertex
bisection chooses which edge to split from a combinatorial tagging rule and then
pays a *conforming closure* to repair the hanging nodes that choice creates, this
engine splits the edge the geometry asks for — the longest edge of every cell
that is still coarser than the metric wants — and needs no closure at all,
because splitting an edge divides **every** cell incident on it at the same new
vertex. There is therefore no hanging node to repair, and no
longest-edge-propagation chain: we never require a neighbour to split its *own*
preferred edge.

Two consequences that matter for adaptation:

* refinement does not spread beyond the cells the metric marked, so the refined
  region hugs the feature rather than a bounded halo around it;
* the marking criterion is the cell **diameter**, not :math:`(d!\\,V)^{1/d}`.
  For bisection the two shrink together and either will do. For any engine that
  reduces volume without shortening the longest edge they diverge badly — a
  measured factor of 3.2 on a centroid-refined mesh, where the volume proxy
  reports the target as met while the mesh is nowhere near resolved.

The topology, coordinates, labels and parallel star-forest are all handled by the
``uwnvb_bisect`` :c:type:`DMPlexTransform` (see
``docs/developer/design/NVB_GRADED_ADAPT.md``), which is the same primitive the
newest-vertex engine uses for each of its sub-passes. That transform bisects a
set of edges named in a per-edge label and requires them to be **pairwise
independent** — no cell may carry two marked edges in one pass — so a pass here
splits an independent subset and the caller iterates.

Notes
-----
One pass does not necessarily satisfy every marked cell: independence caps how
many edges can be split at once. Drive it in a loop that re-marks from the
current mesh, as :meth:`Mesh.adapt` does.

Status
------
Wired into :meth:`Mesh.adapt` as ``engine="edge_split"``. Validated serial and
parallel in 2-D and 3-D: conforming, refinement confined to the marked region,
and the refined mesh identical at np=1/2/3/4. Tests in
``tests/test_0843_edge_split_adapt.py`` and
``tests/parallel/ptest_0843_edge_split_parallel.py``.

Not yet done: the reconnection (flip) pass that repairs element shape. It is not
expressible as a ``DMPlexTransform`` — a flip's output cells span two parents'
closures, while a transform's children may only reference their own parent's —
so it needs a separate parallel design and is tracked outside this module.
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw

_BISECT_LABEL = "uwnvb_bisect_edges"


def _register_transform():
    """Import the compiled extension that registers ``uwnvb_bisect`` in PETSc."""
    from underworld3.utilities import _nvb_transform  # noqa: F401  (registers on import)


def _edge_lengths(dm):
    """Length of every edge, indexed by ``edge_point - edge_start``."""
    cdim = dm.getCoordinateDim()
    vS, _vE = dm.getDepthStratum(0)
    eS, eE = dm.getDepthStratum(1)
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, cdim)
    ends = np.array([dm.getCone(e) for e in range(eS, eE)], dtype=np.int64) - vS
    d = X[ends[:, 0]] - X[ends[:, 1]]
    return np.sqrt(np.einsum("ij,ij->i", d, d))


def _cell_edges(dm):
    """Edge points of each cell, as a list indexed by ``cell - cell_start``.

    In 2-D a cell's cone is already its edges; in 3-D the cone holds faces, so
    the edges come from the transitive closure filtered to the edge stratum.
    """
    eS, eE = dm.getDepthStratum(1)
    cS, cE = dm.getHeightStratum(0)
    if dm.getDimension() == 2:
        return [np.asarray(dm.getCone(c), dtype=np.int64) for c in range(cS, cE)]
    out = []
    for c in range(cS, cE):
        closure = dm.getTransitiveClosure(c)[0]
        out.append(np.array([p for p in closure if eS <= p < eE], dtype=np.int64))
    return out


def cell_diameters(dm):
    """Longest edge length of every cell, in plex cell order.

    This is the quantity the interpolation error of a linear element depends on,
    and the one this engine marks against.
    """
    L = _edge_lengths(dm)
    eS, _eE = dm.getDepthStratum(1)
    return np.array([L[edges - eS].max() for edges in _cell_edges(dm)])


def _sf_logical_or(dm, flag):
    """Logical-OR a point-indexed flag array over the point star-forest, in place.

    Every rank holding a copy of a shared point ends up with the same value, so a
    shared edge chosen for bisection anywhere is split everywhere — the condition
    ``uwnvb_bisect`` needs to keep the child point star-forest conforming.

    For a plex point star-forest the leaf and root spaces are BOTH the local point
    chart, so the SAME array is passed as leaf data and root data. This mirrors
    ``uwnvb_sf_lor`` in ``nvb_transform.c``, which is the proven form. Gathering
    the leaves into a separately-indexed buffer first — the obvious reading of the
    PetscSF signature — mis-sizes the reduce and corrupts the heap.
    """
    # COLLECTIVE, so every rank must reach it: a rank owning no shared point
    # still has to participate or its peers block forever. Only a genuinely
    # serial run may skip, and that is a communicator-size test — never a test
    # of what this rank happens to own.
    if uw.mpi.size == 1:
        return flag
    sf = dm.getPointSF()
    try:
        nroots, _ilocal, _iremote = sf.getGraph()
    except (ValueError, TypeError):
        # An unpopulated star-forest reports a negative root count that petsc4py
        # cannot shape an array from; nothing is shared, so nothing to reconcile.
        return flag
    if nroots < 0:
        return flag

    sf.reduceBegin(MPI.INT32_T, flag, flag, MPI.LOR)
    sf.reduceEnd(MPI.INT32_T, flag, flag, MPI.LOR)
    sf.bcastBegin(MPI.INT32_T, flag, flag, MPI.REPLACE)
    sf.bcastEnd(MPI.INT32_T, flag, flag, MPI.REPLACE)
    return flag


def _owned_count(dm, points):
    """How many of ``points`` this rank owns, i.e. holds as a root not a leaf."""
    if uw.mpi.size == 1:
        return len(points)
    try:
        _nroots, ilocal, _iremote = dm.getPointSF().getGraph()
    except (ValueError, TypeError):
        # Unpopulated star-forest: nothing is shared, so every point is owned.
        return len(points)
    if ilocal is None or len(ilocal) == 0:
        return len(points)
    leaves = set(int(p) for p in ilocal)
    return sum(1 for p in points if int(p) not in leaves)


def _edge_strength(dm):
    """Per-edge sort key making "the strongest candidate in a cell" well defined.

    Length decides; the midpoint coordinate breaks ties. Both are computed from
    the coordinates alone, so the key is identical on every rank holding the edge
    and the selection below is independent of the partition — the property that
    makes the refined mesh the same at any communicator size.
    """
    cdim = dm.getCoordinateDim()
    vS, _vE = dm.getDepthStratum(0)
    eS, eE = dm.getDepthStratum(1)
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, cdim)
    ends = np.array([dm.getCone(e) for e in range(eS, eE)], dtype=np.int64) - vS
    d = X[ends[:, 0]] - X[ends[:, 1]]
    length = np.sqrt(np.einsum("ij,ij->i", d, d))
    mid = 0.5 * (X[ends[:, 0]] + X[ends[:, 1]])
    return length, mid


def _independent_edges(dm, candidates):
    """The candidates that beat every competing candidate sharing a cell.

    This replaces a greedy sweep, which would depend on iteration order and
    therefore on the partition (measured: 414 cells at np=1/2 but 463 at np=3 and
    925 at np=4). A candidate is *vetoed* when a stronger candidate shares one of
    its cells; vetoes are OR-ed across ranks so a shared edge is judged against
    the cells on both sides. What survives is independent by construction — two
    edges in the same cell cannot both beat the other — and is a function of the
    geometry only.
    """
    eS, eE = dm.getDepthStratum(1)
    cS, cE = dm.getHeightStratum(0)
    pStart, pEnd = dm.getChart()

    is_candidate = np.zeros(pEnd - pStart, dtype=np.int32)
    if len(candidates):
        is_candidate[np.asarray(candidates, dtype=np.int64) - pStart] = 1
    _sf_logical_or(dm, is_candidate)

    length, mid = _edge_strength(dm)
    veto = np.zeros(pEnd - pStart, dtype=np.int32)
    edges_of = _cell_edges(dm)
    for c in range(cS, cE):
        edges = edges_of[c - cS]
        rival = edges[is_candidate[edges - pStart] == 1]
        if len(rival) < 2:
            continue
        keys = [(length[e - eS], *mid[e - eS]) for e in rival]
        winner = rival[int(np.lexsort(np.array(keys).T[::-1])[-1])]
        veto[rival[rival != winner] - pStart] = 1
    _sf_logical_or(dm, veto)

    chosen = np.flatnonzero((is_candidate == 1) & (veto == 0)) + pStart
    return chosen[(chosen >= eS) & (chosen < eE)]


def _cells_on_edge(dm, edge):
    """Cells incident on an edge — the star of the vertex a split would insert.

    The walk up from an edge is dimension-dependent and getting it wrong fails
    silently rather than loudly. In 2-D an edge *is* a face, so its support is
    already the cells; in 3-D the support holds faces and the cells are one level
    further up. Applying the 3-D walk in 2-D asks for the support of a cell, which
    is empty, so the function returns no cells at all and every caller reads "this
    edge touches nothing".
    """
    cS, cE = dm.getHeightStratum(0)
    if dm.getDimension() == 2:
        return sorted(int(c) for c in dm.getSupport(edge) if cS <= c < cE)
    seen = set()
    for f in dm.getSupport(edge):
        for c in dm.getSupport(f):
            if cS <= c < cE:
                seen.add(int(c))
    return sorted(seen)


def bisect_longest_edges(dm, cells):
    """Split the longest edge of as many of ``cells`` as one pass allows.

    Parameters
    ----------
    dm : PETSc.DMPlex
        Simplex mesh to refine. Not modified.
    cells : array of int
        Plex cell points to refine.

    Returns
    -------
    refined : PETSc.DMPlex
        A fresh DM, co-partitioned with ``dm`` and carrying its labels forward.
    n_split : int
        Number of edges bisected globally. Zero means the pass was empty and the
        caller should stop.

    Notes
    -----
    Independence caps one pass, so a cell marked here may still exceed the metric
    afterwards. Re-mark from the returned mesh and call again.
    """
    _register_transform()

    cS, _cE = dm.getHeightStratum(0)
    eS, _eE = dm.getDepthStratum(1)
    L = _edge_lengths(dm)
    edges_of = _cell_edges(dm)

    wanted = {int(edges_of[int(c) - cS][np.argmax(L[edges_of[int(c) - cS] - eS])])
              for c in cells}
    chosen = _independent_edges(dm, np.array(sorted(wanted), dtype=np.int64))

    # Count OWNED edges only: a shared edge is held by every rank on the seam, so
    # summing local counts would report it once per sharer and overstate the pass.
    n_split = uw.mpi.comm.allreduce(int(_owned_count(dm, chosen)), op=MPI.SUM)
    if n_split == 0:
        return dm, 0

    work = dm.clone()
    work.createLabel(_BISECT_LABEL)
    label = work.getLabel(_BISECT_LABEL)
    label.setDefaultValue(0)
    for e in chosen:
        label.setValue(int(e), 1)

    transform = PETSc.DMPlexTransform().create(comm=work.comm)
    transform.setType("uwnvb_bisect")
    transform.setDM(work)
    transform.setUp()
    refined = transform.apply(work)
    transform.destroy()

    # The transform copies its driving label onto the output, where it would be
    # read as a stale request by the next pass.
    if refined.hasLabel(_BISECT_LABEL):
        refined.removeLabel(_BISECT_LABEL)
    return refined, n_split
