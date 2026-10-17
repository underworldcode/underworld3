r"""One partition for the add-on hierarchy.

The add-on is the finest mesh built on a static base hierarchy — an ``adapt``
child, the mesh a fault is about to be cut into or placed into — together with
the add-on levels refined from the base toward the fault. ``adapt`` leaves the
add-on co-partitioned with the base; fault surgery needs the fault's cell star
rank-interior (the split's pairs and the placement's cavity are rank-local by
construction) and so used to move cells on its own terms, once per surgery
and once per level, after which no two levels shared a partition and every
transfer, gather and diagnostic carried its own cross-partition handling.

:func:`partition_addon` moves the add-on **once, before any surgery**: one
assignment computed on the finest level and applied to every add-on level
through the parent-cell relation, so the add-on levels stay co-partitioned
with each other exactly, the fault regions are rank-interior, and the load is
balanced by diffusion over the seams: any rank heavier than a neighbour passes
it cells that belong to no fault region, the owner's farthest first. The base
levels never move (load fixes go on the child). Every
level carries, per cell, the cell it came from and its parent on the level
below, so the one pair that is no longer co-partitioned — the base's finest
level against the first add-on level — is an exact relation to fetch, not a
search. Design note:
``docs/developer/design/parallel-addon-distribution-2026-10.md``.

The unit of assignment is a cell of the coarsest add-on level with every
descendant it has on the finer levels (the finest mesh itself when there is
no add-on level below it). All the cells of a unit go to one rank.
"""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw
# A fine cell's centroid lies inside its parent for nested refinement; the
# tolerance is the cell locator's own, so a point it accepts is accepted here.
from underworld3.utilities.fault_mg import _INSIDE_TOL

__all__ = ["partition_addon", "mark_labels", "mark_polylines", "mark_sheets",
           "mark_band"]



def _allreduce(comm, values, op=MPI.SUM):
    """Reduce a float array over the communicator (buffer-based; the
    pickling form applies the operator to Python objects)."""
    out = np.array(values, dtype=float)
    comm.Allreduce(MPI.IN_PLACE, out, op=op)
    return out


# --------------------------------------------------------------------------- #
#  Marks: which vertices the fault touches
# --------------------------------------------------------------------------- #
def _mark_facet_vertices(dm, facets, value, ids):
    """Raise ``ids`` (chart-length) to at least ``value`` on every vertex of
    the given facets (points of the height-1 stratum; others are skipped)."""
    from underworld3.utilities._plex_topology import closure_points
    fS, fE = dm.getHeightStratum(1)
    pStart, _pEnd = dm.getChart()
    facets = np.asarray(facets, dtype=np.int64)
    facets = facets[(facets >= fS) & (facets < fE)]
    if facets.size:
        v = closure_points(dm, facets, 0)
        v = v[v >= 0] - pStart
        np.maximum.at(ids, v, value)


def mark_labels(labels, groups=None):
    """The marking rule for labelled fault facets: region ``k`` on every vertex
    of a facet of group ``k`` (one group per fault when ``groups`` is None).

    ``labels`` is a sequence of ``(name, value)`` pairs; ``groups`` a list of
    lists of those pairs, the faults that must share a rank. Returns the
    function of a level's DM that :func:`partition_addon` takes as
    ``region_marks``."""
    if groups is None:
        groups = [[lab] for lab in labels]

    def marks(dm):
        pStart, pEnd = dm.getChart()
        ids = np.zeros(pEnd - pStart, dtype=np.int32)
        for g, group in enumerate(groups, start=1):
            for name, value in group:
                if (dm.hasLabel(name)
                        and dm.getLabel(name).getStratumSize(int(value)) > 0):
                    _mark_facet_vertices(
                        dm, dm.getLabel(name).getStratumIS(int(value)).getIndices(),
                        g, ids)
        return ids
    return marks


def _segment_distance(X, poly):
    """Distance from each point of ``X`` (N, 2) to the polyline ``poly`` (M, 2)."""
    best = np.full(X.shape[0], np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        t = np.clip(((X - a) @ ab) / max(float(ab @ ab), 1e-300), 0.0, 1.0)
        d = np.linalg.norm(X - (a + t[:, None] * ab), axis=1)
        best = np.minimum(best, d)
    return best


def mark_polylines(polylines):
    """The marking rule for 2-D line cuts: region ``k`` on every vertex within
    one local cell diameter of polyline ``k`` — the vertices the cut may pull
    onto the trace and the corners of every cell it crosses, at the level's
    own resolution."""
    polylines = [np.asarray(poly, dtype=float)[:, :2] for poly in polylines]

    def marks(dm):
        from underworld3.utilities._plex_topology import closure_points
        from underworld3.utilities.edge_split import cell_diameters
        from underworld3.utilities.place_surface import _propagate_vertex
        vS, vE = dm.getDepthStratum(0)
        cS, cE = dm.getHeightStratum(0)
        pStart, pEnd = dm.getChart()
        X = dm.getCoordinatesLocal().array.reshape(
            -1, dm.getCoordinateDim())[: vE - vS]
        h_chart = np.zeros(pEnd - pStart)
        if cE > cS:
            h_cell = np.asarray(cell_diameters(dm))
            CV = closure_points(dm, np.arange(cS, cE), 0) - pStart
            np.maximum.at(h_chart, CV, np.broadcast_to(h_cell[:, None], CV.shape))
        # a seam vertex sees only its local cells: the largest incident cell
        # over every rank, so the reach is a function of the mesh alone
        h_chart = _propagate_vertex(dm, h_chart, MPI.MAX, np.maximum)
        h_vertex = h_chart[vS - pStart: vE - pStart]
        ids = np.zeros(pEnd - pStart, dtype=np.int32)
        for k, poly in enumerate(polylines, start=1):
            near = np.flatnonzero(_segment_distance(X[:, :2], poly) <= h_vertex)
            ids[near + vS - pStart] = np.maximum(ids[near + vS - pStart], k)
        return ids
    return marks


def mark_sheets(sheets, clearance):
    """The marking rule for 3-D sheet placements: the placement's own mask —
    region ``k`` on every vertex within ``(clearance + 1)`` local cell
    diameters of sheet ``k`` (``(points, triangles)`` pairs) — so the carve,
    its growth rounds and the split all fall inside the region once the
    growth layers are added."""
    sheets = [(np.asarray(pts, dtype=float), np.asarray(tri, dtype=np.int64))
              for pts, tri in sheets]

    def marks(dm):
        from underworld3.utilities.place_surface import (
            _coords, _sheet_distance_within, _tet_vertices, _vertex_h_3d)
        vS, vE = dm.getDepthStratum(0)
        pStart, pEnd = dm.getChart()
        X = _coords(dm)[: vE - vS]
        cells = _tet_vertices(dm)
        h_vertex, _h_cell = _vertex_h_3d(dm, cells, len(X))
        reach = (float(clearance) + 1.0) * h_vertex
        ids = np.zeros(pEnd - pStart, dtype=np.int32)
        for k, (points, triangles) in enumerate(sheets, start=1):
            d = _sheet_distance_within(X, points, triangles, reach)
            near = np.flatnonzero(d < reach)
            ids[near + vS - pStart] = np.maximum(ids[near + vS - pStart], k)
        return ids
    return marks


def mark_band(paths, width, clearance):
    """The marking rule for a placed fault band (``place_thin_volume``): a
    superset of the placement's own mask, so its gather finds the region
    rank-interior. The placement marks every vertex within
    ``max(clearance h, 0.6 width) + h`` of the band's skin (``h`` the
    smallest incident cell diameter); the skin lies within a mitred half
    width of the band's path, so a vertex within that reach plus one band
    width of the path is marked here. ``paths`` are the paths the band is
    built along: ``(n, 2)`` polylines in 2-D, planar convex ``(n, 3)``
    polygons in 3-D (the patches)."""
    paths = [np.asarray(P, dtype=float) for P in paths]

    def marks(dm):
        from underworld3.utilities.place_surface import (
            _coords, _sheet_distance_within, _tet_vertices, _vertex_h_3d)
        vS, vE = dm.getDepthStratum(0)
        pStart, pEnd = dm.getChart()
        X = _coords(dm)[: vE - vS]
        h_vertex, _h_cell = _vertex_h_3d(dm, _tet_vertices(dm) if dm.getDimension() == 3
                                         else _cell_vertices(dm), len(X))
        reach = (np.maximum(float(clearance) * h_vertex, 0.6 * float(width))
                 + h_vertex + float(width))
        ids = np.zeros(pEnd - pStart, dtype=np.int32)
        for k, P in enumerate(paths, start=1):
            if dm.getDimension() == 2:
                d = _segment_distance(X[:, :2], P[:, :2])
            else:
                fan = np.array([(0, i, i + 1) for i in range(1, len(P) - 1)],
                               dtype=np.int64)
                d = _sheet_distance_within(X, P, fan, reach)
            near = np.flatnonzero(d < reach)
            ids[near + vS - pStart] = np.maximum(ids[near + vS - pStart], k)
        return ids
    return marks


def _cell_vertices(dm):
    """(n_cells, dim + 1) local vertex indices of every simplex."""
    from underworld3.utilities._plex_topology import closure_points
    vS, _vE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    return (closure_points(dm, np.arange(cS, cE), 0) - vS).reshape(
        cE - cS, dm.getDimension() + 1)

def embedded_surface_names(mesh):
    """The fault surfaces ``mesh`` already carries, as boundary names: every
    split fault (its own label and its ``Plus`` / ``Minus`` sides) and every
    sheet placed into it or an ancestor (``_embedded_surfaces``, recorded by
    the placement). These are glued by every later partition, so a second
    fault or sheet never breaks up the first. A material interface or a
    domain-spanning internal boundary is not among them."""
    names = set(getattr(mesh, "_embedded_surfaces", ()) or ())
    for name in (getattr(mesh, "_fault_point_pairs", {}) or {}):
        names.update((name, f"{name}Plus", f"{name}Minus"))
    return names


def _embedded_surface_marks(mesh, names, first_id):
    """Chart-length ids gluing the named surfaces: ``first_id + j`` on every
    vertex of a facet of the ``j``-th name (sorted). ``names`` is the same on
    every rank, so the ids are."""
    dm = mesh.dm
    pStart, pEnd = dm.getChart()
    ids = np.zeros(pEnd - pStart, dtype=np.int32)
    present = {b.name: int(b.value) for b in mesh.boundaries}
    for j, name in enumerate(names):
        if name not in present or not dm.hasLabel(name):
            continue
        lab = dm.getLabel(name)
        if lab.getStratumSize(present[name]) > 0:
            _mark_facet_vertices(dm, lab.getStratumIS(present[name]).getIndices(),
                                 first_id + j, ids)
    return ids


def inherit_provenance(child, host):
    """Record on ``child`` — a mesh made from ``host`` by a cut, a placement or
    a split — the parent of each of its cells on the level ``host`` records
    them against, while the two geometries still agree. The surgery was
    rank-local (the partition made the region interior), so every child
    centroid lies in a local host cell. Recording it here, rather than
    locating later, keeps the relation exact after the child's nodes move.
    A no-op when ``host`` carries no record (serial, or never partitioned).
    Collective."""
    from underworld3.utilities.fault_mg import locate_in_cells_exhaustive
    parent = getattr(host, "_addon_parent", None)
    if parent is None or getattr(host, "_addon_below", None) is None:
        return
    comm = child.dm.getComm().tompi4py()
    fcent, _v, cvf = _cell_centroids(child)
    _c, vxy_h, cv_h = _cell_centroids(host)
    cell = np.zeros(cvf.shape[0], dtype=np.int64)
    n_bad = 0
    if cvf.shape[0]:
        if cv_h.shape[0] == 0:
            n_bad = cvf.shape[0]
        else:
            cell, lam = locate_in_cells_exhaustive(fcent, vxy_h, cv_h)
            n_bad = int((lam.min(axis=1) < -_INSIDE_TOL).sum())
    n_bad = int(comm.allreduce(n_bad))
    if n_bad:
        raise RuntimeError(
            f"addon_partition: {n_bad} cell(s) of a surgery's result lie in no "
            f"local cell of the mesh it was made from; the surgery was not "
            f"rank-local.")
    child._addon_parent = np.asarray(parent, dtype=np.int64)[cell]
    child._addon_below = host._addon_below


# --------------------------------------------------------------------------- #
#  The parent relation and the units
# --------------------------------------------------------------------------- #
def _cell_centroids(mesh):
    from underworld3.utilities.fault_mg import _cell_vertex_table
    vxy, cv = _cell_vertex_table(mesh)
    return vxy[cv].mean(axis=1), vxy, cv


def _parent_cells_local(coarse, fine):
    """``parent[k]``: the local cell of ``coarse`` holding the centroid of local
    cell ``k`` of ``fine``. Exact for nested refinement and for a grid that
    re-represents the coarse one; raises if any centroid falls outside every
    local coarse cell, which means the two levels are not co-partitioned —
    the add-on has already been moved and must not be partitioned twice."""
    from underworld3.utilities.fault_mg import locate_in_cells_exhaustive
    comm = fine.dm.getComm().tompi4py()
    fcent, _vf, cvf = _cell_centroids(fine)
    _cc, vxy_c, cv_c = _cell_centroids(coarse)
    n_f, n_c = cvf.shape[0], cv_c.shape[0]
    cell = np.full(n_f, -1, dtype=np.int64)
    if n_f and n_c:
        cell, lam = locate_in_cells_exhaustive(fcent, vxy_c, cv_c)
        cell = np.where(lam.min(axis=1) < -_INSIDE_TOL, -1, cell).astype(np.int64)
    n_bad = int(comm.allreduce(int((cell < 0).sum()) + (n_f > 0 and n_c == 0)))
    if n_bad:
        raise RuntimeError(
            f"addon_partition: {n_bad} fine cell(s) have no local parent cell "
            f"on the level below; the add-on levels are not co-partitioned "
            f"(an add-on is partitioned once, as adapt made it).")
    return cell


def _base_parent_addresses(base, lev, comm):
    """``(n_cells, 2)``: rank and local cell of each cell's parent on ``base``.

    Located in this rank's base cells while the two are co-partitioned (as
    ``adapt`` leaves them). Once a partition has moved the add-on, the pair is
    not co-partitioned any more and the relation is READ, not searched: the
    mesh that recorded it (``fault_mg.provenance_source`` — ``lev`` itself, or
    the level a cut or a placement re-represented) is located locally, the
    surgery having been rank-local, and its recorded parent is taken. That is
    what lets a second sheet, or a fault after a material interface, be
    partitioned on top of a first."""
    from underworld3.utilities.fault_mg import (locate_in_cells_exhaustive,
                                                provenance_source)
    pre = provenance_source(base, lev)
    if pre is None:
        local = _parent_cells_local(base, lev)
        return np.column_stack([np.full(local.shape[0], comm.rank,
                                        dtype=np.int64), local])
    if pre is lev:
        return np.asarray(lev._addon_parent, dtype=np.int64).copy()
    fcent, _v, cvf = _cell_centroids(lev)
    _c, vxy_p, cv_p = _cell_centroids(pre)
    cell = np.zeros(cvf.shape[0], dtype=np.int64)
    n_bad = 0
    if cvf.shape[0]:
        if cv_p.shape[0] == 0:
            n_bad = cvf.shape[0]
        else:
            cell, lam = locate_in_cells_exhaustive(fcent, vxy_p, cv_p)
            n_bad = int((lam.min(axis=1) < -_INSIDE_TOL).sum())
    n_bad = int(comm.allreduce(n_bad))
    if n_bad:
        raise RuntimeError(
            f"addon_partition: {n_bad} cell(s) lie in no local cell of the "
            f"level their provenance is recorded on; the surgery since the "
            f"last partition was not rank-local.")
    return np.asarray(pre._addon_parent, dtype=np.int64)[cell]


def _levels_of(mesh):
    """``(base, static, addon)``: the static level the first add-on level's
    transfer reads from (or None), the static wraps that form the tail under
    the add-on, and the add-on levels coarsest first, excluding ``mesh``."""
    from underworld3.utilities.fault_mg import addon_level_indices
    tail = getattr(mesh, "_custom_mg_coarse_meshes", None)
    if tail is not None:
        idx = set(addon_level_indices(tail))
        static = [m for k, m in enumerate(tail) if k not in idx]
        addon = [m for k, m in enumerate(tail) if k in idx]
        return (static[-1] if static else None), static, addon
    # a mesh built with refinement >= 1: the split's rule makes the coarse
    # levels below the finest the tail, the finest wrap being this very grid
    wraps = (list(mesh._coarse_level_meshes())[:-1]
             if len(getattr(mesh, "dm_hierarchy", []) or []) > 1 else [])
    return (wraps[-1] if wraps else None), wraps, []


# --------------------------------------------------------------------------- #
#  Seam neighbours
# --------------------------------------------------------------------------- #
def _face_other_rank(dm):
    """Per local face, the rank holding the cell across the seam, or -1 for a
    face interior to this rank's cells. A shared face has one root and one
    leaf: the leaf side reads the root's rank off the forest and the root
    side gets the leaf's rank by one reduction."""
    fS, fE = dm.getHeightStratum(1)
    pStart, pEnd = dm.getChart()
    other = np.full(pEnd - pStart, -1, dtype=np.int64)
    comm = dm.getComm().tompi4py()
    if comm.size == 1:
        return other[fS - pStart: fE - pStart]
    sf = dm.getPointSF()
    try:
        nroots, ilocal, iremote = sf.getGraph()
    except (ValueError, TypeError):
        # Sanctioned: petsc4py raises on a forest with no graph set, which is
        # a rank that shares no point; every face is then interior.
        return other[fS - pStart: fE - pStart]
    leaf = np.full(pEnd - pStart, -1, dtype=np.int64)
    if ilocal is not None and len(ilocal):
        leaf[np.asarray(ilocal)] = comm.rank
        other[np.asarray(ilocal)] = np.asarray(iremote)[:, 0]
    root = np.full(pEnd - pStart, -1, dtype=np.int64)
    sf.reduceBegin(MPI.INT64_T, leaf, root, MPI.MAX)
    sf.reduceEnd(MPI.INT64_T, leaf, root, MPI.MAX)
    has_leaf = root >= 0
    other[has_leaf] = root[has_leaf]
    return other[fS - pStart: fE - pStart]


def _unit_adjacency(dm, unit_of_cell, face_other):
    """The face adjacency of the units: ``(ua, ub)`` pairs of local units that
    share a face, and ``(u, rank)`` pairs of local units with a face on the
    seam to ``rank``. Computed once; the assignment changes, the mesh does
    not."""
    from underworld3.utilities._plex_topology import supports
    cS, cE = dm.getHeightStratum(0)
    fS, fE = dm.getHeightStratum(1)
    sup = supports(dm, np.arange(fS, fE))
    if sup.shape[1] < 2:
        sup = np.column_stack([sup, np.full((sup.shape[0], 2 - sup.shape[1]), -1)])
    both = (sup[:, 0] >= 0) & (sup[:, 1] >= 0)
    ua = unit_of_cell[sup[both, 0] - cS]
    ub = unit_of_cell[sup[both, 1] - cS]
    inner = np.column_stack([ua, ub])[ua != ub].astype(np.int64)
    one = (sup[:, 0] >= 0) & (sup[:, 1] < 0) & (face_other >= 0)
    seam = np.column_stack([unit_of_cell[sup[one, 0] - cS],
                            face_other[one]]).astype(np.int64)
    return inner.reshape(-1, 2), seam.reshape(-1, 2)


def _units_touching(inner, seam, assign_unit, n_ranks):
    """``(unit, rank)`` pairs: every local unit with the ranks it touches
    across a face, under the current assignment, each pair once. A pair is
    encoded as ``unit * n_ranks + rank`` so the deduplication is a 1-D sort."""
    a, b = inner[:, 0], inner[:, 1]
    keys = np.unique(np.concatenate([
        a * n_ranks + assign_unit[b],
        b * n_ranks + assign_unit[a],
        seam[:, 0] * n_ranks + seam[:, 1]]))
    return np.column_stack([keys // n_ranks, keys % n_ranks])


# --------------------------------------------------------------------------- #
#  Balance by diffusion
# --------------------------------------------------------------------------- #
def _peel_balance(dm, unit_of_cell, assign_unit, weight, unit_region,
                  unit_centroid, region_centres, tolerance, comm,
                  max_sweeps=100):
    """Balance the load by diffusion over the seams: in each sweep a rank
    offers free units (no fault region) across each face it shares with a
    lighter neighbour, up to half the difference in load, the cells farthest
    from the faults first; a receiver's intake is scaled so it never exceeds
    the largest half-gap offered to it. A rank at the mean passes cells on,
    which is what carries load around a fault region that walls its owner
    off from the lightest rank. Stops when every rank is within
    ``tolerance`` of the mean, when nothing can move, or at ``max_sweeps``.
    Collective in every sweep. Returns the assignment and the sweep count."""
    size = comm.size
    face_other = _face_other_rank(dm)
    inner, seam = _unit_adjacency(dm, unit_of_cell, face_other)
    far = (np.full(assign_unit.shape[0], np.inf) if region_centres.shape[0] == 0
           else np.min(np.linalg.norm(
               unit_centroid[:, None, :] - region_centres[None, :, :], axis=2),
               axis=1))
    free = unit_region == 0
    n_sweeps = 0
    for _sweep in range(max_sweeps):
        load = _allreduce(comm, np.bincount(assign_unit, weights=weight,
                                            minlength=size))
        mean = load.sum() / size
        if load.max() <= mean * (1.0 + tolerance):
            break
        n_sweeps += 1
        touching = _units_touching(inner, seam, assign_unit, size)
        mine = free & (assign_unit == comm.rank)
        touching = touching[mine[touching[:, 0]]]
        proposals = {}            # receiver -> list of (unit, weight)
        gap = np.zeros(size)      # my half-gap to each receiver
        lighter = np.flatnonzero(load < load[comm.rank] - tolerance * mean)
        taken = np.zeros(assign_unit.shape[0], dtype=bool)
        for s in lighter[np.argsort(load[lighter], kind="stable")]:
            s = int(s)
            if s == comm.rank:
                continue
            cand = touching[touching[:, 1] == s, 0]
            cand = cand[~taken[cand]]
            if cand.size == 0:
                continue
            cand = cand[np.lexsort((cand, -far[cand]))]
            room = 0.5 * (load[comm.rank] - load[s])
            gap[s] = room
            chosen = []
            for u in cand:
                # a unit heavier than twice the room left would overshoot the
                # gap and could be traded back next sweep
                if room < 0.5 * weight[u]:
                    continue
                chosen.append((int(u), float(weight[u])))
                room -= weight[u]
                taken[u] = True
            proposals[s] = chosen
        offered = np.zeros(size)
        for s, chosen in proposals.items():
            offered[s] = sum(w for _u, w in chosen)
        offered_all = _allreduce(comm, offered)
        gap_all = _allreduce(comm, gap, op=MPI.MAX)
        moved = 0
        for s, chosen in proposals.items():
            scale = (min(1.0, gap_all[s] / offered_all[s])
                     if offered_all[s] > 0 else 0.0)
            budget = scale * offered[s]
            for u, w in chosen:
                if budget <= 0:
                    break
                assign_unit[u] = s
                budget -= w
                moved += 1
        if comm.allreduce(moved) == 0:
            break
    return assign_unit, n_sweeps


# --------------------------------------------------------------------------- #
#  Provenance through the move
# --------------------------------------------------------------------------- #
def _resolve_parents(below, lev):
    """Give level ``lev`` its parents' CURRENT addresses.

    Every cell of ``lev`` knows its parent on ``below`` as it was before the
    move (``_addon_parent_old``: rank, local cell); every cell of ``below``
    knows where it was (``_addon_self_old``). A directory on the old owners
    (one reduction) maps old to new, and one broadcast hands each fine cell
    its parent's new address. Stored as ``lev._addon_parent`` (n_cells, 2):
    rank and local cell on ``below`` as it stands now."""
    comm = lev.dm.getComm()
    m4 = comm.tompi4py()
    parent_old = lev._addon_parent_old
    self_old = getattr(below, "_addon_self_old", None)
    if self_old is None:
        # the level below did not move: old addresses are current
        lev._addon_parent = parent_old.copy()
        return
    n_old = int(below._addon_n_cells_old)
    cS, cE = below.dm.getHeightStratum(0)
    unit = MPI.INT64_T.Create_contiguous(2)
    unit.Commit()
    # directory: old cell -> (new rank, new local)
    sf_dir = PETSc.SF().create(comm=comm)
    sf_dir.setGraph(n_old, None, np.ascontiguousarray(self_old, dtype=np.int32))
    sf_dir.setUp()
    now = np.column_stack([np.full(cE - cS, m4.rank), np.arange(cE - cS)]
                          ).astype(np.int64)
    directory = np.full((n_old, 2), -1, dtype=np.int64)
    # exactly one leaf per root, so REPLACE is the reduction (the predefined
    # reductions do not take a derived datatype)
    sf_dir.reduceBegin(unit, now, directory, MPI.REPLACE)
    sf_dir.reduceEnd(unit, now, directory, MPI.REPLACE)
    sf_dir.destroy()
    # fetch: each fine cell reads its parent's entry
    sf_get = PETSc.SF().create(comm=comm)
    sf_get.setGraph(n_old, None,
                    np.ascontiguousarray(parent_old, dtype=np.int32))
    sf_get.setUp()
    parent_new = np.full(parent_old.shape, -1, dtype=np.int64)
    sf_get.bcastBegin(unit, directory, parent_new, MPI.REPLACE)
    sf_get.bcastEnd(unit, directory, parent_new, MPI.REPLACE)
    sf_get.destroy()
    unit.Free()
    if comm.tompi4py().allreduce(int((parent_new < 0).sum())):
        raise RuntimeError(
            "addon_partition internal: a cell's parent was not found after "
            "the move.")
    lev._addon_parent = parent_new


def _moved_wrap(level, dm, fields, pairs):
    """A new Mesh over the redistributed ``dm`` of ``level``, carrying the
    provenance fields and whatever fault geometry the level had."""
    moved = uw.discretisation.Mesh(
        dm, simplex=level.dm.isSimplex(), qdegree=level.qdegree,
        coordinate_system_type=level.CoordinateSystem.coordinate_type,
        boundaries=level.boundaries, verbose=False)
    moved._mg_static = False
    for attr in ("_fault_traces", "_fault_specs", "_fault_surfaces",
                 "_fault_seam_normals", "_embedded_surfaces"):
        if getattr(level, attr, None) is not None:
            setattr(moved, attr, getattr(level, attr))
    if pairs is not None:
        moved._fault_point_pairs = pairs
    parent_rank, parent_local, self_rank, self_local = fields
    moved._addon_parent_old = np.column_stack([parent_rank, parent_local])
    moved._addon_self_old = np.column_stack([self_rank, self_local])
    cS, cE = level.dm.getHeightStratum(0)
    moved._addon_n_cells_old = cE - cS
    return moved


# --------------------------------------------------------------------------- #
#  The entry point
# --------------------------------------------------------------------------- #
def partition_addon(mesh, region_marks, layers=1, tolerance=0.1,
                    verbose=False):
    """Partition ``mesh`` and its add-on levels for the faults ``region_marks``
    marks.

    Parameters
    ----------
    mesh : Mesh
        The finest add-on level, as ``adapt`` made it (co-partitioned with
        its levels), before any cut, placement or split.
    region_marks : callable
        ``region_marks(dm)`` returns chart-length region ids on a level's DM:
        ``0`` for an unmarked point, ``k >= 1`` the fault region a vertex
        belongs to (:func:`mark_polylines`, :func:`mark_sheets`,
        :func:`mark_labels`). It is evaluated on EVERY add-on level, because
        the same fault's cell star reaches further on a coarser level and the
        split replayed there needs its own star rank-interior. Every fault
        surface the levels already carry (:func:`embedded_surface_names`) is
        glued as well, so an add-on can be partitioned again for a further
        fault without breaking up the first. Regions whose stars touch merge.
    layers : int
        Growth layers beyond the marked vertices' cell star: one makes every
        point of a star cell unshared (the split's seam rule); the 3-D carve's
        growth rounds need two.
    tolerance : float
        Balance target: every rank's share of finest cells within this
        fraction of the mean, if the seams allow it.

    Returns
    -------
    Mesh
        A new finest mesh whose add-on tail is moved the same way, with the
        provenance of every level recorded (``_addon_parent``); or ``mesh``
        itself when every region is already rank-interior and the load is
        within tolerance. Serial: ``mesh``.
    """
    from underworld3.utilities.fault_split import redistribute_by_assignment
    from underworld3.utilities.place_surface import _merge_regions, _region_cells

    comm = mesh.dm.getComm().tompi4py()
    if comm.size == 1:
        return mesh
    base, static, addon = _levels_of(mesh)
    levels = addon + [mesh]
    # the add-on levels are co-partitioned with each other (adapt made them
    # so, or an earlier partition did): their parents are local. The first
    # add-on level's parent on the base is local too unless an earlier
    # partition moved it, in which case the recorded address is used
    parents = [None]
    for l in range(1, len(levels)):
        parents.append(_parent_cells_local(levels[l - 1], levels[l]))
    base_parent = (None if base is None
                   else _base_parent_addresses(base, levels[0], comm))
    cS0, cE0 = levels[0].dm.getHeightStratum(0)
    n_units = cE0 - cS0
    unit_of = [np.arange(n_units, dtype=np.int64)]
    for l in range(1, len(levels)):
        unit_of.append(unit_of[l - 1][parents[l]])
    weight = np.bincount(unit_of[-1], minlength=n_units).astype(float)

    # fault regions on every level, lifted to units: the faults asked for,
    # and every interior surface the levels already carry (placed or split
    # by an earlier call), so a second partition keeps those whole
    marks = [np.asarray(region_marks(lev.dm), dtype=np.int32) for lev in levels]
    n_asked = int(comm.allreduce(max((int(m.max()) for m in marks if m.size),
                                     default=0), op=MPI.MAX))
    glued = sorted(set().union(*comm.allgather(
        set().union(*[embedded_surface_names(lev) for lev in levels]))))
    for l, lev in enumerate(levels):
        marks[l] = np.maximum(marks[l],
                              _embedded_surface_marks(lev, glued, n_asked + 1))
    unit_region = np.zeros(n_units, dtype=np.int32)
    pairs = set()
    for l, lev in enumerate(levels):
        cell_region, _canon, _regions = _region_cells(
            lev.dm, marks[l], layers=layers)
        sel = np.flatnonzero(cell_region)
        u, k = unit_of[l][sel], cell_region[sel]
        # a unit takes the largest id among its cells' and its own; the ids
        # that meet in one unit touch
        best = unit_region.copy()
        np.maximum.at(best, u, k)
        held = np.concatenate([k, unit_region[u]])
        top = np.concatenate([best[u], best[u]])
        meet = (held > 0) & (held != top)
        if meet.any():
            for a, b in np.unique(np.column_stack([held[meet], top[meet]]),
                                  axis=0).tolist():
                pairs.add((int(a), int(b)))
        unit_region = best
    n_ids = int(comm.allreduce(int(unit_region.max()) if n_units else 0,
                               op=MPI.MAX))
    canon = _merge_regions(pairs, n_ids, comm)
    lookup = np.zeros(n_ids + 1, dtype=np.int32)
    for key, value in canon.items():
        lookup[key] = value
    unit_region = lookup[unit_region]
    regions = sorted(set(canon.values()))
    local = np.array([float(weight[unit_region == k].sum()) for k in regions])
    counts = np.asarray(comm.allgather(local)).reshape(comm.size, len(regions))
    # an id no level marked anywhere (a fault outside the mesh, a glued
    # surface absent from these levels) is not a region
    present = counts.sum(axis=0) > 0
    regions = [k for k, keep in zip(regions, present) if keep]
    counts = counts[:, present]
    assign = np.full(n_units, comm.rank, dtype=np.int32)
    owner = {}
    for j, k in enumerate(regions):
        col = counts[:, j]
        owner[k] = int(np.argmax(col))
        if np.count_nonzero(col) > 1:
            assign[unit_region == k] = owner[k]

    # the fault owners shed far cells to their underloaded neighbours
    unit_centroid, _v, _cv = _cell_centroids(levels[0])
    centres = []
    for j, k in enumerate(regions):
        sel = unit_region == k
        s = (unit_centroid[sel] * weight[sel, None]).sum(axis=0)
        tot = _allreduce(comm, np.concatenate([s, [weight[sel].sum()]]))
        centres.append(tot[:-1] / tot[-1])
    centres = np.asarray(centres).reshape(len(regions), -1)
    load_before = _allreduce(comm, np.bincount(assign, weights=weight,
                                               minlength=comm.size))
    assign, n_sweeps = _peel_balance(levels[0].dm, unit_of[0], assign, weight,
                                     unit_region, unit_centroid, centres,
                                     tolerance, comm)
    load_after = _allreduce(comm, np.bincount(assign, weights=weight,
                                              minlength=comm.size))
    n_moved_units = int(comm.allreduce(int((assign != comm.rank).sum())))
    record = {
        "regions": len(regions), "owners": owner,
        "region_weight": {k: float(counts[:, j].sum())
                          for j, k in enumerate(regions)},
        "units_moved": n_moved_units, "sweeps": n_sweeps,
        "load_before": load_before.tolist(), "load_after": load_after.tolist(),
    }
    if n_moved_units == 0:
        # already as the partition would leave it: the mesh stands, with the
        # record of the decision
        mesh._addon_partition = record
        if verbose:
            uw.pprint(f"[addon_partition] {len(regions)} fault region(s) "
                      f"already rank-interior, load within tolerance "
                      f"(max/mean {load_after.max() / load_after.mean():.2f}); "
                      f"nothing moves")
        return mesh

    # Every unit has a cell on every level, so a rank that receives no unit
    # holds nothing on any level. A DM with every point on one rank reads as
    # not distributed, and the Mesh constructor would partition it afresh,
    # which the provenance cannot ride: refuse instead (collectively).
    owners_used = int(np.count_nonzero(_allreduce(
        comm, np.bincount(assign, minlength=comm.size))))
    if owners_used < 2:
        raise RuntimeError(
            "addon_partition: the whole add-on would sit on one rank (its "
            "fault regions cover it); run on fewer ranks or refine the base.")

    # apply the one assignment to every level, provenance riding along
    moved = []
    for l, lev in enumerate(levels):
        cS, cE = lev.dm.getHeightStratum(0)
        n = cE - cS
        a_level = assign[unit_of[l]]
        if l > 0:
            parent_rank = np.full(n, comm.rank, dtype=np.int64)
            parent_local = parents[l]
        elif base_parent is not None:
            parent_rank, parent_local = base_parent[:, 0], base_parent[:, 1]
        else:
            parent_rank = np.full(n, -1, dtype=np.int64)
            parent_local = np.full(n, -1, dtype=np.int64)
        self_rank = np.full(n, comm.rank, dtype=np.int64)
        self_local = np.arange(n, dtype=np.int64)
        fields = [parent_rank, parent_local, self_rank, self_local]
        old_pairs = getattr(lev, "_fault_point_pairs", None)
        dm, new_pairs, dropped, new_fields = redistribute_by_assignment(
            lev.dm, a_level, pairs=old_pairs, fields=fields)
        if dropped:
            raise RuntimeError(
                f"addon_partition: {dropped} fault pair(s) split across ranks "
                f"by the move on level {l}")
        # every unit has a cell on every level, so a moved unit moves every
        # level and the shell partition never hands the input back here
        moved.append(_moved_wrap(lev, dm, new_fields, new_pairs))
    for l, lev in enumerate(moved):
        below = base if l == 0 else moved[l - 1]
        lev._addon_below = below
        if below is None:
            lev._addon_parent = None
        else:
            _resolve_parents(below, lev)

    out = moved[-1]
    out.parent = mesh
    out._relationship_kind = "refinement"
    out._refine_dofs_coincide = False
    out.regions = mesh.regions
    out._parent_mesh_version = mesh._mesh_version
    mesh._registered_children.add(out)
    if static or moved[:-1]:
        # the tail: the static wraps as they stand (never moved) and the
        # add-on levels moved with the finest. A refined base's hierarchy is
        # made explicit here, since the moved wrap no longer carries it.
        out._custom_mg_coarse_meshes = list(static) + moved[:-1]
        out._custom_mg_builder = getattr(mesh, "_custom_mg_builder", "barycentric")
        out._custom_mg_fac_zone = getattr(mesh, "_custom_mg_fac_zone", None)
    out._addon_partition = record
    if verbose:
        uw.pprint(f"[addon_partition] {len(regions)} fault region(s) to ranks "
                  f"{sorted(owner.values())}; {n_moved_units} unit(s) moved "
                  f"over {len(levels)} level(s); finest cells per rank "
                  f"max/mean {load_before.max() / load_before.mean():.2f} -> "
                  f"{load_after.max() / load_after.mean():.2f} after "
                  f"{n_sweeps} balance sweep(s)")
    return out
