r"""Split-node faults: a labelled facet chain becomes a genuine discontinuity.

A conforming surface (:meth:`Mesh.add_conforming_surface`) leaves the fault as
a chain of interior facets that the two flanking cells *share*, so every
continuous finite-element space is continuous across it. Splitting the nodes
removes that: each interior vertex of the chain is duplicated, the cells on one
side are rewired to the replicas, and the fault facets themselves are doubled —
two geometrically coincident copies with no degrees of freedom in common. A
velocity (or any) field can then jump across the fault, which is what a fault
*is*: slip, not strain. The tips are never duplicated, so the field stays
continuous there and a slip datum must taper to zero at the tips.

The two sides become named boundaries — ``<name>Plus`` and ``<name>Minus`` —
so an ordinary essential condition prescribes slip::

    child = split_fault(mesh, "Fault")
    stokes.add_dirichlet_bc(v_bg + s/2 * taper * tangent, "FaultPlus")
    stokes.add_dirichlet_bc(v_bg - s/2 * taper * tangent, "FaultMinus")

``Plus`` is the LEFT side walking the chain from its first tip to its second,
where the first tip is the one lower in coordinate order: with tangent
:math:`\hat t` along the walk, the normal :math:`\hat n = (-t_y, t_x)` points
into the Plus side. The original label survives on BOTH copies, so it remains
the whole-fault handle for visualisation and for the repair passes'
interface protection.

The split is a pure function of the source DM — nothing is cached, the source
is not modified — because a fault MOVES: re-splitting after migration is
re-cutting the (static) base mesh at the new position and calling this again,
the same non-cumulative pattern as :meth:`Mesh.adapt`.

Built on the rebuild machinery of :mod:`underworld3.utilities.reconnect`:
the same fresh-plex construction (`DMPlexSymmetrize` refuses surgery in
place), the same derived-edge rule (an old facet is reused wherever its vertex
pair is still wanted, which is what carries its labels), and the same
one-broadcast star-forest renumbering. Where the deletion pass compacts the
point chart, the split grows it.

Restrictions in this version, all refused loudly rather than mishandled: a
single open chain in 2-D (no junctions, no loops, no single-facet chains) or
a single orientable manifold patch in 3-D (:func:`split_along_label_3d`; the
patch rim is the tip rule one dimension up); the fault must not touch the
domain boundary; and in parallel the fault's cell fans must not touch the
partition seam — the replicas are then rank-local and the star-forest carries
over by renumbering alone, exactly as the deletion pass argues for its own
seam freeze. (2-D additionally supports point crossings of the seam; 3-D does
not yet — a 3-D crossing is a curve, a design of its own.)

An essential condition on the fault is NOT sound under the custom-P geometric
multigrid hierarchy (the coarse levels do not carry the fault — see the
warning on :meth:`Mesh.add_conforming_surface`), so the split mesh is built
standalone, with no coarse tail: the Stokes velocity block then takes its
algebraic-multigrid default.
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities.reconnect import (
    _TOPOLOGY_LABELS, _cell_vertices_and_seam, _coords, _copy_labels,
    _rebuild_point_sf, _shared_points, _write_coordinates)


def _sf_sum(dm, arr):
    """Sum a chart-indexed int32 array over the point star-forest, in place.

    The same-array leaf/root convention proven by ``edge_split._sf_logical_or``;
    used to give every rank the GLOBAL fault-facet count at a shared vertex,
    which is what distinguishes a seam crossing (2) from a tip resting on the
    seam (1) and a junction on the seam (>2). COLLECTIVE — reached on every
    rank, gated only on communicator size.
    """
    if uw.mpi.size == 1:
        return arr
    sf = dm.getPointSF()
    try:
        nroots, _il, _ir = sf.getGraph()
    except (ValueError, TypeError):
        return arr
    if nroots < 0:
        return arr
    sf.reduceBegin(MPI.INT32_T, arr, arr, MPI.SUM)
    sf.reduceEnd(MPI.INT32_T, arr, arr, MPI.SUM)
    sf.bcastBegin(MPI.INT32_T, arr, arr, MPI.REPLACE)
    sf.bcastEnd(MPI.INT32_T, arr, arr, MPI.REPLACE)
    return arr


def _fault_chain(dm, fault_edges, X, vS, shared, pStart, orientation=None):
    """The fault as an ordered vertex path, tip to tip.

    Returns ``(problem, chain)`` where ``problem`` is ``None`` or an
    ``(ExceptionClass, message)`` pair and ``chain`` lists the fault vertices
    in walk order, starting from the tip lower in coordinate order so the walk
    — and with it the Plus/Minus sides — is a function of the geometry, not of
    label-stratum order.

    ``shared`` (chart-indexed rank-sharing flags) refines the two boundary
    rules: a support-1 facet in a fault vertex's star is the DOMAIN boundary
    only if it is unshared — on a partition seam every facet is locally
    one-sided, and a fault may run arbitrarily close to a seam. What it may
    not yet do is have a CHAIN VERTEX on the seam: an unshared vertex owns
    its whole cell fan (any cell touching it would make it shared), so every
    replica and every re-homed spoke stays rank-local; a shared chain vertex
    is a seam crossing, which needs the replica-pair star-forest extension
    (deployment design, milestone 2) and is refused until that lands.
    """
    eS, eE = dm.getDepthStratum(1)

    nbr = {}
    for e in fault_edges:
        a, b = (int(p) for p in dm.getCone(e))
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)

    endpoint_shared = any(
        shared[v - pStart] for v, ns in nbr.items() if len(ns) == 1)
    if len(fault_edges) < 2 and not endpoint_shared:
        return (ValueError,
                "fault_split: the chain has a single facet, so no interior "
                "vertex exists to split. A one-facet fault cannot slip."), None
    if any(len(ns) > 2 for ns in nbr.values()):
        return (ValueError,
                "fault_split: the label meets itself at a junction. Junctions "
                "need a side assignment this version does not define; split "
                "each branch under its own label instead."), None
    tips = [v for v, ns in nbr.items() if len(ns) == 1]
    if len(tips) != 2:
        return (ValueError,
                "fault_split: the labelled facets form a closed loop or "
                "several fragments, not a single open chain."), None

    # The fault must be strictly interior. An edge with a single support cell
    # is a domain-boundary facet — unless it is SHARED, in which case it is a
    # partition-seam facet that merely looks one-sided locally. One in the
    # star of any fault vertex means the slit would reach the boundary, where
    # the tip-clamping argument fails.
    for v in nbr:
        star = dm.getTransitiveClosure(v, useCone=False)[0]
        for p in star:
            p = int(p)
            if eS <= p < eE and len(dm.getSupport(p)) != 2 \
                    and not shared[p - pStart]:
                return (ValueError,
                        "fault_split: the fault touches the domain boundary. "
                        "Only strictly interior faults, with both tips inside "
                        "the mesh, are supported in this version."), None

    # A shared chain vertex is legal ONLY as a local endpoint — the seam
    # crossing, where this rank holds one of the two fault facets and the
    # other rank the other. A shared vertex with BOTH fault facets local has
    # its cell fan divided by the seam without the fault crossing it, and no
    # rank could walk the fan.
    for v, ns in nbr.items():
        if shared[v - pStart] and len(ns) > 1:
            return (RuntimeError,
                    "fault_split: a fault vertex sits on the partition seam "
                    "with both its facets on this rank — the fault touches "
                    "the seam without crossing it. Move the fault or the "
                    "crossing point."), None

    start = min(tips, key=lambda t: (float(X[t - vS][0]),
                                     float(X[t - vS][1]), t))
    chain, prev, cur = [start], None, start
    while True:
        onward = [w for w in nbr[cur] if w != prev]
        if not onward:
            break
        prev, cur = cur, onward[0]
        chain.append(cur)
        if len(chain) > len(nbr):
            return (ValueError,
                    "fault_split: the chain walk did not terminate; the "
                    "labelled facets are not a simple path."), None
    if len(chain) != len(nbr):
        return (ValueError,
                "fault_split: the labelled facets are not one connected "
                "chain. Cut and split one polyline at a time."), None

    # Globally consistent chain direction: a rank holding only a FRAGMENT of
    # a seam-crossing fault cannot use the lexicographic-tip rule (each rank
    # would pick its own), so the caller passes the fault's global direction
    # and every fragment is oriented against it. The Plus side is then the
    # LEFT of the same directed line on every rank.
    if orientation is not None and len(chain) >= 2:
        d = np.asarray(orientation, dtype=float).ravel()[:2]
        if float((X[chain[-1] - vS] - X[chain[0] - vS]) @ d) < 0.0:
            chain.reverse()
    elif endpoint_shared:
        return (RuntimeError,
                "fault_split: a seam-crossing fault needs the fault's global "
                "orientation to keep Plus/Minus consistent across ranks — "
                "pass orientation= (Mesh.add_fault does this itself)."), None
    return None, chain


def _take_sides(dm, chain, verts, cS, cE, crossings=()):
    """Classify every fault-fan cell as Plus (+1) or Minus (-1).

    At each interior chain vertex the incident cells are walked as a fan,
    using the directed link edge each anticlockwise cell triple contributes —
    :func:`reconnect._link_ring` opened at the two fault facets. Walking from
    the outgoing fault neighbour to the incoming one sweeps the cells on the
    LEFT of the directed chain, which is the Plus side. Any structural anomaly
    in the fan (a duplicated directed link edge, a walk that escapes or fails
    to close) means the cell orientations are not consistent there, and is a
    refusal rather than a guess: unlike a declined flip, a mis-sided cell
    would silently weld the fault shut at one node.

    Returns ``(problem, side_of_cell, substitutions)`` where ``substitutions``
    maps a Minus cell to the set of chain vertices it must replace with
    replicas.

    ``crossings`` lists chain ENDPOINTS that are genuine seam crossings:
    globally interior vertices whose fan is divided between two ranks. Their
    local fan is OPEN — bounded by the one local fault facet and the seam —
    so the walk cannot close; instead, walking anticlockwise from the local
    fault neighbour ``u`` to the fan's dead end sweeps exactly the LEFT side
    of the ray v->u, and the missing (other-rank) fault direction lies inside
    the gap. Whether that left sweep is Plus depends on whether ``u`` is
    forward or backward along the globally oriented chain.
    """
    side_of_cell, substitutions = {}, {}

    def _fan_step(v, fan):
        step = {}
        for c in fan:
            tri = [int(t) for t in verts[c - cS]]
            j = tri.index(v)
            pp, qq = tri[(j + 1) % 3], tri[(j + 2) % 3]
            if pp in step:
                return None
            step[pp] = (qq, c)
        return step

    def _assign(cells, side, v):
        for c in cells:
            if side_of_cell.setdefault(c, side) != side:
                return (ValueError,
                        "fault_split: a cell meets the fault from both "
                        "sides (the chain kinks through it). Refine the "
                        "mesh near the kink and re-cut.")
            if side < 0:
                substitutions.setdefault(c, set()).add(v)
        return None

    for v in (crossings or ()):
        u = chain[1] if v == chain[0] else chain[-2]
        u_is_forward = (v == chain[0])
        fan = [int(p) for p in dm.getTransitiveClosure(v, useCone=False)[0]
               if cS <= int(p) < cE]
        step = _fan_step(v, fan)
        if step is None:
            return (ValueError,
                    "fault_split: inconsistent fan orientation at a seam "
                    "crossing."), None, None
        swept, cur, guard = [], u, 0
        while cur in step and guard <= len(step):
            cur, c = step[cur]
            swept.append(c)
            guard += 1
        if cur == u and swept:
            return (ValueError,
                    "fault_split: the fan at a seam-crossing vertex closes "
                    "on this rank — it is not a crossing."), None, None
        rest = [c for c in fan if c not in set(swept)]
        left, right = (swept, rest)
        plus, minus = (left, right) if u_is_forward else (right, left)
        problem = _assign(plus, 1, v) or _assign(minus, -1, v)
        if problem is not None:
            return problem, None, None
    for i in range(1, len(chain) - 1):
        v, before, after = chain[i], chain[i - 1], chain[i + 1]
        fan = [int(p) for p in dm.getTransitiveClosure(v, useCone=False)[0]
               if cS <= int(p) < cE]
        step = {}
        for c in fan:
            tri = [int(t) for t in verts[c - cS]]
            j = tri.index(v)
            p, q = tri[(j + 1) % 3], tri[(j + 2) % 3]
            if p in step:
                return (ValueError,
                        "fault_split: two cells at a fault vertex claim the "
                        "same directed link edge — the fan is not a "
                        "consistently oriented manifold disc."), None, None
            step[p] = (q, c)

        plus_arc, cur, guard = [], after, 0
        while cur != before:
            if cur not in step or guard > len(step):
                return (ValueError,
                        "fault_split: the fan walk at a fault vertex did not "
                        "reach the incoming fault facet — the fan is not a "
                        "manifold disc."), None, None
            cur, c = step[cur]
            plus_arc.append(c)
            guard += 1
        minus_arc = [c for c in fan if c not in set(plus_arc)]
        if not plus_arc or not minus_arc:
            return (ValueError,
                    "fault_split: a fault vertex has all its cells on one "
                    "side, which cannot happen on a manifold interior "
                    "chain."), None, None

        # A cell can meet the fault at two chain vertices and be walked
        # onto opposite sides only when the chain kinks through it (a chord
        # cell); _assign refuses rather than resolves.
        problem = _assign(plus_arc, 1, v) or _assign(minus_arc, -1, v)
        if problem is not None:
            return problem, None, None
    return None, side_of_cell, substitutions


def _clone_labels(new, dm, clone_map):
    """Give each replica point its original's label values.

    :func:`reconnect._copy_labels` maps one source point to at most one new
    point; a split maps the duplicated ones to TWO. The replica's values are
    applied as a second pass over the same strata, filtered by nothing:
    ``Null_Boundary`` and ``UW_Boundaries`` are rebuilt from scratch when a
    ``Mesh`` is constructed on the result, so cloning them costs nothing, and
    every other label on a duplicated point (the fault's own name above all)
    is exactly what the replica must carry.
    """
    if not clone_map:
        return
    twins = {}
    for new_pt, old_pt in clone_map.items():
        twins.setdefault(old_pt, []).append(new_pt)
    for i in range(dm.getNumLabels()):
        name = dm.getLabelName(i)
        if name in _TOPOLOGY_LABELS:
            continue
        source, target = dm.getLabel(name), new.getLabel(name)
        values = source.getValueIS()
        if values is None:
            continue
        for val in values.getIndices():
            points = source.getStratumIS(int(val))
            if points is None:
                continue
            for p in points.getIndices():
                for q in twins.get(int(p), ()):
                    target.setValue(q, int(val))


def split_along_label(dm, name, value, plus_name, plus_value,
                      minus_name, minus_value, orientation=None,
                      verbose=False):
    """Duplicate the interior vertices of a labelled facet chain.

    Builds a fresh plex in which the cells on the Minus side of the chain are
    rewired to replica vertices, so the two sides of the fault share no points
    — and therefore no degrees of freedom in any FE space — except the two
    tips. The fault facets are doubled: the originals keep every label they
    carried and gain ``plus_name``; the replicas clone the originals' labels
    and gain ``minus_name``.

    COLLECTIVE. Every rank must call this, including ranks holding no part of
    the fault (they rebuild their local chart unchanged); a refusal raises the
    same error on every rank.

    Parameters
    ----------
    dm : PETSc.DMPlex
        Source mesh, 2-D simplex. Not modified.
    name : str
        Label holding the fault facets.
    value : int
        The label value marking them.
    plus_name, plus_value : str, int
        Label minted on the original (Plus-side) fault facets.
    minus_name, minus_value : str, int
        Label minted on the replica (Minus-side) fault facets.

    Returns
    -------
    new : PETSc.DMPlex
        The split mesh. Cells keep their source order, so per-cell data
        remains aligned.
    point_map : numpy.ndarray
        Chart-indexed source point -> new point. ``-1`` marks the Minus-side
        facets that were re-homed onto replicas (their replacements appear in
        ``clone_map`` instead).
    clone_map : dict
        ``{new point: source point}`` for every replica vertex and every
        re-homed facet — the provenance a field transfer needs.
    """
    if dm.getDimension() != 2:
        raise NotImplementedError(
            "split_along_label handles 2-D facet chains; use "
            "split_along_label_3d for a labelled facet patch in 3-D.")

    pStart, pEnd = dm.getChart()
    cS, cE = dm.getHeightStratum(0)
    vS, vE = dm.getDepthStratum(0)
    eS, eE = dm.getDepthStratum(1)
    X = _coords(dm)

    fault_edges = []
    # hasLabel, not getLabel-against-None: petsc4py hands back a wrapper
    # around a NULL label for a missing name, and the first stratum query on
    # it aborts the process rather than raising. Likewise an empty stratum
    # hands back a null IS that segfaults in getIndices() — a rank owning no
    # part of the fault is the normal case in parallel.
    if dm.hasLabel(name) and dm.getLabel(name).getStratumSize(int(value)) > 0:
        stratum = dm.getLabel(name).getStratumIS(int(value))
        fault_edges = [int(p) for p in stratum.getIndices()
                       if eS <= int(p) < eE]

    # COLLECTIVE, and reached on every rank whatever the local verdict:
    # the sharing flags feed the chain validation (seam facets are not the
    # domain boundary; a shared chain vertex is a crossing) and the cell
    # triples feed the rebuild.
    shared = _shared_points(dm)
    verts, _frozen = _cell_vertices_and_seam(dm, X, shared)

    # Global fault-facet count per vertex (COLLECTIVE, reached on every
    # rank): distinguishes, at a shared chain endpoint, a seam CROSSING
    # (2 facets worldwide) from a tip resting on the seam (1) and a junction
    # on the seam (>2).
    local_count = np.zeros(pEnd - pStart, dtype=np.int32)
    try:
        _nr, _il, _ir = dm.getPointSF().getGraph()
        leafset = set() if _il is None else {int(q) for q in _il}
    except (ValueError, TypeError):
        leafset = set()
    for e in fault_edges:
        # owned facets only: a facet shared across the seam is held by both
        # ranks and would be double-counted by the star-forest sum.
        if int(e) in leafset:
            continue
        for q in dm.getCone(e):
            local_count[int(q) - pStart] += 1
    global_count = _sf_sum(dm, local_count.copy())

    problem, chain = (None, None)
    if fault_edges:
        problem, chain = _fault_chain(dm, fault_edges, X, vS, shared, pStart,
                                      orientation=orientation)

    crossings = []
    if problem is None and chain:
        for v in {chain[0], chain[-1]}:
            if not shared[v - pStart]:
                continue
            total = int(global_count[v - pStart])
            if total == 2:
                crossings.append(v)
            elif total > 2:
                problem = (ValueError,
                           "fault_split: a junction sits on the partition "
                           "seam. Junctions are refused, and doubly so on a "
                           "seam — offset the segments (the J0 pattern).")
                break
            # total == 1: a true tip resting on the seam — unsplit, allowed.

    # A crossing is legal only when exactly the two fault-carrying ranks
    # share the vertex: a third rank holding cells there has no local fault
    # structure to classify them with, and its cells would stay welded.
    if problem is None:
        third = ((global_count[vS - pStart:vE - pStart] == 2)
                 & (local_count[vS - pStart:vE - pStart] == 0)
                 & (shared[vS - pStart:vE - pStart] > 0))
        if bool(third.any()):
            problem = (RuntimeError,
                       "fault_split: a seam crossing lands on a vertex "
                       "shared with a rank that carries none of the fault — "
                       "a corner of three or more subdomains. Move the "
                       "crossing point along the fault.")

    side_of_cell, substitutions = {}, {}
    if problem is None and fault_edges:
        problem, side_of_cell, substitutions = _take_sides(
            dm, chain, verts, cS, cE, crossings=crossings)

    fault_pair_edge = {}
    for e in fault_edges:
        a, b = (int(p) for p in dm.getCone(e))
        fault_pair_edge[(a, b) if a < b else (b, a)] = e

    if problem is None and fault_edges:
        # Every fault facet must have been classified onto opposite sides by
        # the fans of its endpoints; anything else is a mis-wiring that would
        # otherwise only surface as a wrong stress field.
        for e in fault_pair_edge.values():
            sides = sorted(side_of_cell.get(int(c), 0)
                           for c in dm.getSupport(e))
            if sides != [-1, 1]:
                problem = (ValueError,
                           "fault_split: the two cells of a fault facet were "
                           "not classified onto opposite sides — the side "
                           "assignment is inconsistent along the chain.")
                break

    # One synchronisation point for every refusal, so no rank raises while its
    # peers continue into the collective rebuild below and block.
    if uw.mpi.size > 1:
        gathered = uw.mpi.comm.allgather((problem, len(fault_edges)))
        problems = [p for p, _m in gathered if p is not None]
        # Every rank raises the SAME error, and a seam verdict from any rank
        # wins over the chain-fragment symptoms the other ranks report.
        seam = [p for p in problems if p[0] is RuntimeError]
        problem = seam[0] if seam else (problems[0] if problems else None)
        if problem is None and sum(m for _p, m in gathered) == 0:
            problem = (ValueError,
                       f"fault_split: no facets carry label {name!r} value "
                       f"{value} on any rank.")
    elif problem is None and not fault_edges:
        problem = (ValueError,
                   f"fault_split: no facets carry label {name!r} value "
                   f"{value}.")
    if problem is not None:
        exc, message = problem
        raise exc(message)

    # ----------------------------------------------------------- the rebuild
    # From here the shape follows reconnect.rebuild_without_vertices, with the
    # chart grown instead of compacted: replicas are appended to the vertex
    # stratum, and the edges are re-derived from the substituted cell list so
    # that a facet is reused — labels intact — exactly where its vertex pair
    # survived, and re-homed onto replicas where it did not.
    # Replicas: the chain interior, plus any seam-crossing endpoint — the
    # crossing vertex is globally interior, and its replica is created even
    # when this rank's Minus arc is empty, because the OTHER rank's replica
    # needs a root (or partner) to associate with.
    interior = (chain[1:-1] + crossings) if chain else []
    replicas_of = {v: pEnd + i for i, v in enumerate(sorted(interior))}
    original_of = {t: v for v, t in replicas_of.items()}

    cells = []
    for c in range(cS, cE):
        tri = [int(t) for t in verts[c - cS]]
        if c in substitutions:
            tri = [replicas_of[t] if t in substitutions[c] else t for t in tri]
        cells.append(tuple(tri))

    wanted = set()
    for a, b, c in cells:
        wanted.update(((a, b) if a < b else (b, a),
                       (b, c) if b < c else (c, b),
                       (c, a) if c < a else (a, c)))
    surv_e, pair_of, old_edge_of_pair = [], {}, {}
    for e in range(eS, eE):
        a, b = (int(p) for p in dm.getCone(e))
        pair = (a, b) if a < b else (b, a)
        old_edge_of_pair[pair] = e
        if pair in wanted:
            surv_e.append(e)
            pair_of[e] = pair
    fresh = sorted(wanted - set(pair_of.values()))
    edges = [pair_of[e] for e in surv_e] + fresh

    sizes = {"c": cE - cS, "v": (vE - vS) + len(interior), "e": len(edges)}
    offset, at = {}, pStart
    for _start, key in sorted(((cS, "c"), (vS, "v"), (eS, "e"))):
        offset[key] = at
        at += sizes[key]

    point_map = np.full(pEnd - pStart, -1, dtype=np.int64)
    point_map[np.arange(cS, cE) - pStart] = offset["c"] + np.arange(cE - cS)
    point_map[np.arange(vS, vE) - pStart] = offset["v"] + np.arange(vE - vS)
    if surv_e:
        point_map[np.asarray(surv_e, dtype=np.int64) - pStart] = (
            offset["e"] + np.arange(len(surv_e)))

    replica_new = {t: offset["v"] + (vE - vS) + i
                   for i, t in enumerate(sorted(original_of))}

    def v_new(x):
        return replica_new[x] if x >= pEnd else int(point_map[x - pStart])

    new = PETSc.DMPlex().create(comm=dm.comm)
    new.setDimension(2)
    new.setChart(pStart, at)
    for i in range(sizes["c"]):
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

    source_rows = np.concatenate([
        np.arange(vE - vS, dtype=np.int64),
        np.asarray(sorted(interior), dtype=np.int64) - vS]) \
        if interior else np.arange(vE - vS, dtype=np.int64)
    _write_coordinates(new, dm, (offset["v"], offset["v"] + sizes["v"]),
                       source_rows)

    clone_map = {replica_new[t]: original_of[t] for t in original_of}
    minus_copies = []
    for i, pair in enumerate(fresh):
        e_new = offset["e"] + len(surv_e) + i
        a, b = (original_of.get(p, p) for p in pair)
        collapsed = (a, b) if a < b else (b, a)
        e_old = old_edge_of_pair.get(collapsed)
        if e_old is None:
            # The split invents no connectivity: every fresh facet is an old
            # one re-homed onto replicas. A missing source is a wiring bug.
            raise RuntimeError(
                "fault_split internal: a fresh facet has no source facet.")
        clone_map[e_new] = e_old
        if collapsed in fault_pair_edge:
            minus_copies.append(e_new)

    _copy_labels(new, dm, point_map)
    _clone_labels(new, dm, clone_map)

    new.createLabel(plus_name)
    plus = new.getLabel(plus_name)
    for e in fault_pair_edge.values():
        plus.setValue(int(point_map[e - pStart]), int(plus_value))
    new.createLabel(minus_name)
    minus = new.getLabel(minus_name)
    for e in minus_copies:
        minus.setValue(int(e), int(minus_value))

    if uw.mpi.size > 1:
        # The replica/replacement map rides a second broadcast so a leaf rank
        # can associate ITS replica of a shared point with the owner's — the
        # keyed exchange that makes a seam crossing one star-forest entry.
        dup_new = np.full(pEnd - pStart, -1, dtype=np.int64)
        for new_pt, old_pt in clone_map.items():
            dup_new[old_pt - pStart] = new_pt
        _rebuild_point_sf(new, dm, point_map, at - pStart, dup_new=dup_new)

    if verbose and fault_edges:
        uw.pprint(f"[fault_split {name!r}] duplicated {len(interior)} "
                  f"vertices along {len(fault_edges)} facets; sides are "
                  f"{plus_name!r} / {minus_name!r}")
    return new, point_map, clone_map


def _patch_faces_and_edges(dm, fault_faces):
    """Classify the labelled patch: rim vs interior edges and vertices.

    Returns ``(face_verts, edge_faces, rim_edges, interior_edges, rim_verts,
    interior_verts, problem)`` where ``face_verts[f]`` is the face's vertex
    triple and ``edge_faces`` maps each patch edge to the fault faces using
    it. A patch edge used by one fault face is the RIM (the tip rule one
    dimension up — those points stay unsplit); by two, patch interior; by
    more, a non-manifold junction, refused.
    """
    vS, vE = dm.getDepthStratum(0)
    face_verts, edge_faces = {}, {}
    for f in fault_faces:
        closure, _ = dm.getTransitiveClosure(f)
        face_verts[f] = [int(p) for p in closure if vS <= int(p) < vE]
        for e in dm.getCone(f):
            edge_faces.setdefault(int(e), []).append(f)

    problem = None
    if any(len(fs) > 2 for fs in edge_faces.values()):
        problem = (ValueError,
                   "fault_split: three or more fault faces meet along an "
                   "edge — the patch is non-manifold (a branching sheet). "
                   "Label each sheet separately.")
    rim_edges = [e for e, fs in edge_faces.items() if len(fs) == 1]
    interior_edges = [e for e, fs in edge_faces.items() if len(fs) == 2]
    rim_verts = set()
    for e in rim_edges:
        rim_verts.update(int(q) for q in dm.getCone(e))
    all_verts = set()
    for vv in face_verts.values():
        all_verts.update(vv)
    interior_verts = all_verts - rim_verts
    return (face_verts, edge_faces, rim_edges, interior_edges,
            rim_verts, interior_verts, problem)


def _orient_patch(face_verts, edge_faces, X, vS, orientation=None):
    """Give every fault face a consistently oriented vertex triple.

    Adjacent faces must traverse their shared edge in opposite directions —
    the standard orientability propagation over the patch's face-adjacency
    graph. The propagation also proves the patch is one connected, orientable
    sheet; either failure is refused. The global sign — which side is Plus —
    is the side the oriented normal points into: the ``orientation`` vector
    when given, otherwise the sign making the largest component of the
    area-weighted mean normal positive (deterministic, documented, and no
    more meaningful than 2-D's coordinate-order tip rule).

    Returns ``(oriented, problem)`` with ``oriented[f]`` an ordered triple.
    """
    faces = list(face_verts)
    oriented, problem = {}, None

    def _directed(tri):
        a, b, c = tri
        return {(a, b), (b, c), (c, a)}

    stack = [faces[0]]
    oriented[faces[0]] = list(face_verts[faces[0]])
    while stack:
        f = stack.pop()
        dirs = _directed(oriented[f])
        for e, fs in edge_faces.items():
            if f not in fs or len(fs) != 2:
                continue
            g = fs[0] if fs[1] == f else fs[1]
            shared = [v for v in face_verts[g] if v in face_verts[f]]
            u, w = shared[0], shared[1]
            forward = (u, w) if (u, w) in dirs else (w, u)
            if g not in oriented:
                tri = list(face_verts[g])
                if forward in _directed(tri):
                    tri = [tri[0], tri[2], tri[1]]
                oriented[g] = tri
                stack.append(g)
            elif forward in _directed(oriented[g]):
                return oriented, (ValueError,
                                  "fault_split: the patch is not orientable "
                                  "— no consistent two-sided normal exists.")
    if len(oriented) != len(faces):
        return oriented, (ValueError,
                          "fault_split: the labelled faces form more than "
                          "one connected patch. Split one fault at a time — "
                          "label each patch under its own name.")

    normal_sum = np.zeros(3)
    for tri in oriented.values():
        a, b, c = (X[t - vS] for t in tri)
        normal_sum += np.cross(b - a, c - a)
    reference = (np.asarray(orientation, dtype=float)
                 if orientation is not None
                 else np.eye(3)[int(np.argmax(np.abs(normal_sum)))])
    if float(normal_sum @ reference) < 0.0:
        for f in oriented:
            t0, t1, t2 = oriented[f]
            oriented[f] = [t0, t2, t1]
    return oriented, problem


def _take_sides_3d(dm, oriented, face_verts, cell_verts, interior_verts,
                   fault_face_set, X, vS, cS):
    """Assign every patch-adjacent cell to the Plus or Minus half-ball.

    The two support cells of each fault face are classified geometrically:
    with the face's oriented triple :math:`(a, b, c)` and the cell's fourth
    vertex :math:`d`, the sign of :math:`\\det[b-a,\\; c-a,\\; d-a]` says
    which side of the oriented plane the cell lies on (positive = Plus, the
    side the normal points into). Around each interior patch vertex the rest
    of the star is then flooded: cells connect through shared non-fault faces
    containing the vertex, the patch cuts that star into exactly two
    components, and each component inherits the side of the fault-face
    supports it contains. Anything else — one component (a pinched sheet),
    an unseeded component, conflicting seeds — is a mis-labelling, refused.

    Returns ``(side_of_cell, substitutions, problem)``; ``substitutions[c]``
    is the set of vertices cell ``c`` must swap for replicas.
    """
    side_of_cell, substitutions = {}, {}
    for f in fault_face_set:
        a, b, c = (X[t - vS] for t in oriented[f])
        for cell in (int(q) for q in dm.getSupport(f)):
            tet = cell_verts[cell - cS]
            d = next(t for t in tet if t not in face_verts[f])
            volume = float(np.linalg.det(
                np.array([b - a, c - a, X[d - vS] - a])))
            if volume == 0.0:
                return {}, {}, (ValueError,
                                "fault_split: a fault-flanking cell is "
                                "degenerate — zero volume against the fault "
                                "plane.")
            side = 1 if volume > 0.0 else -1
            if side_of_cell.setdefault(cell, side) != side:
                return {}, {}, (ValueError,
                                "fault_split: a cell flanks fault faces "
                                "from opposite sides — the patch folds "
                                "back through its own star.")

    star = {v: [] for v in interior_verts}
    for i, tet in enumerate(cell_verts):
        for t in tet:
            if int(t) in star:
                star[int(t)].append(cS + i)

    vS_dm, vE_dm = dm.getDepthStratum(0)
    verts_of_face = dict(face_verts)

    def _face_verts(f):
        if f not in verts_of_face:
            closure, _ = dm.getTransitiveClosure(f)
            verts_of_face[f] = [int(p) for p in closure
                                if vS_dm <= int(p) < vE_dm]
        return verts_of_face[f]

    for v, cells in star.items():
        in_star = set(cells)
        neighbours = {c: [] for c in cells}
        for c in cells:
            for f in (int(q) for q in dm.getCone(c)):
                if f in fault_face_set or v not in _face_verts(f):
                    continue
                for other in (int(q) for q in dm.getSupport(f)):
                    if other != c and other in in_star:
                        neighbours[c].append(other)
        components, seen = [], set()
        for c in cells:
            if c in seen:
                continue
            component, queue = [], [c]
            seen.add(c)
            while queue:
                q = queue.pop()
                component.append(q)
                for r in neighbours[q]:
                    if r not in seen:
                        seen.add(r)
                        queue.append(r)
            components.append(component)
        if len(components) != 2:
            return {}, {}, (ValueError,
                            "fault_split: the patch does not cut a vertex "
                            "star into two half-balls — the sheet is "
                            "pinched or the label leaks off the patch.")
        for component in components:
            seeds = {side_of_cell[c] for c in component if c in side_of_cell}
            if len(seeds) != 1:
                return {}, {}, (ValueError,
                                "fault_split: a star component seeds from "
                                "both sides (or neither) — the side "
                                "assignment is inconsistent at a vertex.")
            side = seeds.pop()
            for c in component:
                if side_of_cell.setdefault(c, side) != side:
                    return {}, {}, (ValueError,
                                    "fault_split: a cell is assigned to "
                                    "opposite sides by two vertex stars.")
                if side < 0:
                    substitutions.setdefault(c, set()).add(v)
    return side_of_cell, substitutions, None


def split_along_label_3d(dm, name, value, plus_name, plus_value,
                         minus_name, minus_value, orientation=None,
                         verbose=False):
    """Duplicate the interior of a labelled facet patch in a 3-D mesh.

    The 3-D counterpart of :func:`split_along_label`: the fault is a
    triangulated PATCH of interior faces, its RIM (the boundary edges and
    vertices of the patch) is the tip rule one dimension up and stays
    unsplit, and everything strictly inside the rim — vertices, edges, and
    the faces themselves — is doubled. Slip data must vanish on the rim.

    Where the 2-D rebuild hand-wires edge cones, this one builds the
    substituted cells UNINTERPOLATED (vertex 4-tuples carry no orientation)
    and lets ``DMPlexInterpolate`` derive faces and edges — cell and vertex
    numbering are preserved, and every old face or edge is recovered in the
    new chart by joining its (mapped) vertex tuple. A tuple that no longer
    joins was re-homed onto replicas; its replacement is found by collapsing
    replicas back to originals, exactly the 2-D fresh-facet rule.

    COLLECTIVE. Refusals beyond the 2-D list: a non-manifold or pinched or
    non-orientable patch; a patch so coarse that a fault face has no
    interior vertex (its two copies would be the same vertex triple — the
    3-D analogue of the single-facet chain); and, in this version, ANY
    contact between the patch's cell star and the partition seam — 3-D seam
    crossings are a curve, not a point, and are not yet designed.

    Parameters and returns match :func:`split_along_label`, with
    ``orientation`` a reference NORMAL vector (Plus is the side it points
    into) rather than a chain direction.
    """
    if dm.getDimension() != 3:
        raise NotImplementedError(
            "split_along_label_3d handles 3-D meshes; use split_along_label "
            "for a 2-D facet chain.")

    pStart, pEnd = dm.getChart()
    cS, cE = dm.getHeightStratum(0)
    vS, vE = dm.getDepthStratum(0)
    fS, fE = dm.getHeightStratum(1)
    eS, eE = dm.getDepthStratum(1)
    X = _coords(dm)
    nc, nv = cE - cS, vE - vS

    fault_faces = []
    # hasLabel + stratum-size guards for the petsc4py NULL-wrapper aborts,
    # exactly as in the 2-D path.
    if dm.hasLabel(name) and dm.getLabel(name).getStratumSize(int(value)) > 0:
        stratum = dm.getLabel(name).getStratumIS(int(value))
        fault_faces = [int(p) for p in stratum.getIndices()
                       if fS <= int(p) < fE]
    fault_face_set = set(fault_faces)

    shared = _shared_points(dm)

    cell_verts = np.empty((nc, 4), dtype=np.int64)
    for c in range(cS, cE):
        closure, _ = dm.getTransitiveClosure(c)
        cell_verts[c - cS] = [int(p) for p in closure if vS <= int(p) < vE]

    problem = None
    interior_verts, interior_edges, rim_verts = set(), [], set()
    face_verts, oriented, substitutions = {}, {}, {}
    if fault_faces:
        (face_verts, edge_faces, _rim_edges, interior_edges, rim_verts,
         interior_verts, problem) = _patch_faces_and_edges(dm, fault_faces)

        # v1 seam rule, deliberately the most conservative one: the whole
        # cell star of the patch must be rank-interior. A 3-D seam crossing
        # is a CURVE of shared points, a genuinely new design, not the 2-D
        # crossing-vertex rule applied pointwise.
        #
        # Diagnosed FIRST, before any check that interprets this rank's
        # patch topology: a rank holding only a ghost FRAGMENT of the
        # patch sees the fragment's edge as "rim", its faces as
        # support-1, its interior as empty — every later check would
        # mis-fire with a misleading message where the true verdict is
        # the seam. (The union rim | interior used here is
        # fragment-correct even though the classification is not.)
        if problem is None and bool(shared.any()):
            patch_verts = rim_verts | interior_verts
            for i, tet in enumerate(cell_verts):
                if not (set(int(t) for t in tet) & patch_verts):
                    continue
                closure, _ = dm.getTransitiveClosure(cS + i)
                if shared[np.asarray(closure, dtype=np.int64)
                          - pStart].any():
                    problem = (RuntimeError,
                               "fault_split: the fault patch's cell star "
                               "touches the partition seam. 3-D faults must "
                               "be rank-interior in this version — "
                               "partition around the fault.")
                    break

        if problem is None and any(
                dm.getSupportSize(f) != 2 for f in fault_faces):
            problem = (ValueError,
                       "fault_split: a fault face lies on the domain "
                       "boundary — the patch rim must be strictly interior.")

        if problem is None:
            patch_verts = rim_verts | interior_verts
            boundary_verts = set()
            for f in range(fS, fE):
                if dm.getSupportSize(f) == 1 and not shared[f - pStart]:
                    boundary_verts.update(
                        int(p) for p in dm.getTransitiveClosure(f)[0]
                        if vS <= int(p) < vE)
            if patch_verts & boundary_verts:
                problem = (ValueError,
                           "fault_split: the patch touches the domain "
                           "boundary. A fault that daylights is not yet "
                           "supported — keep the rim strictly interior.")

        if problem is None and any(
                not (set(face_verts[f]) & interior_verts)
                for f in fault_faces):
            problem = (ValueError,
                       "fault_split: a fault face has no interior vertex, "
                       "so its two copies would carry the same vertex "
                       "triple. The patch is too coarse to split — refine "
                       "it until every face reaches inside the rim.")

        if problem is None:
            oriented, problem = _orient_patch(face_verts, edge_faces, X, vS,
                                              orientation=orientation)

        side_of_cell = {}
        if problem is None:
            side_of_cell, substitutions, problem = _take_sides_3d(
                dm, oriented, face_verts, cell_verts, interior_verts,
                fault_face_set, X, vS, cS)

        if problem is None:
            for f in fault_faces:
                sides = sorted(side_of_cell.get(int(q), 0)
                               for q in dm.getSupport(f))
                if sides != [-1, 1]:
                    problem = (ValueError,
                               "fault_split: the two cells of a fault face "
                               "were not classified onto opposite sides — "
                               "the side assignment is inconsistent across "
                               "the patch.")
                    break

    # One synchronisation point for every refusal — the 2-D contract.
    if uw.mpi.size > 1:
        gathered = uw.mpi.comm.allgather((problem, len(fault_faces)))
        problems = [p for p, _m in gathered if p is not None]
        seam = [p for p in problems if p[0] is RuntimeError]
        problem = seam[0] if seam else (problems[0] if problems else None)
        if problem is None and sum(m for _p, m in gathered) == 0:
            problem = (ValueError,
                       f"fault_split: no faces carry label {name!r} value "
                       f"{value} on any rank.")
    elif problem is None and not fault_faces:
        problem = (ValueError,
                   f"fault_split: no faces carry label {name!r} value "
                   f"{value}.")
    if problem is not None:
        exc, message = problem
        raise exc(message)

    # ----------------------------------------------------------- the rebuild
    interior_sorted = sorted(interior_verts)
    replica_index = {v: i for i, v in enumerate(interior_sorted)}
    k = len(interior_sorted)

    def v_uninterp(t, subs):
        if t in subs:
            return nc + nv + replica_index[t]
        return nc + (t - vS)

    new = PETSc.DMPlex().create(comm=dm.comm)
    new.setDimension(3)
    new.setChart(0, nc + nv + k)
    for i in range(nc):
        new.setConeSize(i, 4)
    new.setUp()
    empty = frozenset()
    for c in range(cS, cE):
        subs = substitutions.get(c, empty)
        new.setCone(c - cS, [v_uninterp(int(t), subs)
                             for t in cell_verts[c - cS]])
    new.symmetrize()
    new.stratify()
    new.interpolate()

    vS2, vE2 = new.getDepthStratum(0)
    if (new.getHeightStratum(0) != (0, nc)
            or (vS2, vE2) != (nc, nc + nv + k)):
        raise RuntimeError(
            "fault_split internal: DMPlexInterpolate moved the cell or "
            "vertex numbering — the point-map arithmetic below relies on "
            "both being preserved.")

    source_rows = np.concatenate([
        np.arange(nv, dtype=np.int64),
        np.asarray(interior_sorted, dtype=np.int64) - vS]) \
        if k else np.arange(nv, dtype=np.int64)
    _write_coordinates(new, dm, (vS2, vE2), source_rows)

    point_map = np.full(pEnd - pStart, -1, dtype=np.int64)
    point_map[np.arange(cS, cE) - pStart] = np.arange(nc)
    point_map[np.arange(vS, vE) - pStart] = vS2 + np.arange(nv)

    def v_new(t):
        return int(vS2 + (t - vS))

    def replica_new(t):
        return int(vS2 + nv + replica_index[t])

    clone_map = {replica_new(v): v for v in interior_sorted}

    # Every old face and edge is recovered by joining its vertex tuple in the
    # new chart: found directly, it survived (Plus side keeps the original
    # vertices); found only after replica substitution, it was re-homed onto
    # the Minus side. The doubled points — fault faces and interior patch
    # edges — are found BOTH ways: the original maps, the substituted copy
    # clones. The split invents no connectivity, so a tuple found neither
    # way is a wiring bug, raised.
    interior_edge_set = set(interior_edges)
    minus_faces = []
    for p, tuple_size in ((slice(fS, fE), 3), (slice(eS, eE), 2)):
        for q in range(p.start, p.stop):
            closure, _ = dm.getTransitiveClosure(q)
            verts = [int(t) for t in closure if vS <= int(t) < vE]
            direct = new.getFullJoin([v_new(t) for t in verts])
            doubled = (q in fault_face_set) or (q in interior_edge_set)
            if len(direct) == 1:
                point_map[q - pStart] = int(direct[0])
            subs_tuple = [replica_new(t) if t in replica_index else v_new(t)
                          for t in verts]
            if doubled or len(direct) != 1:
                rehomed = new.getFullJoin(subs_tuple)
                if len(rehomed) != 1:
                    raise RuntimeError(
                        "fault_split internal: an old point joins to "
                        f"{len(direct)} direct and {len(rehomed)} "
                        "substituted points in the new chart.")
                clone_map[int(rehomed[0])] = q
                if q in fault_face_set:
                    minus_faces.append(int(rehomed[0]))

    _copy_labels(new, dm, point_map)
    _clone_labels(new, dm, clone_map)

    new.createLabel(plus_name)
    plus = new.getLabel(plus_name)
    for f in fault_faces:
        plus.setValue(int(point_map[f - pStart]), int(plus_value))
    new.createLabel(minus_name)
    minus = new.getLabel(minus_name)
    for f in minus_faces:
        minus.setValue(int(f), int(minus_value))

    if uw.mpi.size > 1:
        # No dup_new: the seam rule above guarantees every shared point is
        # far from the patch, so the leaf set carries over by renumbering
        # alone — the 2-D no-crossing case.
        _rebuild_point_sf(new, dm, point_map, new.getChart()[1])

    if verbose and fault_faces:
        uw.pprint(f"[fault_split {name!r}] duplicated {k} vertices, "
                  f"{len(interior_edges)} edges and {len(fault_faces)} "
                  f"faces of the patch; sides are {plus_name!r} / "
                  f"{minus_name!r}")
    return new, point_map, clone_map


def _pull_seam_vertices_onto_crossings(dm, poly):
    """Move a SHARED vertex onto each fault-polyline x partition-seam
    intersection; return the (possibly copied) dm.

    The seam-crossing rule needs the fault to pass through a vertex that is
    already shared between the two ranks — only then are the two adjacent
    fault facets rank-local and the fan halves walkable. The nearest vertex
    in general is NOT shared (sharing is topology, not position), so this is
    ``pull_vertex_onto`` restricted to shared vertices: intersections are
    computed rank-locally on the shared (seam) edges, allgathered and
    deduplicated so every rank sees the same targets, and the winning vertex
    is chosen by the same global (distance, x, y) reduction — every rank
    moves its own copy of the same shared vertex, keeping the coordinates of
    the copies identical. COLLECTIVE; a serial run returns immediately.
    """
    if uw.mpi.size == 1:
        return dm

    from underworld3.utilities.reconnect import _coords, _shared_points

    shared = _shared_points(dm)
    pStart, _pEnd = dm.getChart()
    vS, vE = dm.getDepthStratum(0)
    eS, eE = dm.getDepthStratum(1)
    X = _coords(dm)
    P = np.asarray(poly, dtype=float)

    def _segment_hits(a, b):
        """Intersections of seam edge (a, b) with the polyline's segments."""
        hits = []
        d1 = b - a
        for s0, s1 in zip(P[:-1], P[1:]):
            d2 = s1 - s0
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-14:
                continue
            r = ((s0[0] - a[0]) * d2[1] - (s0[1] - a[1]) * d2[0]) / den
            t = ((s0[0] - a[0]) * d1[1] - (s0[1] - a[1]) * d1[0]) / den
            if -1e-9 <= r <= 1 + 1e-9 and -1e-9 <= t <= 1 + 1e-9:
                hits.append(a + r * d1)
        return hits

    local_hits = []
    for e in range(eS, eE):
        if not shared[e - pStart]:
            continue
        a, b = (X[int(q) - vS] for q in dm.getCone(e))
        local_hits.extend(_segment_hits(np.asarray(a), np.asarray(b)))

    gathered = uw.mpi.comm.allgather(
        [tuple(np.round(h, 12)) for h in local_hits])
    targets = sorted({h for rank_hits in gathered for h in rank_hits})
    if not targets:
        return dm

    arr = X.copy()
    # Prefer vertices shared by EXACTLY two ranks: a crossing pulled onto a
    # subdomain corner (three or more holders) is refused downstream, so the
    # corner vertices are excluded whenever a two-rank seam vertex exists.
    # The holder count is an SF-sum of ones.
    mult = _sf_sum(dm, np.ones(len(shared), dtype=np.int32))
    two_rank = np.flatnonzero((shared[vS - pStart:vE - pStart] > 0)
                              & (mult[vS - pStart:vE - pStart] == 2))
    # COLLECTIVE choice of candidate pool, so every rank reduces over the
    # same set. Local emptiness is fine (that rank bids infinity).
    if uw.mpi.comm.allreduce(len(two_rank), op=MPI.SUM) > 0:
        cand = two_rank
    else:
        cand = np.flatnonzero(shared[vS - pStart:vE - pStart])
    every_shared = np.flatnonzero(shared[vS - pStart:vE - pStart])

    def _nearest(pool, t):
        if len(pool):
            d = np.linalg.norm(arr[pool] - t, axis=1)
            k = int(d.argmin())
            return (float(d[k]), float(arr[pool[k], 0]),
                    float(arr[pool[k], 1]))
        return (np.inf, np.inf, np.inf)

    for t in targets:
        t = np.asarray(t, dtype=float)
        # The local cell scale at the crossing: distance to the nearest
        # vertex of any kind. A pull much longer than this squashes cells
        # and the cut's inverted-cell guard then refuses the whole mesh, so
        # a distant two-rank vertex loses to a nearby corner vertex — the
        # corner is refused DOWNSTREAM with a clear message, which beats an
        # inverted-cell refusal here.
        d_all = np.linalg.norm(arr - t, axis=1)
        h_local = uw.mpi.comm.allreduce(
            float(d_all.min()) if len(d_all) else np.inf, op=MPI.MIN)
        dist, cx, cy = uw.mpi.comm.allreduce(_nearest(cand, t), op=MPI.MIN)
        if dist > 3.0 * max(h_local, 1e-12):
            dist, cx, cy = uw.mpi.comm.allreduce(
                _nearest(every_shared, t), op=MPI.MIN)
        move = np.flatnonzero((np.abs(arr[:, 0] - cx) < 1e-12)
                              & (np.abs(arr[:, 1] - cy) < 1e-12))
        arr[move] = t

    out = dm.clone()
    cvec = out.getCoordinatesLocal()
    cvec.array[:] = arr.reshape(-1)
    out.setCoordinatesLocal(cvec)
    return out


def add_fault(mesh, faults, verbose=False):
    """Cut AND split one or more faults into a mesh; return the split Mesh.

    The one-call form of the split-node pipeline: for each fault, the tips
    are placed onto mesh vertices, the mesh is cut so the fault becomes a
    conforming facet chain, and the chain is split into a genuine
    discontinuity with boundaries ``<name>Plus`` / ``<name>Minus`` and the
    coincident DOF pairing recorded. Slip conditions then go through
    ``solver.add_fault_bc(conds, name)`` and an ordinary ``solve()``.

    Parameters
    ----------
    mesh : Mesh
        The mesh to fault — typically an adapted child already refined
        toward the fault. Not modified; the fault position stays a design
        variable (re-call on the base when it moves).
    faults : Surface, (name, points), or a sequence of either
        Each fault is a single open polyline with both tips strictly inside
        the domain. A sequence is a NETWORK: every fault is cut first, then
        every fault is split, so disjoint (offset-junction) networks work in
        one call. Segments must not share vertices — represent a branch or
        crossing as offset segments (a ligament of one or two local cell
        sizes), the J0 pattern of the deployment design.
    verbose : bool, optional
        Report each cut and split.

    Returns
    -------
    Mesh
        The split mesh, carrying every fault's side boundaries and pairing.
    """
    from enum import Enum

    from underworld3.discretisation import Mesh
    from underworld3.meshing.surfaces import (Surface,
                                              _fault_collect_polylines)
    from underworld3.utilities.line_cut import (cut_along_lines,
                                                pull_vertex_onto)

    if isinstance(faults, (Surface, tuple)) or (
            hasattr(faults, "name") and hasattr(faults, "control_points")):
        faults = [faults]

    segments = []
    for entry in faults:
        if isinstance(entry, tuple) and len(entry) == 2 \
                and isinstance(entry[0], str):
            name, points = entry
            polylines = [np.asarray(points, dtype=float)]
        else:
            name = entry.name
            polylines = [np.array([segs[0][0]] + [b for _a, b in segs])
                         for segs in _fault_collect_polylines(entry)]
        if len(polylines) != 1:
            raise ValueError(
                f"fault {name!r} holds {len(polylines)} polylines; a fault "
                "segment is ONE open polyline — pass a sequence of "
                "single-segment faults for a network.")
        segments.append((name, polylines[0]))

    # One boundary value per fault, minted with _boundaries_with's rule
    # (first free ordinary value, stepped past anything taken).
    members = {b.name: b.value for b in mesh.boundaries}
    values = {}
    for name, _poly in segments:
        if name in members:
            raise ValueError(
                f"this mesh already has a boundary called {name!r}.")
        taken = set(members.values())
        ordinary = [v for v in taken if v < 666]
        candidate = (max(ordinary) + 1) if ordinary else 1
        while candidate in taken:
            candidate += 1
        members[name] = values[name] = candidate

    dm = mesh.dm
    for name, poly in segments:
        # EVERY control point gets a vertex, not just the tips: an interior
        # kink is the same problem as a tip — a distinguished point of the
        # geometry that must coincide with a mesh vertex, or the cut leaves
        # the chain fragmented at the turn.
        dm = pull_vertex_onto(dm, np.asarray(poly, dtype=float))
        # Manufacture CLEAN seam crossings: a shared vertex pulled exactly
        # onto each fault x seam intersection, so the cut chain passes
        # through a seam VERTEX with its two fault facets rank-local — the
        # crossing rule's premise. Without this the cut produces a facet
        # STRADDLING the seam (held by both ranks, one support each), which
        # the split rightly refuses as untractable.
        dm = _pull_seam_vertices_onto_crossings(dm, poly)
        dm, info = cut_along_lines(dm, [poly], label=name,
                                   label_value=values[name])
        if verbose:
            uw.pprint(f"[add_fault {name!r}] {info['n_cut_edges']} facets, "
                      f"min angle {info['min_angle']:.2f} deg")

    cut = Mesh(dm, simplex=mesh.dm.isSimplex(),
               coordinate_system_type=mesh.CoordinateSystem.coordinate_type,
               qdegree=mesh.qdegree,
               boundaries=Enum("boundaries", members), verbose=False)
    cut.parent = mesh
    cut._relationship_kind = "refinement"
    cut._refine_dofs_coincide = False
    cut.regions = mesh.regions
    cut._parent_mesh_version = mesh._mesh_version
    mesh._registered_children.add(cut)

    out = cut
    for name, poly in segments:
        out = split_fault(out, name, orientation=poly[-1] - poly[0],
                          verbose=verbose)
    # the traces themselves, for trace-smoothed fault normals
    # (add_fault_bc(..., normal="trace")) and for rendering overlays
    out._fault_traces = {name: np.array(poly, dtype=float)
                         for name, poly in segments}
    return out


def _boundaries_with_sides(mesh, name):
    """The mesh's boundary enum extended with ``<name>Plus`` / ``<name>Minus``.

    The value-minting rule is :meth:`Mesh._boundaries_with`'s — first free
    value past the largest ordinary boundary, stepped past anything taken so
    the sentinels 666/1001 can never be landed on — applied twice, since that
    method can only add one member per constructed mesh.
    """
    from enum import Enum

    members = {b.name: b.value for b in mesh.boundaries}
    if name not in members:
        raise ValueError(
            f"this mesh has no boundary called {name!r}; add the surface with "
            "add_conforming_surface first, then split it.")
    for side in (f"{name}Plus", f"{name}Minus"):
        if side in members:
            raise ValueError(
                f"this mesh already has a boundary called {side!r}; the fault "
                "appears to have been split already.")
        taken = set(members.values())
        ordinary = [v for v in taken if v < 666]
        candidate = (max(ordinary) + 1) if ordinary else 1
        while candidate in taken:
            candidate += 1
        members[side] = candidate
    return Enum("boundaries", members)


def _redistribute_fault_interior(dm, name, value, verbose=False):
    """Redistribute a cut 3-D mesh so the patch's cell star is rank-interior.

    The default partition's balance cuts are ATTRACTED to the locally
    refined patch region, so at any np >= 2 a graded fault mesh
    essentially always violates the 3-D split's rank-interior
    requirement. This reassigns ONLY the patch-star cells (cells
    incident to any patch vertex — a layer about two cells thick) to
    the single rank that already owns most of them; every other cell
    stays where the load-balanced partition put it, and the move is
    applied with a shell partitioner. The star is thin, so the
    imbalance cost is bounded by its size — NOT by the refined band —
    and the split's seam rule then holds by construction. The
    coincident pair blocks of the contact solve are rank-local for the
    same reason. Returns a NEW dm; the input is untouched.
    """
    from mpi4py import MPI
    from petsc4py import PETSc

    comm = dm.getComm().tompi4py()
    if comm.size == 1:
        return dm

    work = dm.clone()
    cS, cE = work.getHeightStratum(0)
    fS, fE = work.getHeightStratum(1)
    vS, vE = work.getDepthStratum(0)
    _pS, pEnd = work.getChart()
    sf = work.getPointSF()

    def propagate(mark):
        # global OR of a chart-length mark across the point SF: remote
        # copies fold into the owner, the owner's verdict returns to
        # every copy — after this, every rank agrees on every point it
        # can see. (A cell can touch a patch vertex without holding any
        # labelled face in its own closure, so purely local label
        # reading under-marks near the current seam.)
        tmp = mark.copy()
        sf.reduceBegin(MPI.INT32_T, tmp, mark, MPI.MAX)
        sf.reduceEnd(MPI.INT32_T, tmp, mark, MPI.MAX)
        out = mark.copy()
        sf.bcastBegin(MPI.INT32_T, mark, out, MPI.REPLACE)
        sf.bcastEnd(MPI.INT32_T, mark, out, MPI.REPLACE)
        return np.maximum(mark, out)

    def star_of_marked(mark):
        cells = set()
        for v in range(vS, vE):
            if mark[v]:
                for q in work.getTransitiveClosure(v, useCone=False)[0]:
                    if cS <= int(q) < cE:
                        cells.add(int(q))
        return cells

    # patch vertices, marked globally
    mark = np.zeros(pEnd, dtype=np.int32)
    if work.hasLabel(name) and \
            work.getLabel(name).getStratumSize(value) > 0:
        for f in work.getLabel(name).getStratumIS(value).getIndices():
            if fS <= int(f) < fE:
                for q in work.getTransitiveClosure(int(f))[0]:
                    if vS <= int(q) < vE:
                        mark[int(q)] = 1
    mark = propagate(mark)
    star = star_of_marked(mark)

    # one growth layer: the split's seam rule needs every point in the
    # CLOSURE of a patch-vertex-star cell unshared, and a point is
    # unshared exactly when all its incident cells are co-resident —
    # so gather every cell touching any vertex of the star cells'
    # closures. One layer is exactly sufficient (each such point's
    # incident cells all touch a marked-closure vertex).
    mark2 = np.zeros(pEnd, dtype=np.int32)
    for c in star:
        for q in work.getTransitiveClosure(c)[0]:
            if vS <= int(q) < vE:
                mark2[int(q)] = 1
    mark2 = propagate(mark2)
    star |= star_of_marked(mark2)

    counts = np.asarray(comm.allgather(len(star)))
    if counts.sum() == 0:
        raise RuntimeError(
            f"fault_split: no faces labelled {name!r} found on any rank.")
    target = int(np.argmax(counts))

    n_local = cE - cS
    assign = np.full(n_local, comm.rank, dtype=np.int32)
    for c in star:
        assign[c - cS] = target
    order = np.argsort(assign, kind="stable").astype(np.int32)
    sizes = np.bincount(assign, minlength=comm.size).astype(np.int32)

    part = work.getPartitioner()
    part.setType(PETSc.Partitioner.Type.SHELL)
    part.setShellPartition(comm.size, sizes=sizes, points=order)
    work.distribute()
    if verbose:
        uw.pprint(f"[fault_split] {name!r}: star of "
                  f"{int(counts.sum())} cells gathered onto rank "
                  f"{target}; local cells now "
                  f"{work.getHeightStratum(0)[1]}")
    return work


def split_fault(mesh, name, orientation=None, verbose=False):
    """Split the nodes along the conforming surface ``name``; return the mesh.

    The result is a new, standalone :class:`Mesh` on which every continuous FE
    space is discontinuous across the fault: the surface's facets are doubled
    into boundaries ``<name>Plus`` and ``<name>Minus`` (coincident, no shared
    DOFs except the tips — the two chain endpoints in 2-D, the patch rim in
    3-D), and slip is prescribed with ordinary essential conditions on those
    two names. A slip datum must taper to zero at the tips/rim, which stay
    single points shared by both sides.

    The source mesh is untouched, and the operation is re-applicable: when the
    fault moves, re-cut the base mesh at the new position and split again —
    the same non-cumulative pattern as :meth:`Mesh.adapt`.

    The split mesh carries NO geometric-multigrid tail. The coarse levels do
    not carry the fault, so an essential condition on it would constrain zero
    coarse DOFs and leave the custom-P coarse operator singular (see the
    warning on :meth:`Mesh.add_conforming_surface`); solvers on this mesh get
    their algebraic-multigrid defaults instead.

    Parameters
    ----------
    mesh : Mesh
        A mesh carrying the surface as a labelled facet chain — normally the
        child returned by :meth:`Mesh.add_conforming_surface`.
    name : str
        The surface's boundary name.
    verbose : bool, optional
        Report what was duplicated.

    Returns
    -------
    Mesh
        The split mesh, with ``mesh`` recorded as its parent.
    """
    from underworld3.discretisation import Mesh

    boundaries = _boundaries_with_sides(mesh, name)
    plus_name, minus_name = f"{name}Plus", f"{name}Minus"
    source_dm = mesh.dm
    if source_dm.getDimension() == 3 and uw.mpi.size > 1:
        if getattr(mesh, "_fault_point_pairs", {}):
            # a prior fault's pairing holds point ids of the CURRENT
            # distribution; redistribution renumbers every point, so
            # carrying it through needs pairing migration along the
            # migration SF — not built yet. Refuse loudly.
            raise NotImplementedError(
                "fault_split: 3-D multi-fault networks in parallel need "
                "pairing migration through the redistribution — split "
                "the network in serial, or one fault per mesh.")
        # the 3-D split requires the patch's cell star rank-interior,
        # and balance cuts are attracted to the refined patch region —
        # gather the (thin) star onto one rank first
        source_dm = _redistribute_fault_interior(
            source_dm, name, int(mesh.boundaries[name].value),
            verbose=verbose)
    pStart, _pEnd = source_dm.getChart()
    splitter = (split_along_label if source_dm.getDimension() == 2
                else split_along_label_3d)
    new_dm, point_map, clone_map = splitter(
        source_dm, name, int(mesh.boundaries[name].value),
        plus_name, int(boundaries[plus_name].value),
        minus_name, int(boundaries[minus_name].value),
        orientation=orientation, verbose=verbose)

    child = Mesh(
        new_dm,
        simplex=mesh.dm.isSimplex(),
        coordinate_system_type=mesh.CoordinateSystem.coordinate_type,
        qdegree=mesh.qdegree,
        boundaries=boundaries,
        verbose=False,
    )
    child.parent = mesh
    child._relationship_kind = "refinement"
    # Replica vertices are coincident with their originals, so a coincidence-
    # based injection would read one side of the jump for both — the transfer
    # must stay geometric.
    child._refine_dofs_coincide = False
    # geometry objects ride along so add_fault_bc can source the
    # constraint frame from them (normal="trace" / normal="surface")
    if getattr(mesh, "_fault_traces", None):
        child._fault_traces = dict(mesh._fault_traces)
    if getattr(mesh, "_fault_surfaces", None):
        child._fault_surfaces = dict(mesh._fault_surfaces)
    child.regions = mesh.regions
    child._parent_mesh_version = mesh._mesh_version
    # The Minus->Plus point pairing in the SPLIT mesh's numbering, for every
    # replica whose original SURVIVED — the duplicated vertices and the
    # doubled fault facets (the P2 midpoint DOFs live on the latter). Fresh
    # minus-side SPOKE facets are excluded: their originals were dropped
    # (point_map -1), because a re-homed spoke is the same geometric edge
    # renumbered, not a coincident copy. This dict is the ONLY source of the
    # pairing — the sides are geometrically coincident, so no coordinate
    # query can ever recover it — and it is what a fault interface condition
    # (slip constraint, friction) pairs degrees of freedom with.
    #
    # Splitting renumbers the whole chart, so PRIOR faults' pairings are
    # carried through point_map rather than copied verbatim — copied ids
    # would silently index the wrong points on the new mesh. A prior fault's
    # points always survive this split (only this fault's minus spokes are
    # re-homed), asserted rather than assumed.
    child._fault_point_pairs = {}
    for prior, pairs in getattr(mesh, "_fault_point_pairs", {}).items():
        remapped = {int(point_map[qm - pStart]): int(point_map[qp - pStart])
                    for qm, qp in pairs.items()}
        if any(q < 0 for kv in remapped.items() for q in kv):
            raise RuntimeError(
                f"fault_split internal: splitting {name!r} dropped a point "
                f"of the prior fault {prior!r}'s pairing — the faults are "
                "not disjoint.")
        child._fault_point_pairs[prior] = remapped
    child._fault_point_pairs[name] = {
        int(q_minus): int(point_map[old_pt - pStart])
        for q_minus, old_pt in clone_map.items()
        if point_map[old_pt - pStart] >= 0}
    mesh._registered_children.add(child)
    return child
