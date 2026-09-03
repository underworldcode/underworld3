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
seam freeze. A fault that DOES touch the seam is not the user's problem:
:func:`split_fault` (and :func:`add_fault`) REDISTRIBUTE first — the fault's
cell star plus one growth layer moves to the rank that already owns most of
it (:func:`_redistribute_fault_interior`), the chain or patch becomes
rank-interior, and the split runs with serial topology. Only a direct call
to the low-level splitters on a seam-touching fault is refused.

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
from underworld3.utilities.dm_labels import label_stratum_indices
from underworld3.utilities.reconnect import (
    _TOPOLOGY_LABELS, _cell_vertices_and_seam, _coords, _copy_labels,
    _rebuild_point_sf, _shared_points, _write_coordinates)


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
    not have is a CHAIN VERTEX on the seam: an unshared vertex owns its
    whole cell fan (any cell touching it would make it shared), so every
    replica and every re-homed spoke stays rank-local; a shared chain
    vertex means the fault touches or crosses the seam, which the split
    cannot represent. :func:`split_fault` REDISTRIBUTES the fault's cell
    star onto one rank before splitting exactly so this never happens; a
    direct low-level call on a seam-touching fault is refused. The check
    comes FIRST — before the fragment/tip topology checks — because a
    rank holding only a fragment of a seam-straddling fault would
    otherwise report a misleading chain-shape symptom where the true
    verdict is the seam.
    """
    eS, eE = dm.getDepthStratum(1)

    nbr = {}
    for e in fault_edges:
        a, b = (int(p) for p in dm.getCone(e))
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)

    if any(shared[v - pStart] for v in nbr):
        return (RuntimeError,
                "fault_split: a fault vertex sits on the partition seam — "
                "the split needs the whole chain rank-interior. split_fault "
                "redistributes the fault's cell star onto one rank before "
                "splitting; a direct split_along_label call must present a "
                "rank-interior fault."), None
    if len(fault_edges) < 2:
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
                # distinguish the domain boundary from the slit of an
                # ALREADY-SPLIT fault: a chain terminating on one is a
                # touching junction, which the split cannot represent —
                # and would not want to: a single shared vertex forces
                # every arm's slip to zero there, STIFFER than the true
                # sector junction. The offset form brackets the truth
                # (see the true-branch teaching example).
                on_slit = any(
                    dm.getLabelValue(lname, p) >= 0
                    for lname in (dm.getLabelName(i) for i in
                                  range(dm.getNumLabels()))
                    if lname.endswith("Plus") or lname.endswith("Minus"))
                if on_slit:
                    return (ValueError,
                            "fault_split: the fault terminates on an "
                            "already-split fault's slit (a touching "
                            "junction). A shared point would clamp every "
                            "arm's slip to zero there — stiffer than a "
                            "true junction. Use the offset form "
                            "(uw.meshing.prepare_fault_network), which "
                            "brackets the true branch."), None
                return (ValueError,
                        "fault_split: the fault touches the domain boundary. "
                        "Only strictly interior faults, with both tips inside "
                        "the mesh, are supported in this version."), None

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

    # Optional caller-supplied chain direction (add_fault passes the
    # polyline's own direction): the walk is aligned against it so the Plus
    # side is the LEFT of the directed line the USER drew, not of the
    # coordinate-order tip rule.
    if orientation is not None and len(chain) >= 2:
        d = np.asarray(orientation, dtype=float).ravel()[:2]
        if float((X[chain[-1] - vS] - X[chain[0] - vS]) @ d) < 0.0:
            chain.reverse()
    return None, chain


def _take_sides(dm, chain, verts, cS, cE):
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
    """
    side_of_cell, substitutions = {}, {}

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
    ``UW_Boundaries`` is rebuilt from scratch when a
    ``Mesh`` is constructed on the result, so cloning it costs nothing, and
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
            # Empty-safe (#589): an absent stratum hands back a NULL IS
            # wrapper, never None — the old `is None` guard was dead and
            # getIndices() on the wrapper segfaults.
            for p in label_stratum_indices(source, val):
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
    # domain boundary; a shared chain vertex is refused — split_fault
    # redistributes so the check never fires on the managed path) and the
    # cell triples feed the rebuild.
    shared = _shared_points(dm)
    verts, _frozen = _cell_vertices_and_seam(dm, X, shared)

    problem, chain = (None, None)
    if fault_edges:
        problem, chain = _fault_chain(dm, fault_edges, X, vS, shared, pStart,
                                      orientation=orientation)

    side_of_cell, substitutions = {}, {}
    if problem is None and fault_edges:
        problem, side_of_cell, substitutions = _take_sides(
            dm, chain, verts, cS, cE)

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
    interior = chain[1:-1] if chain else []
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
    _write_coordinates(new, dm.getCoordinateDim(),
                       (offset["v"], offset["v"] + sizes["v"]),
                       _coords(dm)[source_rows])

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
        # No shared point is ever duplicated or dropped (a shared chain
        # vertex is refused above; split_fault redistributes so the whole
        # chain is rank-interior), so the leaf set carries over by
        # renumbering alone — the same argument as the 3-D splitter.
        _rebuild_point_sf(new, dm, point_map, at - pStart)

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
    3-D analogue of the single-facet chain); and ANY contact between the
    patch's cell star and the partition seam — :func:`split_fault`
    redistributes the star onto one rank so the managed path never sees
    the refusal, exactly as in 2-D.

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
        # cell star of the patch must be rank-interior. split_fault's
        # redistribution satisfies it on the managed path; a direct call
        # on a seam-touching patch is refused.
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
    _write_coordinates(new, dm.getCoordinateDim(), (vS2, vE2),
                       _coords(dm)[source_rows])

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
        # The seam rule above guarantees every shared point is far from
        # the patch, so the leaf set carries over by renumbering alone —
        # the same argument as the 2-D splitter.
        _rebuild_point_sf(new, dm, point_map, new.getChart()[1])

    if verbose and fault_faces:
        uw.pprint(f"[fault_split {name!r}] duplicated {k} vertices, "
                  f"{len(interior_edges)} edges and {len(fault_faces)} "
                  f"faces of the patch; sides are {plus_name!r} / "
                  f"{minus_name!r}")
    return new, point_map, clone_map


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
        dm, info = cut_along_lines(dm, [poly], label=name,
                                   label_value=values[name])
        if verbose:
            uw.pprint(f"[add_fault {name!r}] {info['n_cut_edges']} facets, "
                      f"min angle {info['min_angle']:.2f} deg")

    # In parallel the balanced partition's cuts are attracted to the refined
    # fault band, so a cut chain generally touches the seam. Redistribute
    # ONCE, keyed on EVERY fault's facets together, BEFORE any split: each
    # fault's cell star (plus one growth layer) moves to the rank that
    # already owns most of the union, the chains become rank-interior, and
    # the splits below run with serial topology. Doing it here rather than
    # per split also keeps a network's prior pairings valid — a pairing
    # cannot yet migrate through a redistribution, so split_fault refuses
    # exactly the per-split case this pre-pass avoids.
    if uw.mpi.size > 1 and dm.getDimension() == 2:
        labels = [(name, values[name]) for name, _poly in segments]
        if _fault_labels_touch_seam(dm, labels):
            dm = _redistribute_fault_interior(dm, labels, verbose=verbose)

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
    # (add_fault_bc(..., normal="trace")) and for rendering overlays;
    # Surface OBJECTS ride along too, so normal="surface" spells the
    # same thing in 2-D as it does for a 3-D FaultSurface
    out._fault_traces = {name: np.array(poly, dtype=float)
                         for name, poly in segments}
    surfaces = {entry.name: entry for entry in faults
                if hasattr(entry, "control_points")
                and hasattr(entry, "name")}
    if surfaces:
        out._fault_surfaces = surfaces
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


def _fault_labels_touch_seam(dm, labels):
    """Does any labelled fault facet reach the partition seam? COLLECTIVE.

    ``labels`` is a sequence of ``(name, value)`` pairs. The test matches
    the 2-D split's refusal exactly: a fault facet that is itself shared,
    or has a shared cone vertex, would put a chain vertex on the seam —
    the configuration :func:`_redistribute_fault_interior` exists to
    dissolve. Every rank returns the same verdict (allreduce), so the
    caller's redistribute-or-not decision is collective by construction.
    """
    if uw.mpi.size == 1:
        return False
    shared = _shared_points(dm)
    pStart, _pEnd = dm.getChart()
    fS, fE = dm.getHeightStratum(1)
    touch = False
    for name, value in labels:
        if touch:
            break
        if not (dm.hasLabel(name)
                and dm.getLabel(name).getStratumSize(int(value)) > 0):
            continue
        for f in dm.getLabel(name).getStratumIS(int(value)).getIndices():
            f = int(f)
            if not (fS <= f < fE):
                continue
            if shared[f - pStart] or any(
                    shared[int(q) - pStart]
                    for q in dm.getTransitiveClosure(f)[0]):
                touch = True
                break
    return bool(uw.mpi.comm.allreduce(touch, op=MPI.LOR))


def _redistribute_fault_interior(dm, labels, verbose=False, groups=None):
    """Redistribute a cut mesh so each fault's cell star is rank-interior.

    The default partition's balance cuts are ATTRACTED to the locally
    refined fault region, so at any np >= 2 a graded fault mesh
    essentially always violates the split's rank-interior requirement
    (in 2-D the chain, in 3-D the patch). This reassigns ONLY the
    fault-star cells (cells incident to any fault vertex — a layer
    about two cells thick) to the single rank that already owns most
    of them; every other cell stays where the load-balanced partition
    put it, and the move is applied with a shell partitioner. The star
    is thin, so the imbalance cost is bounded by its size — NOT by the
    refined band — and the split's seam rule then holds by
    construction. The coincident pair blocks of the contact solve are
    rank-local for the same reason. ``labels`` is a sequence of
    ``(name, value)`` pairs — a NETWORK is redistributed in one move,
    keyed on the union of its faults, which is what lets every split
    that follows run without migrating any prior pairing. Returns a
    NEW dm; the input is untouched.

    ``groups`` partitions ``labels`` into the faults that must share a
    rank (a junction-connected cluster); each group is one region of
    :func:`place_surface._gather_regions`, moved to its own rank — or not
    moved at all when its star is already interior to one — so two
    faults a domain apart are never gathered together (#670). Without
    ``groups`` the whole network is one region, as before.
    """
    from underworld3.utilities.place_surface import _gather_regions

    comm = dm.getComm().tompi4py()
    if comm.size == 1:
        return dm
    if groups is None:
        groups = [list(labels)]

    fS, fE = dm.getHeightStratum(1)
    vS, vE = dm.getDepthStratum(0)
    pStart, pEnd = dm.getChart()
    ids = np.zeros(pEnd - pStart, dtype=np.int32)
    for g, group in enumerate(groups, start=1):
        for name, value in group:
            if not (dm.hasLabel(name)
                    and dm.getLabel(name).getStratumSize(int(value)) > 0):
                continue
            for f in dm.getLabel(name).getStratumIS(int(value)).getIndices():
                if fS <= int(f) < fE:
                    for q in dm.getTransitiveClosure(int(f))[0]:
                        if vS <= int(q) < vE:
                            i = int(q) - pStart
                            ids[i] = max(ids[i], g)
    if comm.allreduce(int(ids.max()) if ids.size else 0, op=MPI.MAX) == 0:
        names = sorted(name for name, _value in labels)
        raise RuntimeError(
            f"fault_split: no facets labelled {names} found on any rank.")
    work, n_region, n_moved, owner, _canon = _gather_regions(dm, ids, layers=1)
    if work is dm:
        work = dm.clone()          # the contract: a new dm, untouched input
    if verbose:
        names = sorted(name for name, _value in labels)
        uw.pprint(f"[fault_split] {names}: {len(owner)} region(s) of "
                  f"{n_region} cells, {n_moved} moved (owners "
                  f"{sorted(owner.values())}); local cells now "
                  f"{work.getHeightStratum(0)[1]}")
    return work


def split_faults(mesh, names, verbose=False, groups=None):
    """Split a NETWORK of already-labelled faults, any dimension, any np.

    The parallel obstruction to sequential :func:`split_fault` calls is
    the redistribution each one performs: a prior fault's pairing holds
    point ids of the current distribution and cannot yet migrate. So the
    network redistributes ONCE, keyed on the union of every fault's
    facets (each star moves to the rank already owning most of it), and
    every split that follows runs with serial topology — the same
    pre-pass the 2-D ``add_fault`` performs for freshly cut chains, made
    available for faults that already carry labels (the 3-D embedded
    patches, a reloaded mesh).

    ``groups``, a list of lists of names, says which faults must share a
    rank (a junction-connected cluster); each group is redistributed to
    its own rank, and one whose star is already interior to a rank is
    not moved (#670). Faults not named in any group form one more.
    """
    import underworld3 as uw
    from underworld3.discretisation import Mesh

    out = mesh
    if uw.mpi.size > 1:
        labels = [(n, int(mesh.boundaries[n].value)) for n in names]
        label_groups = None
        if groups is not None:
            value = dict(labels)
            label_groups = [[(n, value[n]) for n in g] for g in groups]
            rest = [n for n in names if not any(n in g for g in groups)]
            if rest:
                label_groups.append([(n, value[n]) for n in rest])
        if _fault_labels_touch_seam(mesh.dm, labels):
            dm = _redistribute_fault_interior(mesh.dm, labels,
                                              verbose=verbose,
                                              groups=label_groups)
            out = Mesh(dm, simplex=mesh.dm.isSimplex(),
                       coordinate_system_type=(
                           mesh.CoordinateSystem.coordinate_type),
                       qdegree=mesh.qdegree, boundaries=mesh.boundaries,
                       verbose=False)
            out.parent = mesh
            out._relationship_kind = "refinement"
            out._refine_dofs_coincide = False
            out.regions = mesh.regions
            out._parent_mesh_version = mesh._mesh_version
            mesh._registered_children.add(out)
    for n in names:
        out = split_fault(out, n, verbose=verbose)
    # The network's child INHERITS a mesh-owned geometric-MG tail — the
    # same rule Mesh.add_fault applies: a cut re-represents the same
    # grid, so the parent's coarse levels serve unchanged with the cut
    # mesh as the finest level (#620/#629). Without this, every solver
    # on a network split silently fell back to GAMG. The FAC zone is
    # NOT inherited (a split fault needs no patch — the keying ruling);
    # callers that key a zone re-adopt on the child explicitly.
    own_tail = getattr(mesh, "_custom_mg_coarse_meshes", None)
    if (own_tail is not None
            and getattr(out, "_custom_mg_coarse_meshes", None) is None):
        out._custom_mg_coarse_meshes = list(own_tail)
        out._custom_mg_builder = getattr(mesh, "_custom_mg_builder",
                                         "barycentric")
        out._custom_mg_fac_zone = None
    return out


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
    if uw.mpi.size > 1:
        labels = [(name, int(mesh.boundaries[name].value))]
        # The split requires the fault's cell star rank-interior, and
        # balance cuts are attracted to the refined fault region — gather
        # the (thin) star onto one rank first. 3-D always redistributes
        # (the patch star essentially always straddles the seam on a
        # graded mesh); 2-D redistributes only when the chain actually
        # touches the seam, which is what lets add_fault's one-shot
        # union redistribution make every subsequent per-fault split a
        # no-move here. Both branches of the decision are collective.
        if _fault_labels_touch_seam(source_dm, labels):
            if getattr(mesh, "_fault_point_pairs", {}):
                # a prior fault's pairing holds point ids of the CURRENT
                # distribution; redistribution renumbers every point, so
                # carrying it through needs pairing migration along the
                # migration SF — not built yet. Refuse loudly (and
                # identically on every rank: the pairing dict and the
                # seam verdict agree across ranks by construction).
                raise NotImplementedError(
                    "fault_split: multi-fault networks in parallel need "
                    "pairing migration through the redistribution — pass "
                    "the whole network to ONE add_fault call (2-D), or "
                    "split the network in serial.")
            source_dm = _redistribute_fault_interior(
                source_dm, labels, verbose=verbose)
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
