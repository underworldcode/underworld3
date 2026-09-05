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
to the low-level splitters on a seam-touching fault is refused. The 2-D
split can instead run THROUGH the seam (``across_seams=True``, the path of
a band meshed through it by ``place_thin_volume(seams="conform")``): the
chain is assembled globally from the star-forest identities of its
vertices, a vertex on the seam is duplicated on every rank holding it with
the replica owned where the original is, and nothing is redistributed.

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
    _install_point_sf, _rebuild_point_sf, _shared_points,
    _write_coordinates)


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


def _fan_sides(dm, v, before, after, verts, cS, cE):
    """The cells around interior chain vertex ``v``, as
    ``(problem, plus_cells, minus_cells)``.

    The incident cells are walked as a fan, using the directed link edge
    each anticlockwise cell triple contributes — :func:`reconnect._link_ring`
    opened at the two fault facets. Walking from the outgoing fault
    neighbour ``after`` to the incoming one ``before`` sweeps the cells on
    the LEFT of the directed chain, which is the Plus side. Any structural
    anomaly in the fan (a duplicated directed link edge, a walk that escapes
    or fails to close) means the cell orientations are not consistent there,
    and is a refusal rather than a guess: unlike a declined flip, a mis-sided
    cell would silently weld the fault shut at one node. The whole fan must
    be local — both neighbours are vertices of this rank.
    """
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
    return None, plus_arc, minus_arc


def _sector_sides(dm, v, x_before, x_after, X, vS, verts, cS, cE):
    """The LOCAL cells around a SHARED interior chain vertex, by geometry.

    A rank holds only part of the fan at a vertex on the partition seam,
    so the fan cannot be walked; but the two fault facets at ``v`` are
    cell edges, so every cell lies wholly inside one of the two angular
    sectors they bound. A cell whose centroid direction lies in the
    anticlockwise sweep from the outgoing neighbour to the incoming one is
    on the LEFT of the directed chain — the Plus side, the same rule
    :func:`_fan_sides` walks. Either side may be empty here: a seam that
    touches the chain at ``v`` without crossing it leaves one rank with
    cells on one side only. The neighbours' positions come from the
    GLOBAL chain (both ranks hold them, whichever rank holds the facets).
    """
    x_v = X[v - vS]
    d_after, d_before = x_after - x_v, x_before - x_v
    a0 = np.arctan2(d_after[1], d_after[0])
    span = (np.arctan2(d_before[1], d_before[0]) - a0) % (2.0 * np.pi)
    if span <= 0.0:
        return (ValueError,
                "fault_split: the chain doubles back on itself at a seam "
                "vertex (the two facets there are collinear and "
                "coincident)."), None, None
    fan = [int(p) for p in dm.getTransitiveClosure(v, useCone=False)[0]
           if cS <= int(p) < cE]
    plus, minus = [], []
    for c in fan:
        cen = X[np.asarray(verts[c - cS], dtype=np.int64) - vS].mean(axis=0)
        d = cen - x_v
        ang = (np.arctan2(d[1], d[0]) - a0) % (2.0 * np.pi)
        (plus if ang < span else minus).append(c)
    return None, plus, minus


def _take_sides(dm, chain, verts, cS, cE):
    """Classify every fault-fan cell as Plus (+1) or Minus (-1).

    At each interior chain vertex the fan is walked (:func:`_fan_sides`).
    Returns ``(problem, side_of_cell, substitutions)`` where
    ``substitutions`` maps a Minus cell to the set of chain vertices it
    must replace with replicas.
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
        problem, plus_arc, minus_arc = _fan_sides(dm, v, before, after,
                                                  verts, cS, cE)
        if problem is not None:
            return problem, None, None
        # A cell can meet the fault at two chain vertices and be walked
        # onto opposite sides only when the chain kinks through it (a chord
        # cell); _assign refuses rather than resolves.
        problem = _assign(plus_arc, 1, v) or _assign(minus_arc, -1, v)
        if problem is not None:
            return problem, None, None
    return None, side_of_cell, substitutions


# ------------------------------------------------ the split across a seam

def _point_keys(dm):
    """The global identity of every local point as ``(owner rank, owner
    index)``: a leaf reads it off the point star-forest, a root is its
    own. Chart-indexed arrays ``(owner, index)``."""
    pStart, pEnd = dm.getChart()
    owner = np.full(pEnd - pStart, uw.mpi.rank, dtype=np.int64)
    index = np.arange(pStart, pEnd, dtype=np.int64)
    try:
        _nroots, ilocal, iremote = dm.getPointSF().getGraph()
    except (ValueError, TypeError):
        return owner, index
    if ilocal is not None and len(ilocal):
        leaves = np.asarray(ilocal, dtype=np.int64)
        remote = np.asarray(iremote).reshape(-1, 2)
        owner[leaves - pStart] = remote[:, 0]
        index[leaves - pStart] = remote[:, 1]
    return owner, index


def _global_fault_chains(dm, fault_edges, X, vS, pStart, shared,
                         orientation=None):
    """The fault as ordered vertex paths over EVERY rank. COLLECTIVE.

    Each rank contributes its fault facets as pairs of global vertex keys
    (:func:`_point_keys`) with their coordinates; the union is one graph
    every rank holds identically, so the walk order — and with it the
    Plus/Minus sides — is a function of the geometry alone, decided the
    same way on every rank. Several components are allowed (a fault the
    placement left uncut somewhere is several chains); each must be an
    open simple path of two or more facets, as in :func:`_fault_chain`.

    A fault facet that is itself shared is refused: its two cells lie on
    different ranks, so the seam runs ALONG the fault there and the two
    sides would have to be told apart across the seam (the along-strike
    case the design note declines). A shared chain VERTEX is what this
    function exists for.

    Returns ``(problem, chains, coord, key_of)``: the chains as lists of
    keys, every chain vertex's position, and the key -> local vertex map
    for the vertices this rank holds. ``problem`` is the same on every
    rank (the verdicts are exchanged).
    """
    eS, eE = dm.getDepthStratum(1)
    vS_, vE = dm.getDepthStratum(0)
    owner, index = _point_keys(dm)

    def key(v):
        return (int(owner[v - pStart]), int(index[v - pStart]))

    problem = None
    local_edges = []
    nbr = {}
    for e in fault_edges:
        a, b = (int(p) for p in dm.getCone(e))
        if shared[e - pStart]:
            problem = (RuntimeError,
                       "fault_split: a fault facet lies ON the partition "
                       "seam (the seam runs along the fault there). The "
                       "split across a seam needs the seam transversal to "
                       "the fault — every fault facet on one rank.")
            break
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
        local_edges.append((key(a), key(b), tuple(map(float, X[a - vS])),
                            tuple(map(float, X[b - vS]))))
    if problem is None:
        # The fault must be strictly interior — the rule of _fault_chain:
        # a one-sided UNSHARED facet in a fault vertex's star is the domain
        # boundary (a seam facet is one-sided locally but shared).
        for v in nbr:
            star = dm.getTransitiveClosure(v, useCone=False)[0]
            for p in star:
                p = int(p)
                if eS <= p < eE and len(dm.getSupport(p)) != 2 \
                        and not shared[p - pStart]:
                    problem = (ValueError,
                               "fault_split: the fault touches the domain "
                               "boundary. Only strictly interior faults, "
                               "with both tips inside the mesh, are "
                               "supported in this version.")
                    break
            if problem is not None:
                break

    every = uw.mpi.comm.allgather((problem, local_edges))
    problems = [p for p, _e in every if p is not None]
    if problems:
        seam = [p for p in problems if p[0] is RuntimeError]
        return (seam[0] if seam else problems[0]), None, None, None

    gnbr, coord = {}, {}
    for _p, edges in every:
        for ka, kb, xa, xb in edges:
            gnbr.setdefault(ka, []).append(kb)
            gnbr.setdefault(kb, []).append(ka)
            coord[ka] = np.asarray(xa, dtype=float)
            coord[kb] = np.asarray(xb, dtype=float)
    if not gnbr:
        return (ValueError, "fault_split: no fault facets on any rank."), \
            None, None, None
    if any(len(ns) > 2 for ns in gnbr.values()):
        return (ValueError,
                "fault_split: the label meets itself at a junction. "
                "Junctions need a side assignment this version does not "
                "define; split each branch under its own label instead."), \
            None, None, None

    # components, each walked from the tip lower in coordinate order
    seen, chains = set(), []
    for start_key in sorted(gnbr):
        if start_key in seen:
            continue
        comp, stack = set(), [start_key]
        while stack:
            k = stack.pop()
            if k in comp:
                continue
            comp.add(k)
            stack.extend(gnbr[k])
        seen |= comp
        tips = [k for k in comp if len(gnbr[k]) == 1]
        n_edges = sum(len(gnbr[k]) for k in comp) // 2
        if n_edges < 2:
            return (ValueError,
                    "fault_split: a chain has a single facet, so no "
                    "interior vertex exists to split. A one-facet fault "
                    "cannot slip."), None, None, None
        if len(tips) != 2:
            return (ValueError,
                    "fault_split: the labelled facets form a closed loop, "
                    "not an open chain."), None, None, None
        start = min(tips, key=lambda t: (float(coord[t][0]),
                                         float(coord[t][1]), t))
        chain, prev, cur = [start], None, start
        while True:
            onward = [w for w in gnbr[cur] if w != prev]
            if not onward:
                break
            prev, cur = cur, onward[0]
            chain.append(cur)
            if len(chain) > len(comp):
                return (ValueError,
                        "fault_split: the chain walk did not terminate; "
                        "the labelled facets are not a simple path."), \
                    None, None, None
        if orientation is not None and len(chain) >= 2:
            d = np.asarray(orientation, dtype=float).ravel()[:2]
            if float((coord[chain[-1]] - coord[chain[0]]) @ d) < 0.0:
                chain.reverse()
        chains.append(chain)

    # the chain vertices this rank holds: the ends of its own facets, and
    # every shared vertex (a seam may touch the chain at a vertex where
    # this rank holds cells but no facet)
    key_of = {}
    for v in range(vS_, vE):
        if shared[v - pStart]:
            key_of[key(v)] = v
    for v in nbr:
        key_of[key(v)] = v
    key_of = {k: v for k, v in key_of.items() if k in gnbr}
    return None, chains, coord, key_of


def _take_sides_across_seams(dm, chains, coord, key_of, shared, verts, X,
                             vS, cS, cE, pStart):
    """Sides and replicas for the global chains, on this rank.

    Every interior chain vertex this rank holds gets a replica (whether or
    not this rank holds a fault facet at it, so the star-forest entries
    for the replica exist on every rank that holds the original). The
    fan at an unshared vertex is walked; at a shared one the local cells
    are sided by the sector rule. Returns ``(problem, side, substitutions,
    interior)`` with ``side`` keyed ``(chain index, cell)`` and
    ``interior`` the local vertices to duplicate.
    """
    side, substitutions, interior = {}, {}, []
    for k, chain in enumerate(chains):
        for i in range(1, len(chain) - 1):
            v = key_of.get(chain[i])
            if v is None:
                continue
            interior.append(v)
            if shared[v - pStart]:
                problem, plus, minus = _sector_sides(
                    dm, v, coord[chain[i - 1]], coord[chain[i + 1]], X, vS,
                    verts, cS, cE)
            else:
                before, after = (key_of.get(chain[i - 1]),
                                 key_of.get(chain[i + 1]))
                if before is None or after is None:
                    return (RuntimeError,
                            "fault_split internal: an unshared chain "
                            "vertex has a neighbour this rank does not "
                            "hold."), None, None, None
                problem, plus, minus = _fan_sides(dm, v, before, after,
                                                  verts, cS, cE)
            if problem is not None:
                return problem, None, None, None
            for cells, sgn in ((plus, 1), (minus, -1)):
                for c in cells:
                    if side.setdefault((k, c), sgn) != sgn:
                        return (ValueError,
                                "fault_split: a cell meets the fault from "
                                "both sides (the chain kinks through it). "
                                "Refine the mesh near the kink and "
                                "re-cut."), None, None, None
                    if sgn < 0:
                        substitutions.setdefault(c, set()).add(v)
    return None, side, substitutions, interior


def _seam_normals(chains, coord, key_of, shared, pStart):
    """The measure-weighted Plus->Minus normal at every SHARED interior
    chain vertex this rank holds, from the global chain — the sum over
    both adjacent facets that a rank holding one of them cannot form
    locally. Same convention as the contact solve's accumulator (the
    facet normal away from the Plus cell, weighted by the facet length):
    with tangent t along the walk the Plus side is the LEFT, so the
    Plus->Minus normal is ``(t_y, -t_x)``."""
    out = {}
    for chain in chains:
        for i in range(1, len(chain) - 1):
            v = key_of.get(chain[i])
            if v is None or not shared[v - pStart]:
                continue
            acc = np.zeros(2)
            for a, b in ((chain[i - 1], chain[i]), (chain[i], chain[i + 1])):
                t = coord[b] - coord[a]
                acc += np.array([t[1], -t[0]])      # |e| * (t_y, -t_x)/|e|
            out[v] = acc
    return out


def _rebuild_point_sf_split(new, dm, point_map, nroots, replica_of_old,
                            fresh_of_old):
    """The point star-forest of a split that crossed the seam. COLLECTIVE.

    Three kinds of leaf: a surviving shared point, renumbered (the
    one-broadcast trick of :func:`reconnect._rebuild_point_sf`); the
    REPLICA of a shared chain vertex, a new point on every rank holding
    the original, owned where the original is (so a coincident pair never
    straddles ranks — the contact solve's pair block needs both points in
    one rank's diagonal portion); and a shared seam edge re-homed onto a
    replica, a fresh edge on both sides. The owner's new index for each
    arrives by broadcasting three root arrays over the OLD star-forest.
    A replica or a re-homing made on one side of the seam only is a
    contract violation, raised on every rank together.
    """
    pStart, pEnd = dm.getChart()
    sf = dm.getPointSF()
    try:
        _nroots, ilocal, iremote = sf.getGraph()
    except (ValueError, TypeError):
        return                            # unpopulated: nothing is shared
    n = pEnd - pStart
    roots = [np.ascontiguousarray(point_map, dtype=np.int32),
             np.full(n, -1, dtype=np.int32), np.full(n, -1, dtype=np.int32)]
    for p, q in replica_of_old.items():
        roots[1][p - pStart] = q
    for p, q in fresh_of_old.items():
        roots[2][p - pStart] = q
    leaves_data = []
    for root in roots:
        leaf = np.full(n, -1, dtype=np.int32)
        # COLLECTIVE: a rank sharing nothing still participates.
        sf.bcastBegin(MPI.INT32_T, root, leaf, MPI.REPLACE)
        sf.bcastEnd(MPI.INT32_T, root, leaf, MPI.REPLACE)
        leaves_data.append(leaf)
    leaf_new, leaf_rep, leaf_fresh = leaves_data

    local, remote, bad = [], [], None
    if ilocal is not None and len(ilocal):
        leaves = np.asarray(ilocal, dtype=np.int64)
        owners = np.asarray(iremote).reshape(-1, 2)[:, 0]
        for p, o in zip(leaves, owners):
            i = int(p) - pStart
            p = int(p)
            if point_map[i] >= 0:
                if leaf_new[i] < 0:
                    bad = "a shared point survived here but not on its owner"
                    break
                local.append(int(point_map[i]))
                remote.append((int(o), int(leaf_new[i])))
            else:
                f = fresh_of_old.get(p)
                if f is None or leaf_fresh[i] < 0:
                    bad = "a shared facet was re-homed on one side only"
                    break
                local.append(int(f))
                remote.append((int(o), int(leaf_fresh[i])))
            r = replica_of_old.get(p)
            if r is not None:
                if leaf_rep[i] < 0:
                    bad = "a shared chain vertex was duplicated here but not on its owner"
                    break
                local.append(int(r))
                remote.append((int(o), int(leaf_rep[i])))
            elif leaf_rep[i] >= 0:
                bad = "a shared chain vertex was duplicated on its owner but not here"
                break
    verdicts = dm.comm.tompi4py().allgather(bad)
    offenders = [(rank, why) for rank, why in enumerate(verdicts) if why]
    if offenders:
        raise RuntimeError(
            "fault_split internal: the split across the seam is not "
            f"consistent between ranks: {offenders}.")
    new_sf = PETSc.SF().create(comm=dm.comm)
    if not local:
        new_sf.setGraph(nroots, np.zeros(0, dtype=PETSc.IntType),
                        np.zeros(0, dtype=PETSc.IntType))
        _install_point_sf(new, new_sf)
        return
    local = np.asarray(local, dtype=PETSc.IntType)
    remote = np.asarray(remote, dtype=PETSc.IntType).reshape(-1, 2)
    order = np.argsort(local, kind="stable")
    new_sf.setGraph(nroots, local[order], remote[order].reshape(-1))
    _install_point_sf(new, new_sf)


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
                      verbose=False, across_seams=False, info=None):
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
    across_seams : bool, optional
        Let the chain cross the partition seam. The chain is assembled
        globally (:func:`_global_fault_chains`), a shared chain vertex is
        duplicated on every rank holding it with the replica owned where
        the original is, and the seam edges re-homed onto a replica keep
        their star-forest entries (:func:`_rebuild_point_sf_split`). The
        seam must be transversal: a fault facet ON the seam is refused.
        The label may then be several chains (each an open path of two
        or more facets), split together. Default ``False``: the chain
        must be rank-interior, as before.
    info : dict or None, optional
        With ``across_seams``, receives ``"seam_normals"`` — the
        Plus->Minus facet-measure normal at every shared interior chain
        vertex this rank holds (new numbering), the sum over both facets
        that no single rank can accumulate.

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

    fault_pair_edge = {}
    for e in fault_edges:
        a, b = (int(p) for p in dm.getCone(e))
        fault_pair_edge[(a, b) if a < b else (b, a)] = e

    across = bool(across_seams) and uw.mpi.size > 1
    problem, chain, interior = None, None, []
    side_of_cell, substitutions = {}, {}
    seam_normals = {}
    if across:
        # The chain is global: every rank assembles the same paths from
        # the union of the facets, so a vertex on the seam is sided by
        # the same two neighbours on both ranks. The verdict is exchanged
        # inside; a problem here is already the same on every rank.
        problem, chains, coord, key_of = _global_fault_chains(
            dm, fault_edges, X, vS, pStart, shared, orientation=orientation)
        chain_of_key = {}
        if problem is None:
            for k, ch in enumerate(chains):
                for kk in ch:
                    chain_of_key[kk] = k
            problem, side_by_chain, substitutions, interior = \
                _take_sides_across_seams(dm, chains, coord, key_of, shared,
                                         verts, X, vS, cS, cE, pStart)
        if problem is None:
            seam_normals = _seam_normals(chains, coord, key_of, shared,
                                         pStart)
            local_key = {v: k for k, v in key_of.items()}
            for (a, b), e in fault_pair_edge.items():
                k = chain_of_key[local_key[a]]
                sides = sorted(side_by_chain.get((k, int(c)), 0)
                               for c in dm.getSupport(e))
                if sides != [-1, 1]:
                    problem = (ValueError,
                               "fault_split: the two cells of a fault "
                               "facet were not classified onto opposite "
                               "sides — the side assignment is "
                               "inconsistent along the chain.")
                    break
    else:
        if fault_edges:
            problem, chain = _fault_chain(dm, fault_edges, X, vS, shared,
                                          pStart, orientation=orientation)
        if problem is None and fault_edges:
            problem, side_of_cell, substitutions = _take_sides(
                dm, chain, verts, cS, cE)
        if problem is None and fault_edges:
            # Every fault facet must have been classified onto opposite
            # sides by the fans of its endpoints; anything else is a
            # mis-wiring that would otherwise only surface as a wrong
            # stress field.
            for e in fault_pair_edge.values():
                sides = sorted(side_of_cell.get(int(c), 0)
                               for c in dm.getSupport(e))
                if sides != [-1, 1]:
                    problem = (ValueError,
                               "fault_split: the two cells of a fault "
                               "facet were not classified onto opposite "
                               "sides — the side assignment is "
                               "inconsistent along the chain.")
                    break
        interior = chain[1:-1] if chain else []

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

    if across:
        # Shared chain vertices ARE duplicated, on every rank holding
        # them, and a shared seam edge on the Minus side is re-homed on
        # both: the star-forest gains the replica pairs and the fresh
        # edges, owner unchanged.
        replica_of_old = {v: replica_new[t] for v, t in replicas_of.items()}
        fresh_of_old = {e_old: e_new for e_new, e_old in clone_map.items()
                        if eS <= e_old < eE and point_map[e_old - pStart] < 0}
        _rebuild_point_sf_split(new, dm, point_map, at - pStart,
                                replica_of_old, fresh_of_old)
        if info is not None:
            info["seam_normals"] = {
                int(point_map[v - pStart]): n for v, n in seam_normals.items()}
    elif uw.mpi.size > 1:
        # No shared point is ever duplicated or dropped (a shared chain
        # vertex is refused above; split_fault redistributes so the whole
        # chain is rank-interior), so the leaf set carries over by
        # renumbering alone — the same argument as the 3-D splitter.
        _rebuild_point_sf(new, dm, point_map, at - pStart)

    if verbose and (fault_edges or interior):
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


def _blind_clipped_ends(dm, label, value, poly, tol, blind, shared=None):
    """Unlabel ``blind`` edges from every chain end that is not a trace end.

    Rank-local by construction (the chains are). A chain left with fewer
    than two edges is unlabelled whole. Returns the number unlabelled.
    With ``shared`` (chart-indexed flags), a chain end ON the seam is a
    crossing the split will carry through, not a clipped end: left alone.
    """
    from underworld3.utilities.line_cut import _coords

    eS, eE = dm.getDepthStratum(1)
    vS, _vE = dm.getDepthStratum(0)
    X = _coords(dm)
    if label.getStratumSize(value) == 0:
        return 0
    edges = [int(e) for e in label.getStratumIS(value).getIndices()
             if eS <= int(e) < eE]
    ends = poly[[0, -1]]
    removed = 0
    for chain in _facet_components(dm, edges):
        nbr = {}
        for e in chain:
            a, b = (int(q) for q in dm.getCone(e))
            nbr.setdefault(a, []).append(e)
            nbr.setdefault(b, []).append(e)
        pStart = dm.getChart()[0]
        tips = [v for v, es in nbr.items() if len(es) == 1
                and not (shared is not None and shared[v - pStart])]
        clipped = [v for v in tips
                   if np.linalg.norm(ends - X[v - vS], axis=1).min() > tol]
        cut = set()
        for v in clipped:
            e = nbr[v][0]
            for _k in range(blind):
                cut.add(e)
                a, b = (int(q) for q in dm.getCone(e))
                v = b if a == v else a
                onward = [f for f in nbr[v] if f != e]
                if not onward:
                    break
                e = onward[0]
        # a CLIPPED chain left with fewer than two edges is unlabelled
        # whole; an unclipped one-edge piece is legitimate — the stretch
        # of a spine between two seam vertices on one rank, which the
        # split across the seam carries as part of the global chain
        if cut and len(chain) - len(cut) < 2:
            cut = set(chain)
        for e in cut:
            label.clearValue(e, value)
        removed += len(cut)
    return removed


def _label_embedded_edges(dm, segments, values, exclude=None, verbose=False,
                          blind=1, across_seams=False):
    """Label the mesh edges already lying on each polyline; return a CLONE.

    The label-only form of the cut (``add_fault(cut=False)``): an edge is
    the fault's when both its ends and its midpoint lie on the line (the
    chord rule of :func:`line_cut._label_cut_edges`). Edges with a support
    cell in the ``exclude`` label are left out — the seam ligament's base
    cells, where the band was clipped and the fault is deliberately uncut.
    A chain end that is not an end of the trace is a CLIPPED end, at the
    seam; ``blind`` edges are unlabelled from it so the cut stops short and
    the band encloses the tip — blind, as a fault under a free surface.
    COLLECTIVE (the tolerance is a global length). The input is untouched.
    """
    from underworld3.utilities.line_cut import (_global_extent,
                                                _label_cut_edges)

    new = dm.clone()
    scale = _global_extent(new)
    # COLLECTIVE, before any per-segment work: the seam crossings the
    # blind rule must leave alone
    shared = _shared_points(new) if (across_seams and uw.mpi.size > 1) \
        else None
    cS, cE = new.getHeightStratum(0)
    excluded = np.zeros(new.getChart()[1] - new.getChart()[0], dtype=bool)
    if exclude is not None and new.hasLabel(exclude):
        lbl = new.getLabel(exclude)
        pStart = new.getChart()[0]
        for v in lbl.getValueIS().getIndices() if lbl.getNumValues() else []:
            if lbl.getStratumSize(int(v)) == 0:
                continue
            for c in lbl.getStratumIS(int(v)).getIndices():
                if cS <= int(c) < cE:
                    excluded[int(c) - pStart] = True
    for name, poly in segments:
        marked = _label_cut_edges(new, [np.asarray(poly, dtype=float)],
                                  1e-9 * scale, name, values[name])
        lbl = new.getLabel(name)
        n_dropped = 0
        if excluded.any():
            pStart = new.getChart()[0]
            for e in marked:
                if any(excluded[int(c) - pStart]
                       for c in new.getSupport(int(e))):
                    lbl.clearValue(int(e), int(values[name]))
                    n_dropped += 1
        if blind > 0:
            n_dropped += _blind_clipped_ends(
                new, lbl, int(values[name]), np.asarray(poly, dtype=float),
                1e-9 * scale, blind, shared=shared)
        n_local = len(marked) - n_dropped
        n_total = int(uw.mpi.comm.allreduce(n_local, op=MPI.SUM))
        if n_total == 0:
            # legitimate under the seam ligament: a short fault that lies
            # wholly in the clipped stretch is not cut at all — the weak
            # plane carries it — so the split below is empty, not wrong
            uw.pprint(f"[add_fault {name!r}] no embedded edge lies on the "
                      "trace: the fault is entirely in the seam ligament "
                      "and stays uncut (weak plane only)")
        elif verbose:
            uw.pprint(f"[add_fault {name!r}] {n_total} embedded facets "
                      f"labelled (no cut)")
    return new


def add_fault(mesh, faults, verbose=False, cut=True, exclude=None,
              blind=1, across_seams=False):
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
    cut : bool, optional
        ``False`` cuts nothing: the fault's facets are the mesh edges
        ALREADY lying on each polyline — the embedded spines of a placed
        band (:func:`~underworld3.utilities.place_surface.place_thin_volume`
        with ``mesher="network"``). Nothing moves, no vertex is pulled,
        and nothing is redistributed: with the seam-ligament placement
        every spine edge is rank-interior by construction, and a fault
        that crosses a rank in and out is split as several sub-chains,
        each with its own uncut tips at the ligaments (#670). A labelled
        edge touching the seam is refused.
    exclude : str or None, optional
        With ``cut=False``, a cell label whose cells contribute no fault
        edge (the placement's ``<label>_ligament`` cells) — so a base edge
        that happens to lie on the line inside a ligament is left alone.
    blind : int, optional
        With ``cut=False``, the edges left uncut at each CLIPPED chain end
        (an end that is not an end of the trace): the cut stops short of
        the seam and the band encloses the tip, blind, as a fault under a
        free surface. Default one band cell.
    across_seams : bool, optional
        With ``cut=False`` on a band meshed THROUGH the seams
        (``seams="conform"``): the embedded spines cross the seam and are
        split through it — a chain end on the seam is a crossing, not a
        clipped end, so it is neither blinded nor pinned; nothing is
        redistributed. See :func:`split_fault`.

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
    if cut:
        for name, poly in segments:
            # EVERY control point gets a vertex, not just the tips: an
            # interior kink is the same problem as a tip — a distinguished
            # point of the geometry that must coincide with a mesh vertex,
            # or the cut leaves the chain fragmented at the turn.
            dm = pull_vertex_onto(dm, np.asarray(poly, dtype=float))
            dm, info = cut_along_lines(dm, [poly], label=name,
                                       label_value=values[name])
            if verbose:
                uw.pprint(f"[add_fault {name!r}] {info['n_cut_edges']} "
                          f"facets, min angle {info['min_angle']:.2f} deg")
    else:
        dm = _label_embedded_edges(dm, segments, values, exclude=exclude,
                                   verbose=verbose, blind=blind,
                                   across_seams=across_seams)

    # In parallel the balanced partition's cuts are attracted to the refined
    # fault band, so a cut chain generally touches the seam. Redistribute
    # ONCE, keyed on EVERY fault's facets together, BEFORE any split: each
    # fault's cell star (plus one growth layer) moves to the rank that
    # already owns most of the union, the chains become rank-interior, and
    # the splits below run with serial topology. Doing it here rather than
    # per split also keeps a network's prior pairings valid — a pairing
    # cannot yet migrate through a redistribution, so split_fault refuses
    # exactly the per-split case this pre-pass avoids. Label-only faults
    # never touch the seam (the placement clipped the band there); one
    # that does is a placement defect and is refused, collectively.
    if uw.mpi.size > 1 and dm.getDimension() == 2 and not across_seams:
        labels = [(name, values[name]) for name, _poly in segments]
        if _fault_labels_touch_seam(dm, labels):
            if not cut:
                raise RuntimeError(
                    "add_fault(cut=False): a fault edge touches the "
                    "partition seam. The embedded spines are rank-interior "
                    "only when the band was placed with seams='ligament'.")
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
                          verbose=verbose, across_seams=across_seams)
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


def split_faults(mesh, names, verbose=False, groups=None,
                 across_seams=False):
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

    ``across_seams`` splits THROUGH the seams instead (2-D, the band
    meshed through them): nothing is redistributed, see
    :func:`split_fault`.
    """
    import underworld3 as uw
    from underworld3.discretisation import Mesh

    out = mesh
    if uw.mpi.size > 1 and not across_seams:
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
        out = split_fault(out, n, verbose=verbose, across_seams=across_seams)
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


def _unsplit_facets(dm, name, value, plus_name, minus_name):
    """The facets labelled ``name`` that carry neither side label yet."""
    if not (dm.hasLabel(name) and dm.getLabel(name).getStratumSize(int(value)) > 0):
        return []
    fS, fE = dm.getHeightStratum(1)
    out = []
    for f in dm.getLabel(name).getStratumIS(int(value)).getIndices():
        f = int(f)
        if not (fS <= f < fE):
            continue
        if (dm.hasLabel(plus_name) and dm.getLabelValue(plus_name, f) >= 0) \
                or (dm.hasLabel(minus_name)
                    and dm.getLabelValue(minus_name, f) >= 0):
            continue
        out.append(f)
    return out


def _facet_components(dm, facets):
    """Rank-local connected components of a facet set, joined through a
    shared cone point (a vertex in 2-D, an edge in 3-D). Each component
    is a sorted list; the list is ordered by its smallest facet."""
    facets = [int(f) for f in facets]
    parent = {f: f for f in facets}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    first = {}
    for f in facets:
        for q in dm.getCone(f):
            other = first.setdefault(int(q), f)
            if other != f:
                ra, rb = find(other), find(f)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for f in facets:
        groups.setdefault(find(f), []).append(f)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def split_fault(mesh, name, orientation=None, verbose=False,
                across_seams=False):
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
    across_seams : bool, optional
        Split THROUGH the partition seam instead of redistributing: the
        chain is assembled globally and a vertex on the seam is
        duplicated on every rank holding it (2-D; the band meshed through
        the seam by ``place_thin_volume(seams="conform")``). Nothing
        moves. The coincident pair at a seam vertex is owned by one rank,
        as the contact solve requires; its Plus->Minus normal is recorded
        on the child (``_fault_seam_normals``) for the accumulator that
        cannot form it from one rank's facets.

    Returns
    -------
    Mesh
        The split mesh, with ``mesh`` recorded as its parent.
    """
    from underworld3.discretisation import Mesh

    boundaries = _boundaries_with_sides(mesh, name)
    plus_name, minus_name = f"{name}Plus", f"{name}Minus"
    source_dm = mesh.dm
    across = bool(across_seams) and uw.mpi.size > 1
    if across and source_dm.getDimension() != 2:
        raise NotImplementedError(
            "fault_split: the split across a partition seam is 2-D; the "
            "3-D thin volume still gathers (design note, item B).")
    if uw.mpi.size > 1 and not across:
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
    splitter = (split_along_label if source_dm.getDimension() == 2
                else split_along_label_3d)
    value = int(mesh.boundaries[name].value)
    plus_value = int(boundaries[plus_name].value)
    minus_value = int(boundaries[minus_name].value)
    comm = uw.mpi.comm

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
    #
    # A fault may be several SUB-CHAINS on a rank — the seam-ligament
    # placement leaves it uncut where it crosses the seam, so a rank the
    # fault crosses in and out holds two pieces (#670). Each pass splits
    # one piece per rank under a temporary label, collectively, until no
    # rank has a piece left; a single whole chain takes the plain path.
    prior = {p: dict(v)
             for p, v in getattr(mesh, "_fault_point_pairs", {}).items()}
    prior_normals = {p: dict(v) for p, v in
                     getattr(mesh, "_fault_seam_normals", {}).items()}
    pairs = {}
    seam_normals = {}
    dm_cur = source_dm
    tmp = f"_{name}_split_pass"
    n_pass = 0
    while True:
        remaining = _unsplit_facets(dm_cur, name, value, plus_name, minus_name)
        if across:
            # ONE collective pass over the global chain(s): every piece
            # on every rank at once, so a vertex on the seam is sided and
            # duplicated by both ranks in the same pass.
            if n_pass > 0 or comm.allreduce(len(remaining), op=MPI.SUM) == 0:
                break
            lname, lvalue = name, value
            comps = []
        else:
            comps = [c for c in _facet_components(dm_cur, remaining)
                     if len(c) >= 2]
        if not across:
            whole = (len(comps) <= 1
                     and sum(len(c) for c in comps) == len(remaining))
            n_max, all_whole = comm.allreduce(len(comps), op=MPI.MAX), \
                comm.allreduce(whole, op=MPI.LAND)
            n_left = comm.allreduce(len(remaining), op=MPI.SUM)
            if n_max == 0 and (n_pass > 0 or n_left == 0):
                break
        if across:
            pass
        elif n_pass == 0 and (all_whole or n_max == 0):
            lname, lvalue = name, value        # today's path, verbatim
        else:
            if dm_cur.hasLabel(tmp):
                dm_cur.removeLabel(tmp)
            dm_cur.createLabel(tmp)
            lbl = dm_cur.getLabel(tmp)
            for f in (comps[0] if comps else []):
                lbl.setValue(int(f), 1)
            lname, lvalue = tmp, 1
        pStart, _pEnd = dm_cur.getChart()
        pass_info = {}
        extra = {"across_seams": True, "info": pass_info} if across else {}
        new_dm, point_map, clone_map = splitter(
            dm_cur, lname, lvalue, plus_name, plus_value, minus_name,
            minus_value, orientation=orientation, verbose=verbose, **extra)

        def carry(d, what):
            out = {int(point_map[qm - pStart]): int(point_map[qp - pStart])
                   for qm, qp in d.items()}
            if any(q < 0 for kv in out.items() for q in kv):
                raise RuntimeError(
                    f"fault_split internal: splitting {name!r} dropped a "
                    f"point of {what}'s pairing — the faults are not "
                    "disjoint.")
            return out

        def carry_keys(d, what):
            out = {int(point_map[q - pStart]): n for q, n in d.items()}
            if any(q < 0 for q in out):
                raise RuntimeError(
                    f"fault_split internal: splitting {name!r} dropped a "
                    f"seam vertex of {what}.")
            return out

        for p in prior:
            prior[p] = carry(prior[p], f"the prior fault {p!r}")
        for p in prior_normals:
            prior_normals[p] = carry_keys(prior_normals[p],
                                          f"the prior fault {p!r}")
        pairs = carry(pairs, f"{name!r}'s earlier piece")
        seam_normals = carry_keys(seam_normals, f"{name!r}'s earlier piece")
        seam_normals.update(pass_info.get("seam_normals", {}))
        pairs.update({
            int(q_minus): int(point_map[old_pt - pStart])
            for q_minus, old_pt in clone_map.items()
            if point_map[old_pt - pStart] >= 0})
        if new_dm.hasLabel(tmp):
            new_dm.removeLabel(tmp)
        dm_cur = new_dm
        n_pass += 1
        if lname == name:
            break
    if verbose and n_pass > 1:
        uw.pprint(f"[split_fault {name!r}] {n_pass} sub-chain passes")
    if n_pass == 0:
        # nothing split anywhere (the fault lies wholly in a seam
        # ligament): the side boundaries still exist, empty, so that a
        # condition on them is a no-op rather than a null-label query
        dm_cur = dm_cur.clone()
        for lname in (plus_name, minus_name):
            if not dm_cur.hasLabel(lname):
                dm_cur.createLabel(lname)

    child = Mesh(
        dm_cur,
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
    child._fault_point_pairs = dict(prior)
    child._fault_point_pairs[name] = pairs
    if prior_normals or seam_normals:
        child._fault_seam_normals = dict(prior_normals)
        child._fault_seam_normals[name] = seam_normals
    mesh._registered_children.add(child)
    return child
