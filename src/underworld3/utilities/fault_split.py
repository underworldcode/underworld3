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

Restrictions in this version, all refused loudly rather than mishandled:
2-D only; a single open chain (no junctions, no loops, no single-facet
chains); the fault must not touch the domain boundary; and in parallel the
fault's cell fans must not touch the partition seam — the replicas are then
rank-local and the star-forest carries over by renumbering alone, exactly as
the deletion pass argues for its own seam freeze.

An essential condition on the fault is NOT sound under the custom-P geometric
multigrid hierarchy (the coarse levels do not carry the fault — see the
warning on :meth:`Mesh.add_conforming_surface`), so the split mesh is built
standalone, with no coarse tail: the Stokes velocity block then takes its
algebraic-multigrid default.
"""

import numpy as np
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities.reconnect import (
    _TOPOLOGY_LABELS, _cell_vertices_and_seam, _coords, _copy_labels,
    _rebuild_point_sf, _shared_points, _write_coordinates)


def _fault_chain(dm, fault_edges, X, vS):
    """The fault as an ordered vertex path, tip to tip.

    Returns ``(problem, chain)`` where ``problem`` is ``None`` or an
    ``(ExceptionClass, message)`` pair and ``chain`` lists the fault vertices
    in walk order, starting from the tip lower in coordinate order so the walk
    — and with it the Plus/Minus sides — is a function of the geometry, not of
    label-stratum order.
    """
    eS, eE = dm.getDepthStratum(1)

    nbr = {}
    for e in fault_edges:
        a, b = (int(p) for p in dm.getCone(e))
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)

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
    # is a domain-boundary facet; one in the star of any fault vertex means the
    # slit would reach the boundary, where the tip-clamping argument fails.
    for v in nbr:
        star = dm.getTransitiveClosure(v, useCone=False)[0]
        for p in star:
            p = int(p)
            if eS <= p < eE and len(dm.getSupport(p)) != 2:
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

        for cells, side in ((plus_arc, 1), (minus_arc, -1)):
            for c in cells:
                if side_of_cell.setdefault(c, side) != side:
                    # A cell can meet the fault at two chain vertices and be
                    # walked onto opposite sides only when the chain kinks
                    # through it (a chord cell). No side assignment is right
                    # for such a cell, so it is refused, not resolved.
                    return (ValueError,
                            "fault_split: a cell meets the fault from both "
                            "sides (the chain kinks through it). Refine the "
                            "mesh near the kink and re-cut."), None, None
        for c in minus_arc:
            substitutions.setdefault(c, set()).add(v)
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
                      minus_name, minus_value, verbose=False):
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
            "fault_split handles 2-D meshes; in 3-D the fault is not yet a "
            "conforming facet chain to begin with.")

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

    problem, chain = (None, None)
    if fault_edges:
        problem, chain = _fault_chain(dm, fault_edges, X, vS)

    # COLLECTIVE, and reached on every rank whatever the local verdict so far:
    # the seam flags and the cell triples are needed for the rebuild anyway,
    # and computing them before any refusal keeps every rank in step.
    shared = _shared_points(dm)
    verts, frozen = _cell_vertices_and_seam(dm, X, shared)

    # The seam verdict is taken from the facet cones alone, independent of the
    # chain checks, and OUTRANKS them: a fault crossing the seam presents to
    # each rank as a broken chain, so the fragments are the symptom and the
    # seam is the diagnosis.
    if fault_edges:
        fault_vertices = {int(p) for e in fault_edges for p in dm.getCone(e)}
        for v in fault_vertices:
            star = dm.getTransitiveClosure(v, useCone=False)[0]
            if any(frozen[int(p) - cS] for p in star
                   if cS <= int(p) < cE):
                problem = (RuntimeError,
                           "fault_split: the fault's cell fans touch the "
                           "partition seam. This version refuses to split "
                           "across ranks — repartition, or place the fault "
                           "inside one rank's subdomain.")
                break

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
        _rebuild_point_sf(new, dm, point_map, at - pStart)

    if verbose and fault_edges:
        uw.pprint(f"[fault_split {name!r}] duplicated {len(interior)} "
                  f"vertices along {len(fault_edges)} facets; sides are "
                  f"{plus_name!r} / {minus_name!r}")
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
        dm = pull_vertex_onto(dm, np.vstack([poly[0], poly[-1]]))
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
    for name, _poly in segments:
        out = split_fault(out, name, verbose=verbose)
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


def split_fault(mesh, name, verbose=False):
    """Split the nodes along the conforming surface ``name``; return the mesh.

    The result is a new, standalone :class:`Mesh` on which every continuous FE
    space is discontinuous across the fault: the surface's facets are doubled
    into boundaries ``<name>Plus`` and ``<name>Minus`` (coincident, no shared
    DOFs except the two tips), and slip is prescribed with ordinary essential
    conditions on those two names. A slip datum must taper to zero at the
    tips, which stay single points shared by both sides.

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
    pStart, _pEnd = mesh.dm.getChart()
    new_dm, point_map, clone_map = split_along_label(
        mesh.dm, name, int(mesh.boundaries[name].value),
        plus_name, int(boundaries[plus_name].value),
        minus_name, int(boundaries[minus_name].value),
        verbose=verbose)

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
