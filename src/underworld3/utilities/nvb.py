"""Newest-vertex bisection (NVB) — a *graded* simplicial refinement engine.

PETSc's ``refine_sbr`` (longest-edge bisection) has an **unbounded** conforming
closure for region marking: a single marked cell drains the longest-edge
propagation path clear to the nearest mesh-size contrast, so it can only ever
produce a *uniform-finest patch*, never a graded mesh (a level+1 ring around a
level+2 sub-ring around a finer core). Newest-vertex bisection has a *provably
bounded* closure (Binev–Dahmen–DeVore 2004; Stevenson 2008) given a compatible
initial edge labelling, and yields conforming, 2:1-balanced, shape-regular graded
meshes with a finite number of similarity classes — exactly the staircase we
want, with the simplices UW3 is built on (no p4est/DMForest quad/hex).

This module is the **serial Route-A** implementation (design note
``docs/developer/design/NVB_GRADED_ADAPT.md``): the triangulation is maintained as
pure-Python/numpy arrays with a per-cell refinement-edge labelling; after a
refinement pass the PETSc ``DMPlex`` is rebuilt from the cell list
(``createFromCellList``) and the boundary / region labels are transferred onto the
children. It plugs in where ``custom_mg.sbr_refine`` does today and feeds the same
coordinate-based custom-P geometric-MG hierarchy.

Parallel (np>1) is **out of scope** for this implementation: ``createFromCellList``
builds a fresh serial DM whose point star-forest does not preserve the parent's
decomposition (rank *r* owning the refinements of rank *r*'s base cells), which the
custom-P parallel path requires. That needs a native ``DMPlexTransform`` (Route B);
see the design note. ``mesh.adapt(engine="nvb")`` raises ``NotImplementedError`` at
np>1 rather than silently corrupting the decomposition.

Data model (newest-vertex convention)
--------------------------------------
``cell = (peak, b0, b1)`` — ``peak`` is the newest vertex; the **refinement edge**
is ``{b0, b1}`` (opposite the peak). Bisecting splits ``{b0, b1}`` at its midpoint
``m`` and makes ``m`` the peak of both children: ``(m, peak, b0)`` and
``(m, peak, b1)`` — so each child's refinement edge is a parent edge incident to
the apex. Cycling the refinement edge this way (rather than re-picking the
geometric longest edge every step, as LEB does) is what bounds the recursion to a
finite number of similarity classes.
"""

import numpy as np
from petsc4py import PETSc

__all__ = ["NVBMesh"]


def _fs(a, b):
    """A sorted (frozen) vertex pair — the canonical edge key."""
    return (a, b) if a < b else (b, a)


class NVBMesh:
    """A 2D newest-vertex-bisection triangulation with boundary-edge and
    region-cell labels carried across bisections.

    Construct from a DMPlex with :meth:`from_dm`, refine marked cells with
    :meth:`refine` (the bounded conforming closure runs automatically), and emit a
    fresh labelled DMPlex with :meth:`to_dm`. The instance is **stateful** — the
    refinement-edge labelling propagates parent→child, which is what preserves the
    similarity-class (shape-regularity) bound across successive refinement passes;
    re-seeding from scratch each level would not.

    2D only for now (the design generalises to tets — Bänsch/Maubach/Stevenson —
    but that is a follow-up).
    """

    def __init__(self, coords, tris):
        self.coords = [np.asarray(p, float) for p in coords]
        self.cells = {}                 # cid -> (peak, b0, b1)
        self.depth = {}                 # cid -> bisection depth from its base cell
        self.region = {}                # cid -> region value (or None)
        self.edge2cells = {}            # (a,b) -> set(cid)
        self.edge2mid = {}              # (a,b) -> midpoint vertex id
        self.edge_label = {}            # (a,b) -> boundary value (propagated)
        self._next = 0
        # Initial labelling: refinement edge = LONGEST edge (a robust seed that
        # always conforming-terminates in 2D; NVB's marked-edge propagation takes
        # over from here — this is NOT the same as running longest-edge bisection,
        # which re-picks the longest edge at every step).
        for t in tris:
            self._add_cell(self._longest_edge_order(t))

    # ---- construction from a DMPlex ---------------------------------------
    @classmethod
    def from_dm(cls, dm, boundaries=(), regions=()):
        """Build an :class:`NVBMesh` from a 2D simplex ``DMPlex``.

        ``boundaries`` / ``regions`` are iterables of ``(name, value)`` (e.g. from a
        mesh's boundary / region enum). Boundary-face (edge) labels seed
        :attr:`edge_label` keyed by the edge's vertex pair — this is the *only*
        thing that lets an interior/fault interface survive refinement, since
        geometry alone cannot distinguish which boundary an edge belongs to. Region
        labels seed :attr:`region` per base cell and propagate parent→child.
        """
        if dm.getDimension() != 2:
            raise NotImplementedError("NVBMesh is 2D only (tets are a follow-up).")
        vS, vE = dm.getDepthStratum(0)
        eS, eE = dm.getDepthStratum(1)
        cS, cE = dm.getHeightStratum(0)
        coords = dm.getCoordinatesLocal().array.reshape(-1, 2)
        if coords.shape[0] != vE - vS:
            raise RuntimeError(
                f"NVBMesh.from_dm: {coords.shape[0]} coords vs {vE - vS} vertices "
                f"(degree-1 coordinate DM expected).")

        tris, cell_pts = [], []
        for c in range(cS, cE):
            clos = dm.getTransitiveClosure(c)[0]
            verts = [p - vS for p in clos if vS <= p < vE]
            if len(verts) != 3:
                raise RuntimeError(f"cell {c} has {len(verts)} vertices (not a tri)")
            tris.append(tuple(verts))
            cell_pts.append(c)

        self = cls(coords, tris)

        for name, value in boundaries:
            if not dm.hasLabel(name):
                continue
            iset = dm.getStratumIS(name, value)
            if iset is None:
                continue
            for p in iset.getIndices():
                if eS <= p < eE:                          # a boundary edge
                    a, b = (int(v - vS) for v in dm.getCone(p))
                    self.edge_label[_fs(a, b)] = int(value)

        if regions:
            cell_of_pt = {cell_pts[i]: i for i in range(len(cell_pts))}
            for name, value in regions:
                if not dm.hasLabel(name):
                    continue
                iset = dm.getStratumIS(name, value)
                if iset is None:
                    continue
                for p in iset.getIndices():
                    if p in cell_of_pt:
                        self.region[cell_of_pt[p]] = int(value)
        return self

    # ---- vertex / cell bookkeeping ----------------------------------------
    def _longest_edge_order(self, t):
        a, b, c = t
        P = self.coords
        keys = (_fs(a, b), _fs(b, c), _fs(c, a))
        b0, b1 = max(keys, key=lambda k: np.linalg.norm(P[k[0]] - P[k[1]]))
        peak = ({a, b, c} - {b0, b1}).pop()
        return (peak, b0, b1)

    def _add_cell(self, pbr, depth=0, region=None):
        cid = self._next
        self._next += 1
        self.cells[cid] = pbr
        self.depth[cid] = depth
        self.region[cid] = region
        p, b0, b1 = pbr
        for e in (_fs(p, b0), _fs(b0, b1), _fs(b1, p)):
            self.edge2cells.setdefault(e, set()).add(cid)
        return cid

    def _del_cell(self, cid):
        p, b0, b1 = self.cells.pop(cid)
        self.depth.pop(cid, None)
        self.region.pop(cid, None)
        for e in (_fs(p, b0), _fs(b0, b1), _fs(b1, p)):
            self.edge2cells[e].discard(cid)
            if not self.edge2cells[e]:
                del self.edge2cells[e]

    def _midpoint(self, e):
        """Midpoint vertex of edge ``e`` (created once). Propagates any boundary
        label on ``e`` onto its two halves — midpoints are only ever created on an
        edge being bisected, so this is exactly where a boundary edge subdivides."""
        if e not in self.edge2mid:
            a, b = e
            self.coords.append(0.5 * (self.coords[a] + self.coords[b]))
            m = len(self.coords) - 1
            self.edge2mid[e] = m
            lab = self.edge_label.pop(e, None)
            if lab is not None:
                self.edge_label[_fs(a, m)] = lab
                self.edge_label[_fs(m, b)] = lab
        return self.edge2mid[e]

    def _ref_edge(self, cid):
        _, b0, b1 = self.cells[cid]
        return _fs(b0, b1)

    def _neighbour(self, cid, e):
        others = self.edge2cells.get(e, set()) - {cid}
        return next(iter(others)) if others else None

    # ---- the bisection -----------------------------------------------------
    def _split(self, cid, m):
        p, b0, b1 = self.cells[cid]
        d = self.depth[cid] + 1
        r = self.region[cid]
        self._del_cell(cid)
        self._add_cell((m, p, b0), d, r)        # newest vertex m, ref edge {p,b0}
        self._add_cell((m, p, b1), d, r)        # newest vertex m, ref edge {p,b1}

    def bisect(self, cid, _depth=0):
        """Bisect cell ``cid`` with the bounded conforming completion: if the
        refinement edge is not shared as the neighbour's refinement edge, bisect the
        neighbour first (recursively) until the edge is *compatible*, then split the
        pair at the common midpoint. Terminates (bounded closure) — unlike
        longest-edge bisection's drain-to-interface chain."""
        if _depth > 100000:
            raise RuntimeError("NVB closure did not terminate (LEPP-like loop)")
        e = self._ref_edge(cid)
        nb = self._neighbour(cid, e)
        while nb is not None and self._ref_edge(nb) != e:
            self.bisect(nb, _depth + 1)
            nb = self._neighbour(cid, e)        # a child of the old neighbour
        m = self._midpoint(e)
        self._split(cid, m)
        if nb is not None:
            self._split(nb, m)

    def refine(self, marked):
        """Refine each still-live cell id in ``marked`` (a set/iterable)."""
        for cid in list(marked):
            if cid in self.cells:
                self.bisect(cid)

    # ---- export / diagnostics ---------------------------------------------
    def arrays(self):
        """``(coords (V,2), tris (N,3), regions (N,), cids (N,))`` for live cells,
        in a stable cell order. ``regions`` is ``-1`` where unset."""
        coords = np.array(self.coords)
        cids = list(self.cells)
        tris = np.array([self.cells[c] for c in cids], dtype=np.int64)
        regions = np.array(
            [self.region[c] if self.region.get(c) is not None else -1 for c in cids],
            dtype=np.int64)
        return coords, tris, regions, np.array(cids, dtype=np.int64)

    def centroids_h(self):
        """``(centroids (N,2), h (N,), cids (N,))`` for live cells, where
        ``h = (2·area)**0.5`` is the simplex characteristic size
        (``h ≈ (dim!·vol)^(1/dim)`` with ``dim=2``) used by the metric marking."""
        coords, tris, _, cids = self.arrays()
        a, b, c = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
        cen = (a + b + c) / 3.0
        area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                            (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
        return cen, np.sqrt(2.0 * area), cids

    def check_conforming(self):
        """``(hanging, overshared)``: ``hanging`` = live cells carrying an edge that
        has been bisected (a hanging node); ``overshared`` = edges shared by >2
        cells. Both must be 0 for a conforming mesh."""
        hanging = 0
        for (p, b0, b1) in self.cells.values():
            for e in (_fs(p, b0), _fs(b0, b1), _fs(b1, p)):
                if e in self.edge2mid:
                    hanging += 1
        overshared = sum(1 for cs in self.edge2cells.values() if len(cs) > 2)
        return hanging, overshared

    def similarity_classes(self, ndigits=6):
        """Number of distinct triangle *shapes* (similarity classes) among live
        cells — sorted, scale-normalised edge-length triples, rounded. NVB bounds
        this by a small constant (≤4 per base triangle in 2D); a blow-up signals
        degenerating elements. Diagnostic only."""
        coords = np.array(self.coords)
        classes = set()
        for (p, b0, b1) in self.cells.values():
            P = coords[[p, b0, b1]]
            ls = np.sort([np.linalg.norm(P[0] - P[1]),
                          np.linalg.norm(P[1] - P[2]),
                          np.linalg.norm(P[2] - P[0])])
            ls = ls / ls[-1]
            classes.add(tuple(np.round(ls, ndigits)))
        return len(classes)

    # ---- DMPlex build + label transfer ------------------------------------
    def to_dm(self, boundaries=(), regions=(), comm=None):
        """Build a fresh interpolated ``DMPlex`` from the current triangulation and
        transfer boundary / region labels onto it.

        ``boundaries`` / ``regions`` are ``(name, value)`` iterables naming the
        labels to create. Boundary edges are matched to their carried label by
        **vertex pair** (vertices identified by coordinate, robust to PETSc's
        renumbering — never by array index); each labelled edge also labels its two
        vertices, matching UW's boundary-label convention so ``Mesh()`` derives
        ``UW_Boundaries`` / ``Null_Boundary`` from them. ``All_Boundaries`` is the
        geometric outer boundary (``markBoundaryFaces``), independent of our labels.
        """
        from scipy.spatial import cKDTree

        coords, tris, region_of, _ = self.arrays()
        # createFromCellList needs consistent (CCW) winding; the internal
        # (peak,b0,b1) ordering is geometry-agnostic, so reorient the EXPORTED cell
        # list only (swap the last two verts where the signed area is negative).
        a, b, c = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
        sa = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - \
             (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        neg = sa < 0
        tris[neg, 1], tris[neg, 2] = tris[neg, 2], tris[neg, 1].copy()

        dm = PETSc.DMPlex().createFromCellList(
            2, tris.astype(np.int32), coords, interpolate=True,
            comm=comm or PETSc.COMM_WORLD)

        vS, vE = dm.getDepthStratum(0)
        eS, eE = dm.getDepthStratum(1)
        cS, cE = dm.getHeightStratum(0)

        # DM vertex point -> NVB vertex id by coordinate (midpoints are exact float
        # averages, computed identically here and in the DM, so the match is exact).
        dm_vcoords = dm.getCoordinatesLocal().array.reshape(-1, 2)
        _, nearest = cKDTree(np.array(self.coords)).query(dm_vcoords)
        nvb_of_dmvert = {vS + i: int(nearest[i]) for i in range(vE - vS)}

        # outer geometric boundary (UW convention)
        dm.markBoundaryFaces("All_Boundaries", 1001)

        # named boundary labels: each carried edge + its two vertices
        wanted = {value for _, value in boundaries}
        if wanted:
            for name, value in boundaries:
                dm.createLabel(name)
            labels = {value: dm.getLabel(name) for name, value in boundaries}
            for p in range(eS, eE):
                va, vb = (nvb_of_dmvert[v] for v in dm.getCone(p))
                lab = self.edge_label.get(_fs(va, vb))
                if lab in wanted:
                    labels[lab].setValue(p, lab)
                    for v in dm.getCone(p):
                        labels[lab].setValue(v, lab)

        # region cell labels (cells are points [cS, cE) in input order)
        if regions:
            for name, value in regions:
                dm.createLabel(name)
            rlabels = {value: dm.getLabel(name) for name, value in regions}
            for i in range(cE - cS):
                rv = int(region_of[i])
                if rv in rlabels:
                    rlabels[rv].setValue(cS + i, rv)

        return dm
