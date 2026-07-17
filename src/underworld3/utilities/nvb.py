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

__all__ = ["NVBMesh", "TaggedBisectionMesh"]


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
        # TODO(#360): coordinate row i is assumed to be vertex point vS+i — the
        # assumption class outlawed for Mesh coordinate arrays by #360. It holds
        # HERE because this serial cell-list engine only ever sees fresh
        # createFromCellList / undistributed base DMs whose degree-1 coordinate
        # section is vertex-ordered (checked below), and the np>1 path uses the
        # native transform instead (NotImplementedError before reaching this).
        # If NVBMesh ever ingests distributed/permuted DMs, switch to a
        # section-offset mapping (cf. Mesh._coord_rows_for_points).
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
        # TODO(#360): row i == vertex point vS+i is assumed, as in from_dm above.
        # Safe here: the DM was created two lines up by createFromCellList on this
        # rank (serial-only path), so its coordinate section is vertex-ordered.
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


class TaggedBisectionMesh:
    r"""Dimension-general tagged-simplex bisection: Maubach's rule with the
    Diening--Gehring--Storn (DGS) coloring initialization (arXiv:2306.02674).

    Unlike the slot-based 2D :class:`NVBMesh` above, this engine carries the
    refinement state as the **vertex ordering itself** plus a tag, which is
    what makes it work in any dimension. A cell is ``(verts, tag)`` with
    ``verts`` an ORDERED ``(dim+1)``-tuple and tag :math:`\gamma \in
    \{1..dim\}`; its bisection edge is ``verts[0]--verts[tag]``. Splitting at
    the edge midpoint ``m`` produces children (both tagged :math:`\gamma-1`,
    or ``dim`` when :math:`\gamma=1`)::

        T1 = verts with v_gamma replaced by m
        T2 = verts with v_0 dropped and m inserted at position gamma

    The DGS initialization (a greedy vertex coloring of the base mesh in a
    deterministic coordinate-lexicographic order; each cell's vertices sorted
    by ascending color, the globally-largest color first, tag = dim) makes
    this provably terminate on ANY conforming simplicial base mesh — with the
    sharp Binev--Dahmen--DeVore closure bound, at most
    :math:`dim! \cdot dim \cdot 2^{dim-2}` similarity classes per base cell
    (4 in 2D, 36 in 3D), and shape regularity. No pre-refinement, no
    compatibility search. Validated in
    ``docs/developer/design/nvb_prototype_3d.py`` (capstone stage 1a).

    The conforming closure generalises the 2D neighbour rule to the edge
    STAR: an edge is bisected only when it is the bisection edge of *every*
    cell containing it; incompatible star members are recursively bisected
    first (DGS Algorithm 3).

    Boundary labels are carried per FACET (edges in 2D, triangular faces in
    3D), keyed by vertex set, and split with the facet when one of its edges
    is bisected — the only way an interior/fault interface survives
    refinement, since geometry cannot distinguish which boundary a facet
    belongs to. Same DM interface as :class:`NVBMesh`
    (``from_dm`` / ``refine`` / ``to_dm``); serial (this is the np=1
    reference engine — the parallel path is the native transform).

    This is the unified engine of the 2026-07 adaptivity capstone: the 3D
    path uses it now; the 2D dispatch migrates onto it (and ``NVBMesh`` is
    deleted) when the native transform adopts the tagged rule (stage 1e).
    """

    def __init__(self, coords, cells, dim):
        self.dim = int(dim)
        self.coords = [np.asarray(p, float) for p in coords]
        self.cells = {}                 # cid -> (ordered verts tuple, tag)
        self.depth = {}                 # cid -> bisection depth (generation)
        self.region = {}                # cid -> region value (or None)
        self.base_of = {}               # cid -> base-cell ancestor cid
        self.edge2cells = {}            # edge key -> set(cid)
        self.edge2mid = {}              # edge key -> midpoint vertex id
        self.facet_label = {}           # sorted facet vertex tuple -> value
        self._edge2lf = {}              # edge key -> set(labelled facet keys)
        self._next = 0
        self._init_coloring_and_tags(cells)

    # ---- DGS initialization ------------------------------------------------
    def _init_coloring_and_tags(self, cells):
        """Greedy vertex coloring in coordinate-lexicographic order (the
        deterministic, partition-independent choice), then the DGS
        Definition-2 ordering + tag = dim per cell."""
        nv = len(self.coords)
        adj = [set() for _ in range(nv)]
        for t in cells:
            for a in t:
                for b in t:
                    if a != b:
                        adj[a].add(b)
        order = sorted(range(nv),
                       key=lambda i: tuple(np.round(self.coords[i], 12)))
        color = [-1] * nv
        for v in order:
            used = {color[w] for w in adj[v] if color[w] >= 0}
            c = 0
            while c in used:
                c += 1
            color[v] = c
        N = max(color) if color else 0
        self.vertex_color = color
        for t in cells:
            vs = sorted(t, key=lambda v: color[v])   # ascending color
            if color[vs[-1]] == N:                   # global max color FIRST
                vs = [vs[-1]] + vs[:-1]
            self._add_cell(tuple(vs), self.dim, depth=0, region=None,
                           base=None)

    # ---- construction from a DMPlex ----------------------------------------
    @classmethod
    def from_dm(cls, dm, boundaries=(), regions=()):
        """Build from a serial simplex ``DMPlex`` (2D or 3D).

        ``boundaries`` / ``regions`` are ``(name, value)`` iterables. Labelled
        facets (height-1 stratum: edges in 2D, faces in 3D) seed
        :attr:`facet_label` keyed by vertex set; region labels seed
        :attr:`region` per base cell and propagate parent-to-child.
        """
        dim = dm.getDimension()
        if dim not in (2, 3):
            raise NotImplementedError(
                f"TaggedBisectionMesh supports 2D/3D simplex meshes, got dim={dim}.")
        vS, vE = dm.getDepthStratum(0)
        fS, fE = dm.getHeightStratum(1)
        cS, cE = dm.getHeightStratum(0)
        # TODO(#360): coordinate row i is assumed to be vertex point vS+i, as
        # in NVBMesh.from_dm above — valid for the fresh/undistributed serial
        # DMs this engine ingests (checked below); distributed DMs go to the
        # native transform instead.
        coords = dm.getCoordinatesLocal().array.reshape(-1, dm.getCoordinateDim())
        if coords.shape[0] != vE - vS:
            raise RuntimeError(
                f"TaggedBisectionMesh.from_dm: {coords.shape[0]} coords vs "
                f"{vE - vS} vertices (degree-1 coordinate DM expected).")

        cells, cell_pts = [], []
        for c in range(cS, cE):
            clos = dm.getTransitiveClosure(c)[0]
            verts = [p - vS for p in clos if vS <= p < vE]
            if len(verts) != dim + 1:
                raise RuntimeError(
                    f"cell {c} has {len(verts)} vertices (not a {dim}-simplex)")
            cells.append(tuple(verts))
            cell_pts.append(c)

        self = cls(coords, cells, dim)

        for name, value in boundaries:
            if not dm.hasLabel(name):
                continue
            iset = dm.getStratumIS(name, value)
            if iset is None:
                continue
            for p in iset.getIndices():
                if fS <= p < fE:                      # a labelled facet
                    clos = dm.getTransitiveClosure(p)[0]
                    fverts = tuple(sorted(
                        int(q - vS) for q in clos if vS <= q < vE))
                    self._set_facet_label(fverts, int(value))

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

    # ---- cell / facet bookkeeping -------------------------------------------
    def _cell_edges(self, verts):
        return [_fs(verts[i], verts[j])
                for i in range(self.dim + 1)
                for j in range(i + 1, self.dim + 1)]

    def _add_cell(self, verts, tag, depth, region, base):
        cid = self._next
        self._next += 1
        self.cells[cid] = (verts, tag)
        self.depth[cid] = depth
        self.region[cid] = region
        self.base_of[cid] = cid if base is None else base
        for e in self._cell_edges(verts):
            self.edge2cells.setdefault(e, set()).add(cid)
        return cid

    def _del_cell(self, cid):
        verts, _ = self.cells.pop(cid)
        self.depth.pop(cid, None)
        self.region.pop(cid, None)
        for e in self._cell_edges(verts):
            s = self.edge2cells.get(e)
            if s is not None:
                s.discard(cid)
                if not s:
                    del self.edge2cells[e]

    def _set_facet_label(self, fkey, value):
        self.facet_label[fkey] = value
        for i in range(len(fkey)):
            for j in range(i + 1, len(fkey)):
                self._edge2lf.setdefault(_fs(fkey[i], fkey[j]),
                                         set()).add(fkey)

    def _drop_facet_label(self, fkey):
        self.facet_label.pop(fkey, None)
        for i in range(len(fkey)):
            for j in range(i + 1, len(fkey)):
                s = self._edge2lf.get(_fs(fkey[i], fkey[j]))
                if s is not None:
                    s.discard(fkey)

    def _midpoint(self, ekey):
        """Midpoint vertex of edge ``ekey`` (created once). Every labelled
        facet containing the edge splits with it: F -> (F \\ {b}) + {m} and
        (F \\ {a}) + {m} — midpoints are only created on an edge being
        bisected, so this is exactly where a boundary facet subdivides."""
        if ekey not in self.edge2mid:
            a, b = ekey
            self.coords.append(0.5 * (self.coords[a] + self.coords[b]))
            m = len(self.coords) - 1
            self.edge2mid[ekey] = m
            for fkey in list(self._edge2lf.get(ekey, ())):
                val = self.facet_label.get(fkey)
                if val is None:
                    continue
                self._drop_facet_label(fkey)
                f1 = tuple(sorted([v for v in fkey if v != b] + [m]))
                f2 = tuple(sorted([v for v in fkey if v != a] + [m]))
                self._set_facet_label(f1, val)
                self._set_facet_label(f2, val)
        return self.edge2mid[ekey]

    def bse(self, cid):
        """Bisection edge of cell ``cid``: ``verts[0]--verts[tag]``."""
        verts, tag = self.cells[cid]
        return verts[0], verts[tag]

    # ---- the Maubach split ---------------------------------------------------
    def _split(self, cid, m):
        verts, g = self.cells[cid]
        gp = g - 1 if g >= 2 else self.dim
        d = self.depth[cid] + 1
        r = self.region[cid]
        base = self.base_of[cid]
        v = list(verts)
        t1 = tuple(v[:g] + [m] + v[g + 1:])          # m replaces v_gamma
        t2 = tuple(v[1:g + 1] + [m] + v[g + 1:])     # v_0 dropped
        self._del_cell(cid)
        self._add_cell(t1, gp, d, r, base)
        self._add_cell(t2, gp, d, r, base)

    # ---- bisection with recursive conforming closure ---------------------------
    def bisect(self, cid, _depth=0):
        """Bisect ``cid`` at its bisection edge with the conforming star
        closure: the edge splits only when it is the bisection edge of every
        cell in its star; incompatible star members are recursively bisected
        first. If ``cid`` is itself consumed by a recursive step, its
        bisection has already happened and the call returns."""
        if _depth > 100000:
            raise RuntimeError("tagged-bisection closure did not terminate")
        ekey = _fs(*self.bse(cid))
        while True:
            if cid not in self.cells:
                return
            star = self.edge2cells.get(ekey, set())
            blocked = [c for c in star if _fs(*self.bse(c)) != ekey]
            if not blocked:
                break
            self.bisect(blocked[0], _depth + 1)
        m = self._midpoint(ekey)
        for c in list(self.edge2cells.get(ekey, ())):
            self._split(c, m)

    def refine(self, marked):
        """Refine each still-live cell id in ``marked`` (a set/iterable)."""
        for cid in list(marked):
            if cid in self.cells:
                self.bisect(cid)

    # ---- export / diagnostics ---------------------------------------------
    def arrays(self):
        """``(coords (V,dim), cells (N,dim+1), regions (N,), cids (N,))`` for
        live cells in a stable order; ``regions`` is ``-1`` where unset."""
        coords = np.array(self.coords)
        cids = list(self.cells)
        cells = np.array([self.cells[c][0] for c in cids], dtype=np.int64)
        regions = np.array(
            [self.region[c] if self.region.get(c) is not None else -1
             for c in cids], dtype=np.int64)
        return coords, cells, regions, np.array(cids, dtype=np.int64)

    def centroids_h(self):
        """``(centroids, h, cids)`` for live cells, with
        ``h = (dim! * vol)**(1/dim)`` the simplex characteristic size used by
        the metric marking (matches the SBR/NVB paths in ``_adapt_nested``)."""
        coords, cells, _, cids = self.arrays()
        cen = coords[cells].mean(axis=1)
        e = coords[cells[:, 1:]] - coords[cells[:, :1]]
        h = np.abs(np.linalg.det(e)) ** (1.0 / self.dim)
        return cen, h, cids

    def check_conforming(self):
        """``(hanging, overshared)``: live cells owning a bisected edge, and
        facets shared by more than two cells. Both must be 0."""
        hanging = 0
        for verts, _ in self.cells.values():
            for e in self._cell_edges(verts):
                if e in self.edge2mid:
                    hanging += 1
        facet_count = {}
        for verts, _ in self.cells.values():
            vs = list(verts)
            for k in range(len(vs)):
                f = tuple(sorted(vs[:k] + vs[k + 1:]))
                facet_count[f] = facet_count.get(f, 0) + 1
        overshared = sum(1 for n in facet_count.values() if n > 2)
        return hanging, overshared

    def similarity_classes(self, ndigits=6):
        """Maximum number of distinct cell *shapes* (scale-normalised sorted
        edge-length tuples) among the descendants of any single base cell.
        Bounded by ``dim! * dim * 2**(dim-2)`` (4 in 2D, 36 in 3D); a blow-up
        signals a broken child rule. Diagnostic only."""
        coords = np.array(self.coords)
        per_base = {}
        for cid, (verts, _) in self.cells.items():
            P = coords[list(verts)]
            ls = np.sort([np.linalg.norm(P[i] - P[j])
                          for i in range(len(verts))
                          for j in range(i + 1, len(verts))])
            key = tuple(np.round(ls / ls[-1], ndigits))
            per_base.setdefault(self.base_of[cid], set()).add(key)
        return max(len(s) for s in per_base.values())

    # ---- DMPlex build + label transfer ------------------------------------
    def to_dm(self, boundaries=(), regions=(), comm=None):
        """Build a fresh interpolated ``DMPlex`` from the current cell list
        and transfer boundary / region labels onto it.

        Labelled facets are matched by **vertex set** (vertices identified by
        coordinate — midpoints are exact float averages computed identically
        here and in the DM, so the match is exact). Each labelled facet also
        labels its full closure (edges and vertices), matching the gmsh/UW
        convention so ``Mesh()`` derives ``UW_Boundaries`` from them;
        ``All_Boundaries`` is the geometric outer boundary.
        """
        from scipy.spatial import cKDTree

        coords, cells, region_of, _ = self.arrays()
        # createFromCellList needs positively-oriented cells; the Maubach
        # ordering is refinement state, not orientation, so reorient only the
        # EXPORTED copy (swap the last two vertices where det is negative).
        e = coords[cells[:, 1:]] - coords[cells[:, :1]]
        neg = np.linalg.det(e) < 0
        cells[neg, -2], cells[neg, -1] = (cells[neg, -1],
                                          cells[neg, -2].copy())

        dm = PETSc.DMPlex().createFromCellList(
            self.dim, cells.astype(np.int32), coords, interpolate=True,
            comm=comm or PETSc.COMM_WORLD)

        vS, vE = dm.getDepthStratum(0)
        fS, fE = dm.getHeightStratum(1)
        cS, cE = dm.getHeightStratum(0)

        # DM vertex point -> engine vertex id by coordinate (exact match).
        # TODO(#360): row i == vertex point vS+i — safe: the DM was created
        # two lines up by createFromCellList on this rank (serial path).
        dm_vcoords = dm.getCoordinatesLocal().array.reshape(
            -1, dm.getCoordinateDim())
        _, nearest = cKDTree(np.array(self.coords)).query(dm_vcoords)
        eng_of_dmvert = {vS + i: int(nearest[i]) for i in range(vE - vS)}

        dm.markBoundaryFaces("All_Boundaries", 1001)

        wanted = {value for _, value in boundaries}
        if wanted:
            for name, value in boundaries:
                dm.createLabel(name)
            labels = {value: dm.getLabel(name) for name, value in boundaries}
            for p in range(fS, fE):
                clos = dm.getTransitiveClosure(p)[0]
                fverts = tuple(sorted(
                    eng_of_dmvert[q] for q in clos if vS <= q < vE))
                lab = self.facet_label.get(fverts)
                if lab in wanted:
                    for q in clos:            # facet + its edges + vertices
                        labels[lab].setValue(q, lab)

        if regions:
            for name, value in regions:
                dm.createLabel(name)
            rlabels = {value: dm.getLabel(name) for name, value in regions}
            for i in range(cE - cS):
                rv = int(region_of[i])
                if rv in rlabels:
                    rlabels[rv].setValue(cS + i, rv)

        return dm
