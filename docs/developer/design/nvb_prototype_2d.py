"""Phase-1 prototype: serial newest-vertex bisection (NVB) on a 2D triangulation.

Goal: discover the complexity and PROVE that NVB gives a GRADED mesh (a level+1
ring, level+2 sub-ring dividing SOME of them, ...) where longest-edge SBR gave a
uniform-finest patch.

Data model (newest-vertex convention):
  cell = (peak, b0, b1)  -- 'peak' is the newest vertex; the REFINEMENT EDGE is
  {b0,b1} (opposite the peak). Bisecting splits {b0,b1} at midpoint m and makes m
  the peak of both children: (m, peak, b0) and (m, peak, b1).

Conforming completion (recursive): to bisect a cell whose refinement edge is not
shared as the refinement edge of its neighbour, first bisect the neighbour
(recursively) until the edge becomes compatible, then bisect the pair. This is
the BOUNDED closure (cf. longest-edge's unbounded drain-to-interface).
"""
import numpy as np


def fs(a, b):
    return (a, b) if a < b else (b, a)


class NVBMesh:
    def __init__(self, coords, tris):
        self.coords = [np.asarray(p, float) for p in coords]
        self.cells = {}                 # cid -> (peak, b0, b1)
        self.depth = {}                 # cid -> bisection depth from its base cell
        self.edge2cells = {}            # (a,b) sorted -> set(cid)
        self.edge2mid = {}              # (a,b) sorted -> midpoint vertex id
        self._next = 0
        # initial labelling: refinement edge = LONGEST edge (a seed; NVB's
        # marked-edge propagation takes over from here)
        for t in tris:
            self._add_cell(self._longest_edge_order(t))

    # ---- vertex / cell bookkeeping ----------------------------------------
    def _longest_edge_order(self, t):
        a, b, c = t
        P = self.coords
        e = {fs(a, b): (a, b, c), fs(b, c): (b, c, a), fs(c, a): (c, a, b)}
        lens = {k: np.linalg.norm(P[k[0]] - P[k[1]]) for k in e}
        kmax = max(lens, key=lens.get)
        # want refinement edge {b0,b1} opposite peak -> order (peak, b0, b1)
        b0, b1 = kmax
        peak = ({a, b, c} - {b0, b1}).pop()
        return (peak, b0, b1)

    def _add_cell(self, pbr, depth=0):
        cid = self._next
        self._next += 1
        self.cells[cid] = pbr
        self.depth[cid] = depth
        p, b0, b1 = pbr
        for e in (fs(p, b0), fs(b0, b1), fs(b1, p)):
            self.edge2cells.setdefault(e, set()).add(cid)
        return cid

    def _del_cell(self, cid):
        p, b0, b1 = self.cells.pop(cid)
        self.depth.pop(cid, None)
        for e in (fs(p, b0), fs(b0, b1), fs(b1, p)):
            self.edge2cells[e].discard(cid)
            if not self.edge2cells[e]:
                del self.edge2cells[e]

    def _midpoint(self, e):
        if e not in self.edge2mid:
            a, b = e
            self.coords.append(0.5 * (self.coords[a] + self.coords[b]))
            self.edge2mid[e] = len(self.coords) - 1
        return self.edge2mid[e]

    def _ref_edge(self, cid):
        p, b0, b1 = self.cells[cid]
        return fs(b0, b1)

    def _neighbour(self, cid, e):
        others = self.edge2cells.get(e, set()) - {cid}
        return next(iter(others)) if others else None

    # ---- the bisection -----------------------------------------------------
    def _split(self, cid, m):
        p, b0, b1 = self.cells[cid]
        d = self.depth[cid] + 1
        self._del_cell(cid)
        self._add_cell((m, p, b0), d)  # newest vertex m, ref edge {p,b0}
        self._add_cell((m, p, b1), d)  # newest vertex m, ref edge {p,b1}

    def bisect(self, cid, _depth=0):
        if _depth > 100000:
            raise RuntimeError("NVB closure did not terminate (LEPP-like loop)")
        e = self._ref_edge(cid)
        nb = self._neighbour(cid, e)
        # make compatible: neighbour must share e as ITS refinement edge
        while nb is not None and self._ref_edge(nb) != e:
            self.bisect(nb, _depth + 1)
            nb = self._neighbour(cid, e)        # a child of the old neighbour
        m = self._midpoint(e)
        self._split(cid, m)
        if nb is not None:
            self._split(nb, m)

    def refine(self, marked):
        # marked is a set of current cell ids; refine each if still alive
        for cid in list(marked):
            if cid in self.cells:
                self.bisect(cid)

    # ---- export / diagnostics ---------------------------------------------
    def arrays(self):
        coords = np.array(self.coords)
        tris = np.array([(p, b0, b1) for (p, b0, b1) in self.cells.values()])
        return coords, tris

    def depths(self):
        """Per-cell bisection depth from its base ancestor, in cells order
        (clean refinement colouring on a non-uniform base, unlike area-derived)."""
        return np.array([self.depth[cid] for cid in self.cells])

    def check_conforming(self):
        """No hanging nodes: no live cell may carry an edge that has been
        bisected (i.e. that has a midpoint)."""
        bad = 0
        for cid, (p, b0, b1) in self.cells.items():
            for e in (fs(p, b0), fs(b0, b1), fs(b1, p)):
                if e in self.edge2mid:
                    bad += 1
        # every interior edge shared by exactly 2 cells, boundary by 1
        overshared = sum(1 for e, cs in self.edge2cells.items() if len(cs) > 2)
        return bad, overshared

    def cell_centroids_levels(self, base_area):
        """Returns (centroids, generation, area). 'generation' counts BISECTIONS
        (NVB bisects 1->2, so area halves each bisection); a full isotropic SBR
        'level' (1->4) is two generations."""
        coords = np.array(self.coords)
        tris = np.array([(p, b0, b1) for (p, b0, b1) in self.cells.values()])
        cen = coords[tris].mean(axis=1)
        a = coords[tris[:, 0]]; b = coords[tris[:, 1]]; c = coords[tris[:, 2]]
        area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                            (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
        gen = np.round(np.log2(base_area / area)).astype(int)
        return cen, gen, area
