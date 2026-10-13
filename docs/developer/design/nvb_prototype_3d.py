"""3D newest-vertex bisection (NVB) oracle — stage 1a of the adaptivity
capstone (``ADAPTIVITY_3D_SPHERICAL_2026-07.md``).

Pure-numpy tagged-tetrahedron bisection following Maubach's routine with the
Diening--Gehring--Storn (DGS) coloring initialization, which is proven to
terminate with bounded conforming closure, bounded similarity classes
(``n! * n * 2**(n-2)`` = 36 per base tet) and shape regularity for ANY
conforming initial triangulation in any dimension (arXiv:2306.02674,
Theorems 7 and 8). This removes the "compatible initial labelling" risk that
the classical Maubach/Traxler/Stevenson route carries on unstructured meshes.

The bisection rule (DGS Algorithm 1, Maubach 1995): a tagged n-simplex
``T = [v0, ..., vn]_g`` with tag ``g`` in {1..n} bisects its edge ``v0--vg``
at midpoint ``m``; the children (both tagged ``g-1``, or ``n`` when ``g==1``)
are::

    T1 = [v0, v1, ..., v_{g-1}, m, v_{g+1}, ..., vn]    (m replaces v_g)
    T2 = [v1, v2, ..., v_g,     m, v_{g+1}, ..., vn]    (v0 dropped)

Initialization (DGS Definition 2 + Algorithm 2): greedy-color the vertices in
a deterministic order (smallest color not used by an edge-neighbour); in each
tet sort vertices by ascending color; if the tet contains the globally
largest color N, that vertex goes FIRST; tag = n.

The conforming closure (DGS Algorithm 3): to bisect T at ``e = bse(T)``,
every tet in the edge star of ``e`` must have ``e`` as its own bisection
edge; incompatible star members are recursively bisected first, then the
whole star splits at the shared midpoint.

Acceptance battery (mirrors ``nvb_prototype_2d.py`` plus the MG-structural
checks the coordinate-based custom-P tail relies on):

* conformity after every operation (no live cell owns a bisected edge; every
  face shared by <= 2 cells; volume conserved);
* bounded closure (one tet deep in a uniformly refined patch -> O(1) cells);
* graded bullseye (generation rings, not a uniform core);
* similarity classes plateau under deep uniform refinement (<= 36 per base
  class);
* children geometrically nested in parents (fine-in-coarse point location is
  what the barycentric prolongation builder does).

Run:  python nvb_prototype_3d.py            (numpy required, scipy optional)
"""

import numpy as np


def _fs(a, b):
    """Canonical (sorted) vertex-pair key for an edge."""
    return (a, b) if a < b else (b, a)


def _tet_edges(verts):
    """The 6 canonical edge keys of a 4-vertex cell."""
    v = verts
    return (_fs(v[0], v[1]), _fs(v[0], v[2]), _fs(v[0], v[3]),
            _fs(v[1], v[2]), _fs(v[1], v[3]), _fs(v[2], v[3]))


class NVBMesh3D:
    """A tagged-tetrahedron bisection mesh (Maubach rule, DGS coloring init).

    ``cells[cid] = (verts, tag)`` with ``verts`` an ORDERED 4-tuple (the
    Maubach ordering IS the refinement state) and ``tag`` in {1, 2, 3}.
    ``depth[cid]`` counts bisections from the base cell (the generation).
    """

    NDIM = 3

    def __init__(self, coords, tets):
        self.coords = [np.asarray(p, float) for p in coords]
        self.cells = {}                 # cid -> (ordered verts tuple, tag)
        self.depth = {}                 # cid -> bisection depth
        self.parent = {}                # cid -> parent cid (or None)
        self.edge2cells = {}            # edge key -> set(cid)
        self.edge2mid = {}              # edge key -> midpoint vertex id
        self._next = 0
        self._init_coloring_and_tags(tets)

    # ---- DGS initialization -------------------------------------------
    def _init_coloring_and_tags(self, tets):
        """Greedy vertex coloring in deterministic (coordinate-lexicographic)
        order, then the DGS Definition-2 vertex ordering + tag n per cell."""
        nv = len(self.coords)
        adj = [set() for _ in range(nv)]
        for t in tets:
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
        N = max(color)
        self.vertex_color = color
        for t in tets:
            vs = sorted(t, key=lambda v: color[v])   # ascending color
            if color[vs[-1]] == N:                   # global max color first
                vs = [vs[-1]] + vs[:-1]
            self._add_cell(tuple(vs), self.NDIM, depth=0, parent=None)

    # ---- bookkeeping ---------------------------------------------------
    def _add_cell(self, verts, tag, depth, parent):
        cid = self._next
        self._next += 1
        self.cells[cid] = (verts, tag)
        self.depth[cid] = depth
        self.parent[cid] = parent
        for e in _tet_edges(verts):
            self.edge2cells.setdefault(e, set()).add(cid)
        return cid

    def _del_cell(self, cid):
        verts, _ = self.cells.pop(cid)
        self.depth.pop(cid, None)
        for e in _tet_edges(verts):
            s = self.edge2cells.get(e)
            if s is not None:
                s.discard(cid)
                if not s:
                    del self.edge2cells[e]

    def bse(self, cid):
        """Bisection edge of cell ``cid`` (ordered endpoints v0, v_tag)."""
        verts, tag = self.cells[cid]
        return verts[0], verts[tag]

    def _midpoint(self, ekey):
        if ekey not in self.edge2mid:
            a, b = ekey
            self.coords.append(0.5 * (self.coords[a] + self.coords[b]))
            self.edge2mid[ekey] = len(self.coords) - 1
        return self.edge2mid[ekey]

    # ---- the Maubach split ----------------------------------------------
    def _split(self, cid, m):
        verts, g = self.cells[cid]
        gp = g - 1 if g >= 2 else self.NDIM
        d = self.depth[cid] + 1
        v = list(verts)
        t1 = tuple(v[:g] + [m] + v[g + 1:])          # m replaces v_g
        t2 = tuple(v[1:g + 1] + [m] + v[g + 1:])     # v0 dropped
        self._del_cell(cid)
        self._add_cell(t1, gp, d, cid)
        self._add_cell(t2, gp, d, cid)

    # ---- bisection with recursive conforming closure ---------------------
    def bisect(self, cid, _depth=0):
        """Bisect ``cid`` at its bisection edge, first making the edge
        *compatible* (the bisection edge of every tet in its star) by
        recursively bisecting incompatible star members — DGS Algorithm 3.
        If ``cid`` itself is consumed by a recursive closure step, its
        bisection has happened and the call returns."""
        if _depth > 100000:
            raise RuntimeError("NVB-3D closure did not terminate")
        ekey = _fs(*self.bse(cid))
        while True:
            if cid not in self.cells:
                return                       # bisected during closure
            star = self.edge2cells.get(ekey, set())
            bad = [c for c in star if _fs(*self.bse(c)) != ekey]
            if not bad:
                break
            self.bisect(bad[0], _depth + 1)
        m = self._midpoint(ekey)
        for c in list(self.edge2cells.get(ekey, ())):
            self._split(c, m)

    def refine(self, marked):
        """Bisect every still-live cell id in ``marked`` once."""
        for cid in list(marked):
            if cid in self.cells:
                self.bisect(cid)

    def refine_uniform(self, passes=1):
        """``passes`` sweeps of bisect-every-live-cell (n sweeps = one full
        isotropic refinement level in the Maubach scheme)."""
        for _ in range(passes):
            self.refine(list(self.cells))

    # ---- geometry / diagnostics ------------------------------------------
    def arrays(self):
        coords = np.array(self.coords)
        cids = list(self.cells)
        tets = np.array([self.cells[c][0] for c in cids], dtype=np.int64)
        return coords, tets, np.array(cids, dtype=np.int64)

    def volumes(self):
        coords, tets, _ = self.arrays()
        e = coords[tets[:, 1:]] - coords[tets[:, :1]]
        return np.abs(np.linalg.det(e)) / 6.0

    def centroids(self):
        coords, tets, cids = self.arrays()
        return coords[tets].mean(axis=1), cids

    def check_conforming(self):
        """(hanging, overshared, min_vol): a live cell owning a bisected edge
        is a hanging node; a face triple shared by >2 cells is broken
        topology; min_vol must stay positive."""
        hanging = 0
        for verts, _ in self.cells.values():
            for e in _tet_edges(verts):
                if e in self.edge2mid:
                    hanging += 1
        face_count = {}
        for verts, _ in self.cells.values():
            v = verts
            for f in ((v[0], v[1], v[2]), (v[0], v[1], v[3]),
                      (v[0], v[2], v[3]), (v[1], v[2], v[3])):
                face_count[tuple(sorted(f))] = \
                    face_count.get(tuple(sorted(f)), 0) + 1
            # a face shared by >2 cells is impossible in a conforming mesh
        overshared = sum(1 for n in face_count.values() if n > 2)
        vols = self.volumes()
        return hanging, overshared, float(vols.min(initial=np.inf))

    def similarity_classes(self, ndigits=8):
        """(global, per_base_max): distinct tet shapes (sorted, scale-
        normalised edge-length 6-tuples) over all live cells, and the
        maximum count among the descendants of any single base cell.
        DGS Theorem 7(b) bounds the PER-BASE count by n!·n·2^(n-2) = 36;
        a blow-up there signals a wrong child rule."""
        coords = np.array(self.coords)
        classes = set()
        per_base = {}
        for cid, (verts, _) in self.cells.items():
            P = coords[list(verts)]
            ls = np.sort([np.linalg.norm(P[i] - P[j])
                          for i in range(4) for j in range(i + 1, 4)])
            key = tuple(np.round(ls / ls[-1], ndigits))
            classes.add(key)
            a = cid
            while self.parent[a] is not None:
                a = self.parent[a]
            per_base.setdefault(a, set()).add(key)
        return len(classes), max(len(s) for s in per_base.values())

    def check_nesting(self):
        """MG-structural check: every live cell's vertices lie inside its
        base ancestor (barycentric coords in [0,1] up to rounding) — the
        invariant the coordinate-based custom-P point location relies on."""
        coords = np.array(self.coords)
        base = {}
        for cid in self.cells:
            a = cid
            while self.parent[a] is not None:
                a = self.parent[a]
            base[cid] = a
        worst = 0.0
        for cid, (verts, _) in self.cells.items():
            bverts, _ = self._base_cells[base[cid]]
            B = coords[list(bverts)]
            T = np.column_stack([B[1] - B[0], B[2] - B[0], B[3] - B[0]])
            Tinv = np.linalg.inv(T)
            for v in verts:
                lam = Tinv @ (coords[v] - B[0])
                bary = np.concatenate([[1.0 - lam.sum()], lam])
                worst = max(worst, float(-bary.min(initial=0.0)))
        return worst

    def snapshot_base(self):
        """Record the base cells (call right after construction) so
        ``check_nesting`` can locate ancestors after they are consumed."""
        self._base_cells = {cid: (verts, tag)
                            for cid, (verts, tag) in self.cells.items()}


# ---- base meshes ---------------------------------------------------------

def kuhn_cube_mesh(n=2):
    """Structured n^3 grid of unit cubes, each split into 6 Kuhn tets —
    the 'reflected' baseline every bisection paper starts from."""
    idx = {}
    coords = []
    for i in range(n + 1):
        for j in range(n + 1):
            for k in range(n + 1):
                idx[(i, j, k)] = len(coords)
                coords.append(np.array([i, j, k], float) / n)
    # Kuhn: the 6 permutation paths from (0,0,0) to (1,1,1)
    import itertools
    tets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                o = np.array([i, j, k])
                for perm in itertools.permutations(range(3)):
                    path = [o.copy()]
                    for p in perm:
                        q = path[-1].copy()
                        q[p] += 1
                        path.append(q)
                    tets.append(tuple(idx[tuple(q)] for q in path))
    return coords, tets


def delaunay_mesh(npts=60, seed=7):
    """Unstructured conforming tet mesh: Delaunay of random points in the
    unit cube (plus the 8 corners so the hull is the cube). The arbitrary-
    initial-triangulation stress test for the DGS initialization."""
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(seed)
    pts = np.vstack([rng.random((npts, 3)),
                     np.array(np.meshgrid([0, 1], [0, 1], [0, 1])
                              ).reshape(3, -1).T.astype(float)])
    dt = Delaunay(pts)
    keep = [tuple(int(v) for v in s) for s in dt.simplices]
    coords = [pts[i] for i in range(len(pts))]
    # drop degenerate slivers from the random cloud (volume ~ 0)
    good = []
    for t in keep:
        e = np.array([coords[t[i]] - coords[t[0]] for i in (1, 2, 3)])
        if abs(np.linalg.det(e)) / 6.0 > 1e-12:
            good.append(t)
    return coords, good


# ---- acceptance battery ---------------------------------------------------

def _assert_conforming(mesh, label):
    h, o, vmin = mesh.check_conforming()
    ok = (h == 0 and o == 0 and vmin > 0)
    print(f"    {label}: cells={len(mesh.cells)} hanging={h} "
          f"overshared={o} min_vol={vmin:.3e} -> {'OK' if ok else 'FAIL'}")
    return ok


def run_battery(name, coords, tets):
    print(f"\n== {name}: {len(tets)} base tets, {len(coords)} vertices ==")
    ok = True

    # 1. uniform refinement: conforming after EVERY sweep; the per-base-cell
    # similarity-class count must respect the DGS Theorem 7(b) bound of 36
    # (the global count merely saturates and is reported for interest).
    mesh = NVBMesh3D(coords, tets)
    mesh.snapshot_base()
    vol0 = mesh.volumes().sum()
    hist = []
    for sweep in range(9):        # 9 sweeps = 3 full isotropic levels
        mesh.refine_uniform(1)
        ok &= _assert_conforming(mesh, f"sweep {sweep + 1}")
        hist.append(mesh.similarity_classes())
    print(f"  [uniform] classes per sweep (global, per-base max): {hist}")
    per_base_max = hist[-1][1]
    print(f"  [uniform] per-base-cell classes {per_base_max} "
          f"(theorem bound 36)")
    ok &= per_base_max <= 36
    dv = abs(mesh.volumes().sum() - vol0) / max(vol0, 1e-300)
    print(f"  [uniform] volume drift {dv:.2e}")
    ok &= dv < 1e-12
    worst = mesh.check_nesting()
    print(f"  [uniform] nesting: worst barycentric excursion {worst:.2e}")
    ok &= worst < 1e-9

    # 2. bounded closure: one cell deep inside a uniformly refined patch
    mesh = NVBMesh3D(coords, tets)
    mesh.snapshot_base()
    mesh.refine_uniform(6)                           # 2 full levels
    n_before = len(mesh.cells)
    cen, cids = mesh.centroids()
    target = cids[np.argmin(np.linalg.norm(cen - 0.5, axis=1))]
    mesh.bisect(int(target))
    added = len(mesh.cells) - n_before
    ok &= _assert_conforming(mesh, "deep single mark")
    print(f"  [closure] one deep mark added {added} cells "
          f"(bounded-local expected, SBR-style drain would be thousands)")
    ok &= added < 200

    # 3. graded bullseye: shrinking marked radii, one full isotropic level
    # (3 mark+bisect sweeps) per radius -> generation rings, finest central
    mesh = NVBMesh3D(coords, tets)
    mesh.snapshot_base()
    for r in (0.40, 0.25, 0.15, 0.09):
        for _ in range(3):
            cen, cids = mesh.centroids()
            marked = cids[np.linalg.norm(cen - 0.5, axis=1) < r]
            mesh.refine(list(marked))
        ok &= _assert_conforming(mesh, f"bullseye r<{r}")
    cen, cids = mesh.centroids()
    depth = np.array([mesh.depth[c] for c in cids])
    print("  [bullseye] generation -> (count, max centroid radius):")
    for g in sorted(set(depth)):
        sel = depth == g
        rmax = float(np.linalg.norm(cen[sel] - 0.5, axis=1).max())
        print(f"      gen {g:2d}: {int(sel.sum()):6d} cells, r_max={rmax:.3f}")
    deepest = depth.max()
    r_deep = float(np.linalg.norm(cen[depth == deepest] - 0.5, axis=1).max())
    ok &= r_deep < 0.25                # finest generation stays central
    print(f"  [bullseye] total {len(mesh.cells)} cells; deepest gen "
          f"{deepest} confined to r<{r_deep:.3f}")
    worst = mesh.check_nesting()
    print(f"  [bullseye] nesting: worst barycentric excursion {worst:.2e}")
    ok &= worst < 1e-9

    print(f"== {name}: {'PASS' if ok else 'FAIL'} ==")
    return ok


if __name__ == "__main__":
    all_ok = True
    coords, tets = kuhn_cube_mesh(2)
    all_ok &= run_battery("Kuhn 2x2x2 (structured/reflected)", coords, tets)
    try:
        coords, tets = delaunay_mesh(60)
        all_ok &= run_battery("Delaunay-60 (arbitrary unstructured)",
                              coords, tets)
    except ImportError:
        print("\n(scipy not available - skipping the Delaunay stress test)")
    print(f"\nORACLE VERDICT: {'PASS' if all_ok else 'FAIL'}")
