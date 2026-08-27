"""The user-facing 2-D fault-network toolkit.

A fault is specified ONCE — a trace, its place in the hierarchy, and
the properties it carries — and then realised. The realisation is a
keyword, not a different subsystem: the same specification, the same
prepared pieces and the same meshed band become either a cut with
node-pair contact (``realisation="split"``) or a volumetric weak plane
(``realisation="ti"``). The band carries the fault's own points and
segments as mesh vertices and edges, so it can be cut whatever its
width — the choice of realisation is not constrained by the mesh.
What differs is what the width MEANS: for the split it is a resolution
parameter that gives the cut its vertices, while for the weak plane it
is constitutive (``V = 2 e_nt w``) and wants two or three elements
across it.

One object carries the validated network recipe end to end:

1. **Hierarchy-respecting junction preparation** — where faults cross
   or abut, the junior trace is severed and pulled back a short
   ligament; the senior runs through (:func:`prepare_fault_network`).
2. **Network-refined meshing** — a graded mesh following every trace,
   then one ribbon band placed along every prepared piece, cut or left
   whole according to the realisation.
3. **Imposition** — :meth:`FaultNetwork.apply` gives the solver the
   no-opening pair constraint, or the weak-plane rheology.
4. **Damage-zone glue** — small viscoplastic plugs at the junctions
   connect the network mechanically; the stress lobes of the abutting
   tips decide how slip transfers, no reconnection geometry is ever
   prescribed.
5. **Fault-attached properties** — :meth:`FaultNetwork.surface` returns
   the retained :class:`~underworld3.meshing.surfaces.Surface` for a
   piece; friction, accumulated slip and damage live there, on the
   fault, and outlive any one realisation of it.

The dials encode the measured rulings (2026-08): junction gaps of one
or two elements transmit slip within a few percent of a continuous
fault and cost nothing in solver health; the glue's strength and
regularisation dial move TOGETHER (``dial``), with zone stress
proportional to the dial down to ~100x viscosity contrast before
anything degrades — the compact plug never triggers thin-inclusion
conditioning. Oriented (TI) damage in the plug is deliberately NOT
offered: a junction must accommodate corner-turning deformation,
which a weak-plane fabric cannot relax (measured; see the study
README under ``~/+Simulations/fault_junction_rheology``).

Example
-------
>>> net = uw.meshing.FaultNetwork(
...     [("Main", main_pts), ("Splay", splay_pts)],
...     hierarchy=["Main", "Splay"])          # Main severs Splay
>>> mesh = net.prepare(h=0.006).build(width=0.01)   # realisation="split"
>>> stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
>>> stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
>>> stokes.constitutive_model.yield_mode = "min"
>>> stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
>>> stokes.constitutive_model.Parameters.yield_stress = \\
...     net.damage_yield(v, dial=0.05)
>>> stokes.consistent_jacobian = True
>>> net.apply(stokes)
>>> # ... boundary conditions ...
>>> info = net.solve(stokes)
>>> net.slips(stokes)
{'Main': 0.14, 'Splay_1': 0.05, 'Splay_2': 0.04}

The same specification as a weak plane — one keyword, one more
constitutive number, everything else unchanged::

>>> mesh = net.prepare(h=0.006).build(width=0.01, realisation="ti")
>>> net.apply(stokes, eta_1=0.01)
>>> net.slips(stokes)                  # the layer's own throughput
"""

import numpy as np
import sympy

from .surfaces import Surface, prepare_fault_network
from .faults import FaultSurface


def _nearest_segment_normals(P, X):
    """Unit normal of the polyline segment nearest to each point of ``X``.

    The director of a weak plane, cell by cell: a curved trace has no one
    orientation, and the nearest SEGMENT is the piece of fault the cell
    actually lies against. 2-D; ``P`` is ``(n, 2)`` and ``X`` ``(m, 2)``.
    """
    P = np.asarray(P, dtype=float)[:, :2]
    X = np.asarray(X, dtype=float)[:, :2]
    A, D = P[:-1], np.diff(P, axis=0)
    L2 = np.maximum(np.einsum("sj,sj->s", D, D), 1e-300)
    W = X[:, None, :] - A[None, :, :]
    t = np.clip(np.einsum("psj,sj->ps", W, D) / L2[None, :], 0.0, 1.0)
    R = W - t[:, :, None] * D[None, :, :]
    k = np.argmin(np.einsum("psj,psj->ps", R, R), axis=1)
    T = D[k] / np.sqrt(L2[k])[:, None]
    return np.column_stack([-T[:, 1], T[:, 0]])


def _densify_polyline(E, margin, per_segment=4):
    """Points along an extended polyline, ``per_segment`` per edge, each
    flagged as lying on the CUT (an edge between two user points) or in
    the extrapolated margin. ``margin`` is the ``(start, end)`` count of
    margin points of ``E``."""
    E = np.asarray(E, dtype=float)
    n = len(E)
    m0, m1 = margin
    f = np.linspace(0.0, 1.0, per_segment, endpoint=False)
    Q = (E[:-1, None, :] + f[None, :, None] * np.diff(E, axis=0)[:, None, :])
    Q = np.vstack([Q.reshape(-1, E.shape[1]), E[-1:]])
    edge_on_cut = np.array([m0 <= i and i + 1 <= n - 1 - m1
                            for i in range(n - 1)])
    on_cut = np.concatenate([np.repeat(edge_on_cut, per_segment),
                             [False]])
    return Q, on_cut


class FaultNetwork:
    """A hierarchy of fault traces (2-D) or planar patches (3-D)
    meshed, split, and glued.

    Parameters
    ----------
    faults : sequence
        2-D: ``(name, points)`` open polylines (``(N, 2)`` or
        ``(N, 3)``), exactly as imported — crossings and abutments
        included; the toolkit converts them.
        3-D: triangulated :class:`FaultSurface` objects (planar,
        convex rims) — crossing patches are trimmed to the offset
        form by the 3-D preparer.
    hierarchy : sequence of str, optional
        Names in seniority order (most major first). Junior traces are
        severed by senior ones at crossings. Default: the input order.
    """

    def __init__(self, faults, hierarchy=None):
        faults = list(faults)
        self.dim = 3 if any(isinstance(f, FaultSurface)
                            for f in faults) else 2
        if self.dim == 3:
            if not all(isinstance(f, FaultSurface) for f in faults):
                raise ValueError("mixing FaultSurface and 2-D traces "
                                 "in one network is not supported")
            self.surfaces = list(faults)
            self.faults = [(s.name, np.asarray(s.rim_polygon(),
                                               dtype=float))
                           for s in faults]
        else:
            self.surfaces = None
            self.faults = [(str(n), np.asarray(p, dtype=float)[:, :2])
                           for n, p in faults]
        names = [n for n, _ in self.faults]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate fault names: {names}")
        self.hierarchy = list(hierarchy) if hierarchy is not None \
            else list(names)
        unknown = set(self.hierarchy) - set(names)
        if unknown:
            raise ValueError(f"hierarchy names not in the network: "
                             f"{sorted(unknown)}")
        self.h_near = None
        self.prepared = None
        self.junctions = None
        self.report = None
        self.mesh = None
        # the realisation state, set by build()
        self.realisation = None
        self.width = None
        self.info = None
        self.fault_surfaces = {}
        self.ti = None
        self.glue = None
        self.margin_rings = None
        self._eta0_var = None
        self._band_yield_var = None

    # ------------------------------------------------------------------
    def prepare(self, h, ligament=2.0, through=None, verbose=True):
        """Convert crossings/abutments to offset junctions.

        ``h`` is the near-fault mesh size the network will be built at;
        ``ligament`` (in multiples of ``h``) sets the junction gap —
        the measured floor is 1, the default 2 gives one clear cell of
        intact material and slip transmission within ~5% of a
        continuous fault. ``through`` declares ABSOLUTE masters (never
        cut anywhere); the hierarchy handles everything else pairwise.
        """
        self.h_near = float(h)
        if self.dim == 3:
            from .fault_network_3d import prepare_fault_surfaces
            if through:
                raise NotImplementedError(
                    "3-D preparation ranks by hierarchy only")
            self.prepared, self.report, self.junctions = \
                prepare_fault_surfaces(
                    self.surfaces, spacing=self.h_near,
                    ligament=ligament, hierarchy=self.hierarchy,
                    verbose=verbose)
        else:
            self.prepared, self.report, self.junctions = \
                prepare_fault_network(
                    self.faults, spacing=self.h_near, ligament=ligament,
                    through=through, hierarchy=self.hierarchy,
                    verbose=verbose, return_junctions=True)
        return self

    # ------------------------------------------------------------------
    def build(self, base=None, h_far=None, band=0.03, ramp=0.08,
              max_levels=2, qdegree=2, mesher=None,
              width=None, realisation="split",
              margin_rings=2, carve_clearance=0.3):
        """Mesh the network: graded refinement along every RAW trace,
        then the chosen REALISATION of the faults on that mesh.

        ``base`` is an existing coarse mesh to adapt (default: a unit
        ``UnstructuredSimplexBox`` at ``h_far = 4 h``); the refinement
        holds ``h`` within ``band`` of any trace and grades to
        ``h_far`` over ``ramp``.

        ``width`` is the fault band's thickness. Give it and the network
        is placed as a ribbon band (2-D) that both realisations share:
        the SAME mesh is cut and split (``realisation="split"``) or left
        whole for a volumetric weak-plane rheology
        (``realisation="ti"``), which is what makes the two comparable.
        What ``width`` MEANS differs: for the split it is a resolution
        parameter — the band exists to give the cut its own vertices —
        while for TI it is constitutive, the layer thickness that sets
        the slip rate ``V = 2 e_nt w``, so it wants two or three elements
        across it. The width does NOT decide whether the fault can be
        cut: the band is meshed around the trace's own points and
        segments, so the cut chain is part of the mesh by construction at
        any width (measured to ``w = h_far / 10``; see
        ``~/+Simulations/fault_split_at_ti_width``).

        ``width=None`` keeps the original no-band path: graded refinement
        cut directly. It is split-only, and its mesh is NOT the one a TI
        run would use, so do not compare the two across that choice.

        ``mesher`` picks how the fault is meshed into the mesh, from
        the choices that dimension offers. In 2-D: ``"network"``
        (default) places every strand in one fused call, so strands may
        touch, and ``"ladder"`` places them sequentially and lets placed
        levels nest (see
        :func:`~underworld3.utilities.place_surface.place_fault_ribbon_2d`).
        In 3-D: ``"embed"`` (default) and ``"place"``, described in
        :meth:`_build_3d`.
        """
        if self.prepared is None:
            raise RuntimeError("call prepare(h=...) first")
        meshers = {2: ("network", "ladder"), 3: ("embed", "place")}[self.dim]
        if mesher is None:
            mesher = meshers[0]
        if mesher not in meshers:
            raise ValueError(
                f"mesher must be one of {meshers} in {self.dim}-D, not "
                f"{mesher!r}")
        if realisation not in ("split", "ti"):
            raise ValueError(
                f"realisation must be 'split' or 'ti', not {realisation!r}")
        if realisation == "ti" and width is None:
            raise ValueError(
                "realisation='ti' needs width=: the weak plane is a LAYER, "
                "and its thickness is constitutive (V = 2 e_nt w). Pass the "
                "band width you intend to resolve.")
        self.realisation = realisation
        self.width = None if width is None else float(width)
        if self.dim == 3:
            if realisation != "split":
                raise NotImplementedError(
                    "the 3-D network builds the split realisation only; "
                    "place the patches with place_thin_volume for a "
                    "volumetric zone.")
            if width is not None:
                raise NotImplementedError(
                    "the 3-D network does not place a band: its patches "
                    "are meshed conforming (mesher='embed') or placed as "
                    "sheets (mesher='place'), both of zero thickness. For "
                    "a finite-width 3-D zone call place_thin_volume.")
            return self._build_3d(h_far=h_far, qdegree=qdegree,
                                  mesher=mesher)
        from .cartesian import UnstructuredSimplexBox

        h = self.h_near
        h_far = 4.0 * h if h_far is None else float(h_far)
        if base is None:
            base = UnstructuredSimplexBox(cellSize=h_far, refinement=1,
                                          qdegree=qdegree)
        surfs = []
        for n, p in self.faults:            # RAW traces: the metric must
            s = Surface(f"FN_{n}", base,     # refine the junction gaps too
                        np.column_stack([p, np.zeros(len(p))]),
                        symbol=n[:1])
            s.discretize()
            surfs.append(s)

        def metric(pts_, _ss=surfs, _h=h, _hf=h_far, _b=band, _r=ramp):
            d = np.min([s_.unsigned_distance(pts_) for s_ in _ss], axis=0)
            hh = np.where(d < _b, _h,
                          np.minimum(_h + (_hf - _h) * (d - _b) / _r, _hf))
            return 1.0 / hh ** 2

        child = base.adapt(metric, max_levels=max_levels)
        if width is None:
            self.mesh = child.add_fault(
                [(n, p.copy()) for n, p in self.prepared])
            self.info = None
        else:
            from underworld3.utilities.place_surface import (
                place_fault_ribbon_2d)
            # The ribbons must MEET across every junction ligament: the
            # ribbon is the fault as the weak plane sees it, continuous
            # through a junction where the cut stops short, and that
            # difference is what junction_cells() reads. So the tip
            # margin reaches at least as far as the longest pull-back.
            margin_rings = self._junction_margins(int(margin_rings))
            self.margin_rings = margin_rings
            self.mesh, self.info = place_fault_ribbon_2d(
                child, [(n, p.copy()) for n, p in self.prepared],
                self.width, margin_rings=margin_rings,
                clearance=carve_clearance,
                split=(realisation == "split"), mesher=mesher)
        self._make_surfaces()
        return self.mesh

    def _junction_margins(self, rings):
        """Per-end margin counts: ``rings`` at a free tip; at an end that
        sits on a prepared junction, enough rings for the ribbon to reach
        the OTHER piece's cut (plus half a width) — the whole ligament
        lies in both ribbons, so the junction cells cover it end to end.
        The margin is built by continuing the end segment, so the count
        depends on that segment's length."""
        pieces = {n: np.asarray(P, dtype=float) for n, P in self.prepared}
        out = []
        for name, P in self.prepared:
            P = pieces[name]
            ends = []
            for end, seg in ((P[0], P[1] - P[0]), (P[-1], P[-1] - P[-2])):
                reach = 0.0
                for j in (self.junctions or []):
                    if name not in j["faults"]:
                        continue
                    pull = float(j["pull"])
                    near = (np.linalg.norm(np.asarray(j["point"]) - end)
                            <= pull + self.width)
                    if not near:
                        continue
                    other = [n for n in j["faults"] if n != name][0]
                    gap = float(np.linalg.norm(pieces[other] - end,
                                               axis=1).min())
                    reach = max(reach, gap + 0.5 * self.width)
                seg_len = float(np.linalg.norm(seg))
                ends.append(max(rings, int(np.ceil(reach / seg_len)) + 1)
                            if reach > 0 else rings)
            out.append((ends[0], ends[1]))
        return out

    def _build_3d(self, h_far=None, qdegree=2, mesher="embed",
                  minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
                  band=None, ramp=None, max_levels=2, clearance=0.8):
        """3-D path, two meshers sharing one contract.

        ``mesher="embed"`` (default): the gmsh conforming multi-patch
        embed — fast and proven for NETWORKS (the whole junction study).

        ``mesher="place"`` (EXPERIMENTAL for networks): the native route
        box (``refinement=1``), metric-graded 3-D adaptation along the
        network, each prepared patch triangulated at ``h`` (interior
        grid + rim, no all-rim triangles by construction) and PLACED
        (:func:`place_surface.place_sheet`), then split. Non-cumulative:
        the fault position is a design variable, and moving the network
        re-places from the same base. Parallel-capable end to end
        (#521; ptest_0852 is the chain's gate).

        The place route on a UNIFORM base is healthy and
        parallel-validated (ptest_0852: place -> split -> contact,
        22 s including the solve). On edge_split ADAPT CHILDREN it is
        SLOW but not sick: re-measured 2026-08-20 (#621), the composed
        chain built 27x the cells embed did for the same nominal sizes
        and solved 41x slower — proportionate under 3-D Stokes scaling
        (per-cell cost comparable, one nonlinear iteration, machine-zero
        leak, agreeing slip). The over-build is this route's sizing: the
        base box is built at ``cellSize=h_far`` WITH ``refinement=1``,
        so the far field is ``h_far/2`` everywhere. Until #621 lands,
        place is opt-in here, and expect embed to be much cheaper at
        matched request.
        """
        from underworld3.utilities.fault_split import split_fault

        h = self.h_near
        h_far = 4.0 * h if h_far is None else float(h_far)

        if mesher == "embed":
            from .cartesian import BoxInternalPatch
            mesh = BoxInternalPatch(
                cellSize=h_far, minCoords=minCoords, maxCoords=maxCoords,
                patch_points=[(n, p) for n, p in self.prepared],
                patch_cellSize=h, qdegree=qdegree)
            for name, _p in self.prepared:
                mesh = split_fault(mesh, name)
            self.mesh = mesh
            self._make_surfaces()
            return self.mesh
        if mesher != "place":
            raise ValueError(f"mesher must be 'place' or 'embed', "
                             f"got {mesher!r}")

        from .cartesian import UnstructuredSimplexBox
        from underworld3.utilities.place_surface import _sheet_distance

        sheets = [(n, *self._triangulate_rim(p, h))
                  for n, p in self.prepared]

        base = UnstructuredSimplexBox(
            cellSize=h_far, minCoords=minCoords, maxCoords=maxCoords,
            refinement=1, qdegree=qdegree)
        b = float(band) if band is not None else 3.0 * h
        r = float(ramp) if ramp is not None else 10.0 * h

        def metric(pts_, _sheets=sheets, _h=h, _hf=h_far, _b=b, _r=r):
            d = np.min([_sheet_distance(pts_[:, :3], sp, st)
                        for _n, sp, st in _sheets], axis=0)
            hh = np.where(d < _b, _h,
                          np.minimum(_h + (_hf - _h) * (d - _b) / _r,
                                     _hf))
            return 1.0 / hh ** 2

        child = base.adapt(metric, max_levels=max_levels,
                           engine="edge_split")

        mesh = child
        for n, sp, st in sheets:
            # clearance 0.8 measured as the working window on
            # edge_split children (0.6 under-reaches the graded
            # transition shell and pinches; >=1.0 over-swallows).
            # Mesh-level placement, so each cut child inherits the adapt
            # hierarchy as its coarse multigrid tail (the 2-D contract;
            # the split below still forfeits it — see add_fault).
            mesh = mesh.add_conforming_sheet(sp, st, n, clearance=clearance)
        for n, _sp, _st in sheets:
            mesh = split_fault(mesh, n)
        self.mesh = mesh
        self._make_surfaces()
        return self.mesh

    @staticmethod
    def _triangulate_rim(poly, h):
        """Triangulate a convex planar rim polygon at spacing ``h``:
        boundary resampled at ``h``, an interior grid at ``h``, in-plane
        Delaunay. No all-rim triangles by construction (asserted) —
        the split's precondition, and interior points guarantee it
        everywhere a patch is wider than a cell."""
        from scipy.spatial import Delaunay
        from .fault_network_3d import _plane_basis, _to_2d, _to_3d

        o, nh, uh, vh = _plane_basis(np.asarray(poly, dtype=float))
        p2 = _to_2d(poly, o, uh, vh)
        # rim resampled at h
        rim_pts = []
        for i in range(len(p2)):
            a, b = p2[i], p2[(i + 1) % len(p2)]
            seg = np.linalg.norm(b - a)
            m = max(1, int(np.ceil(seg / h)))
            for t in np.arange(m) / m:
                rim_pts.append(a + t * (b - a))
        rim_pts = np.array(rim_pts)
        # interior grid at h, kept clear of the rim by 0.4 h
        lo, hi = p2.min(axis=0), p2.max(axis=0)
        gx = np.arange(lo[0] + 0.5 * h, hi[0], h)
        gy = np.arange(lo[1] + 0.5 * h, hi[1], h)
        GX, GY = np.meshgrid(gx, gy)
        grid = np.column_stack([GX.ravel(), GY.ravel()])

        def inside(pts, margin):
            keep = np.ones(len(pts), dtype=bool)
            n_ = len(p2)
            area2 = float(np.dot(p2[:, 0], np.roll(p2[:, 1], -1))
                          - np.dot(p2[:, 1], np.roll(p2[:, 0], -1)))
            sgn = 1.0 if area2 > 0 else -1.0
            for i in range(n_):
                a, b = p2[i], p2[(i + 1) % n_]
                e = b - a
                nrm = sgn * np.array([-e[1], e[0]])
                nrm = nrm / max(np.linalg.norm(nrm), 1e-30)
                keep &= ((pts - a) @ nrm) > margin
            return keep

        grid = grid[inside(grid, 0.4 * h)]

        def centreline():
            # NARROW pieces (a ligament-trimmed strip ~1 cell wide is a
            # NORMAL product of the preparer) get interior points along
            # the principal axis instead of a grid — every triangle of
            # a strip then touches an interior point.
            c0 = p2.mean(axis=0)
            _u2, _s2, vt2 = np.linalg.svd(p2 - c0, full_matrices=False)
            ax = vt2[0]
            t = (p2 - c0) @ ax
            ts = np.arange(t.min() + 0.5 * h, t.max(), h)
            pts_ = c0[None, :] + ts[:, None] * ax[None, :]
            for margin in (0.35 * h, 0.15 * h, 0.05 * h):
                keep = inside(pts_, margin)
                if keep.any():
                    return pts_[keep]
            return np.zeros((0, 2))

        n_rim = len(rim_pts)

        def flip_all_rim(all2, tris):
            """Edge-flip every all-rim triangle against a neighbour
            holding an interior vertex (the corner-quad diagonal flip,
            generalised to Delaunay output). Returns tris or None."""
            tris = [tuple(int(v) for v in t) for t in tris]
            for _round in range(4):
                edge_of = {}
                for i, t in enumerate(tris):
                    for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                        edge_of.setdefault(tuple(sorted(e)), []).append(i)
                bad = [i for i, t in enumerate(tris)
                       if all(v < n_rim for v in t)]
                if not bad:
                    return np.asarray(tris, dtype=np.int64)
                changed = False
                for i in bad:
                    t = tris[i]
                    if not all(v < n_rim for v in t):
                        continue                    # fixed by an earlier flip
                    for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                        key = tuple(sorted(e))
                        nbrs = [j for j in edge_of.get(key, []) if j != i]
                        if not nbrs:
                            continue
                        j = nbrs[0]
                        opp_j = [v for v in tris[j] if v not in key][0]
                        if opp_j < n_rim:
                            continue                # neighbour is no help
                        opp_i = [v for v in t if v not in key][0]
                        a, b = key
                        # the flip: (a,b,opp_i)+(a,b,opp_j) ->
                        #           (a,opp_i,opp_j)+(b,opp_i,opp_j)
                        def area(p, q, r):
                            return abs(np.cross(all2[q] - all2[p],
                                                all2[r] - all2[p]))
                        if (area(a, opp_i, opp_j) < 1e-14
                                or area(b, opp_i, opp_j) < 1e-14):
                            continue                # degenerate flip
                        tris[i] = (a, opp_i, opp_j)
                        tris[j] = (b, opp_i, opp_j)
                        changed = True
                        break
                if not changed:
                    return None
            return None

        for interior in (grid, np.vstack([grid, centreline()])
                         if len(grid) else centreline()):
            all2 = np.vstack([rim_pts, interior]) if len(interior) \
                else rim_pts
            tri = Delaunay(all2)
            cen = all2[tri.simplices].mean(axis=1)
            tris = tri.simplices[inside(cen, 0.0)]
            fixed = flip_all_rim(all2, tris)
            if fixed is not None:
                return _to_3d(all2, o, uh, vh), fixed
        raise NotImplementedError(
            "patch triangulation could not avoid all-rim triangles even "
            "with centreline points and edge flips — the piece is "
            "degenerate; drop it or refine h.")

    # ------------------------------------------------------------------
    def _make_surfaces(self):
        """Retain a :class:`~underworld3.meshing.surfaces.Surface` per
        prepared piece, on the BUILT mesh.

        The fault is specified once and then realised; the surface object
        is what survives both realisations, and it is where properties
        that belong to the FAULT rather than to the mesh live —
        ``add_variable("friction")``, accumulated slip, a damage state.
        The realisation reads them; it does not own them. In 3-D the
        input :class:`FaultSurface` objects play the same role.
        """
        if self.dim == 3:
            self.fault_surfaces = {s.name: s for s in (self.surfaces or [])}
            return self.fault_surfaces
        from .surfaces import Surface

        self.fault_surfaces = {}
        for name, P in self.prepared:
            pts = np.asarray(P, dtype=float)
            if pts.shape[1] == 2:
                pts = np.column_stack([pts, np.zeros(len(pts))])
            self.fault_surfaces[name] = Surface(name, self.mesh, pts)
        return self.fault_surfaces

    def surface(self, name):
        """The retained fault surface for one prepared piece."""
        if not self.fault_surfaces:
            raise RuntimeError("call build() first")
        if name not in self.fault_surfaces:
            raise KeyError(f"no fault surface {name!r}; the network holds "
                           f"{sorted(self.fault_surfaces)}")
        return self.fault_surfaces[name]

    # ------------------------------------------------------------------
    def apply(self, solver, conds=0, eta_1=None, eta_0=1.0, tag="",
              normal=None):
        """Impose the network on ``solver``, in whichever realisation
        ``build()`` made.

        ``realisation="split"`` registers the no-opening contact pair on
        every prepared piece (``conds`` is the datum, 0 for free slip).
        ``realisation="ti"`` paints the weak-plane fields on the honoured
        footprints and hands the solver a
        :class:`~underworld3.constitutive_models.TransverseIsotropicFlowModel`:
        ``eta_1`` (required) is the weak-plane viscosity, ``eta_0`` the
        background — a float, or a per-cell array if the background is
        itself painted (a terrane, say). ``tag`` disambiguates the field
        names when more than one network is applied to one mesh.

        ``normal`` is the split's fault normal (see
        :meth:`~underworld3.systems.Stokes.add_fault_bc`). Pass
        ``"trace"`` when the traces are SAMPLED SMOOTH CURVES — the
        default per-node normal zig-zags at the sampling kinks, and
        the no-opening constraint then notches the slip.
        """
        if self.mesh is None:
            raise RuntimeError("call build() first")
        if self.realisation == "split":
            for name, _p in self.prepared:
                solver.add_fault_bc(conds, boundary=name, normal=normal)
            return self
        if eta_1 is None:
            raise ValueError(
                "realisation='ti' needs eta_1=: the weak-plane viscosity "
                "is the other half of the constitutive pair (with width).")
        import underworld3 as uw

        eta1, ndir, foot = self.ti_fields(eta_1, eta_0=eta_0, tag=tag)
        solver.constitutive_model = \
            uw.constitutive_models.TransverseIsotropicFlowModel
        params = solver.constitutive_model.Parameters
        params.shear_viscosity_0 = (
            float(eta_0) if np.ndim(eta_0) == 0 else self._eta0_var.sym[0])
        params.shear_viscosity_1 = eta1.sym[0]
        params.director = ndir.sym
        return self

    def apply_contact(self, solver, conds=0, normal=None):
        """Register the split-node contact on every prepared piece.

        The split realisation's half of :meth:`apply`, kept under its own
        name for callers that only ever want the contact.
        """
        if self.realisation not in (None, "split"):
            raise RuntimeError(
                f"this network was built as {self.realisation!r}; there "
                f"are no fault pairs to constrain. Use apply().")
        if self.mesh is None:
            raise RuntimeError("call build() first")
        for name, _p in self.prepared:
            solver.add_fault_bc(conds, boundary=name, normal=normal)
        return self

    # ------------------------------------------------------------------
    def ti_fields(self, eta_1, eta_0=1.0, tag=""):
        """The weak-plane (TI) realisation's painted P0 fields.

        ``eta_1`` inside each fault's HONOURED footprint — the band cells
        whose nearest spine sample is a USER point, never the whole band,
        whose margin is extrapolated surround — and the background
        elsewhere; the director is the unit normal of the nearest segment
        of the strand that owns the cell, so a curved trace carries its
        own orientation cell by cell.

        Returns ``(eta_1_var, director_var, footprint_mask)``. The mask is
        also the right ``fac_zone`` key for a multigrid patch (#629).
        """
        if self.info is None:
            raise RuntimeError(
                "no band on this mesh: build(width=...) first (the weak "
                "plane is a layer, and the layer has to be meshed).")
        import underworld3 as uw

        foots = self.info["footprints"]
        foot = np.zeros_like(next(iter(foots.values())))
        for m_ in foots.values():
            foot = foot | m_

        eta1 = uw.discretisation.MeshVariable(
            f"fnEta1{tag}", self.mesh, 1, degree=0)
        eta0_vals = np.broadcast_to(np.asarray(eta_0, dtype=float),
                                    (len(eta1.coords),))
        eta1.array[:, 0, 0] = np.where(foot, float(eta_1), eta0_vals)
        self._eta0_var = None
        if np.ndim(eta_0) != 0:
            self._eta0_var = uw.discretisation.MeshVariable(
                f"fnEta0{tag}", self.mesh, 1, degree=0)
            self._eta0_var.array[:, 0, 0] = eta0_vals

        dim = self.mesh.dim
        ndir = uw.discretisation.MeshVariable(
            f"fnDir{tag}", self.mesh, dim, degree=0, continuous=False)
        cen = np.asarray(ndir.coords)[:, :dim]
        dvals = np.zeros((len(cen), dim))
        dvals[:, -1] = 1.0                  # any unit vector outside the
        for name, P in self.prepared:       # footprints: eta_1 == eta_0
            m_ = foots[name]                # there, so TI is isotropic
            if not m_.any():
                continue
            dvals[m_] = _nearest_segment_normals(P, cen[m_])
        ndir.array[...] = dvals.reshape(ndir.array.shape)
        self.ti = {"eta_1": eta1, "director": ndir, "footprint": foot}
        return eta1, ndir, foot

    # ------------------------------------------------------------------
    @property
    def band(self):
        """The band's cell mask — the material the fault is embedded in.

        The band is not scaffolding for the weak plane. It is a meshed
        region of material AROUND the fault, and the split wants it as
        much: a segmented fault does its interesting work at the tips and
        in the ligaments between strands, and damage there needs cells to
        live in. This mask (and :attr:`footprints`, per strand) is how a
        rheology addresses that region in either realisation.
        """
        if self.info is None:
            raise RuntimeError(
                "no band on this mesh: build(width=...) first")
        return self.info["band"]

    @property
    def footprints(self):
        """Per-strand FAULT footprints — the band cells whose nearest
        spine sample is a USER point, never the extrapolated margin."""
        if self.info is None:
            raise RuntimeError(
                "no band on this mesh: build(width=...) first")
        return self.info["footprints"]

    def band_yield(self, tau_y, tau_far=1.0e8, tag=""):
        """Von Mises yield confined to the band, as an expression.

        The band's cells yield at ``tau_y``; everything else is given
        ``tau_far``, high enough never to yield. Pair it with a
        :class:`~underworld3.constitutive_models.ViscoPlasticFlowModel`
        and ``consistent_jacobian = True``::

            stokes.constitutive_model.Parameters.yield_stress = \
                net.band_yield(tau_y=4.0)

        This is the damage the SPLIT realisation wants. A released fault
        flank sits far below ``tau_y`` and is untouched; the places that
        sit far above it — a strand's tips, the weld where a cut stops
        short, the sliver at a junction — yield by themselves, so the
        breakdown appears where the mechanics puts it rather than where a
        geometric plug was placed. Compare :meth:`damage_yield`, which
        places plugs at the junctions by construction and is the right
        tool when the junction glue itself is the object of study.

        The region is a sharp mask, not a blend: never taper a
        rheological parameter towards a large sentinel.
        """
        if self.info is None:
            raise RuntimeError(
                "no band on this mesh: build(width=...) first (the "
                "damage needs cells to live in)")
        import underworld3 as uw

        ybar = uw.discretisation.MeshVariable(
            f"fnTauY{tag}", self.mesh, 1, degree=0)
        ybar.array[:, 0, 0] = np.where(self.info["band"], float(tau_y),
                                       float(tau_far))
        self._band_yield_var = ybar
        return ybar.sym[0]

    # ------------------------------------------------------------------
    def junction_cells(self, ring=1):
        """The cells where the split cannot join: the ribbon minus the cut.

        The ribbon (the band, with its extrapolated tip margins) is
        everything the weak-plane realisation treats as fault; the cut
        chains are what the split actually sliced. A band cell whose
        nearest point on a piece's extended spine lies in that piece's
        margin is fault the split did not cut. Where such a cell also
        lies inside a SECOND piece's ribbon, and on neither piece's cut,
        two pieces meet there and the split has left them welded — a
        stop-short abutment, a kissing branch, the intact bridge of a
        stepover. Those cells, dilated by ``ring`` vertex rings within
        the band, are the junction cells.

        The rule is geometric and comes entirely from the placement: no
        stress threshold, and the free tips are excluded on purpose (a
        margin that runs into intact material rather than another
        ribbon is a tip, not a junction; damage placed there lengthens
        the fault instead of joining it).

        ``ring=1`` is not optional in practice. Measured on the S-fault
        rig (2026-08-27): the bare junction cells cannot repair the weld
        even when fully plastic (a fifth to a quarter of the deficit),
        because the weld's stiffness lives in the ring of intact material
        around the two tips; one ring restores 0.8-0.97 of a continuous
        fault's transmission at both resolutions tested.

        Returns a boolean cell mask over the mesh's cells.
        """
        if self.info is None:
            raise RuntimeError(
                "no band on this mesh: build(width=...) first (the "
                "junction cells are read off the ribbon)")
        if self.realisation != "split":
            raise RuntimeError(
                "junction cells are the SPLIT realisation's joints; the "
                "weak plane has no cut to fall off")
        from underworld3.utilities.place_surface import _cell_centroids_of

        band = self.info["band"]
        ids, cen = _cell_centroids_of(self.mesh.dm, band)
        mask = np.zeros_like(band)
        if len(ids) == 0:
            return mask
        half_width = 0.5 * self.width
        on_any_cut = np.zeros(len(ids), dtype=bool)
        ribbons = np.zeros(len(ids), dtype=int)
        for k, (_name, P) in enumerate(self.prepared):
            E = np.asarray(self.info["extended"][k], dtype=float)
            Q, on_cut = _densify_polyline(E, self.info["margin_rings"][k])
            d = cen[:, None, :] - Q[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", d, d)
            j = np.argmin(d2, axis=1)
            # a sample spacing's worth of tolerance: the densified spine
            # is a polyline, the cell centroid a point beside it
            tol = 0.35 * float(self.info["spacing"][k])
            inside = np.sqrt(d2[np.arange(len(ids)), j]) <= half_width + tol
            ribbons += inside
            on_any_cut |= inside & on_cut[j]
        # unjoined = in two ribbons and on nobody's cut
        mask[ids[(ribbons >= 2) & ~on_any_cut]] = True
        for _ in range(int(ring)):
            mask = self._vertex_ring(mask) & band
        return mask

    def _vertex_ring(self, mask):
        """Cells sharing a vertex with a masked cell (rank-local)."""
        dm = self.mesh.dm
        cS, cE = dm.getHeightStratum(0)
        vS, vE = dm.getDepthStratum(0)
        out = mask.copy()
        for c in np.flatnonzero(mask):
            for q in dm.getTransitiveClosure(int(c) + cS)[0]:
                if vS <= int(q) < vE:
                    for c2 in dm.getTransitiveClosure(int(q), useCone=False)[0]:
                        if cS <= int(c2) < cE:
                            out[int(c2) - cS] = True
        return out

    def junction_patch(self, eta_0=1.0, ratio=0.01, ring=1, tag=""):
        """The junction glue: a weak isotropic patch on the junction cells.

        Returns the viscosity to give an isotropic flow model, as a P0
        field expression: ``eta_0`` everywhere, ``ratio * eta_0`` on
        :meth:`junction_cells`. ``eta_0`` is a float or a per-cell array
        (a painted background). Use it as the split realisation's
        background viscosity::

            stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
            stokes.constitutive_model.Parameters.shear_viscosity_0 = \
                net.junction_patch(eta_0=1.0)
            net.apply(stokes)

        The glue is a viscosity RATIO rather than a yield stress because
        the joint only has to be broken, and a ratio needs no stress
        scale. Measured on the S-fault rig (2026-08-27, coarse and fine):
        the weak patch reproduces a fully plastic patch on the same cells
        to 1-2%, is insensitive to ``ratio`` from 0.01 to 0.001, carries
        the segment's slip through the cut itself (the segment's pair
        jump reaches a continuous fault's), leaves the rest of the
        network the split's own answer (main strand within 2.5%), and
        costs the split's velocity iterations. The pressure block alone
        notices the contrast, which is why 0.01 is the default rather
        than something smaller. The solve stays linear.

        Isotropic on purpose: inside a junction there is no single plane
        to be weak on (see :meth:`damage_yield`, whose plugs are placed
        at the prepared junction POINTS by radius; this patch is read off
        the mesh instead and also catches stepover bridges, which are not
        prepared junctions).
        """
        import underworld3 as uw

        cells = self.junction_cells(ring=ring)
        eta = uw.discretisation.MeshVariable(
            f"fnGlue{tag}", self.mesh, 1, degree=0)
        eta0_vals = np.broadcast_to(np.asarray(eta_0, dtype=float),
                                    (len(eta.coords),))
        eta.array[:, 0, 0] = np.where(cells, float(ratio) * eta0_vals,
                                      eta0_vals)
        self.glue = {"cells": cells, "viscosity": eta, "ratio": float(ratio)}
        return eta.sym[0]

    # ------------------------------------------------------------------
    def damage_yield(self, velocity, dial=0.05, radius=None,
                     tau_far=1.0e3):
        """The junction glue: a regularised viscoplastic yield stress.

        Inside a small plug at each junction the yield is
        ``dial + 2 dial * edot_II`` — strength and the xi-style
        rate regularisation share ONE dial, per the weakness-ladder
        ruling (moving them apart makes the solve harsh without making
        the zone weaker). Zone stress scales in proportion to the
        dial; 0.05 is near-invisible at unchanged solver cost, 0.01
        reaches the transmission ceiling at roughly double cost.
        Regions are sharp ``Piecewise`` — never blend a rheological
        parameter against a large sentinel through a smooth mask.

        ``velocity`` is the solver's velocity variable (the strain
        rate is formed symbolically from it). ``radius`` defaults to
        ``max(2.5 h, 1.2 * pull)`` per junction.
        """
        if self.mesh is None:
            raise RuntimeError("call build() first")
        if self.junctions is None or len(self.junctions) == 0:
            return sympy.sympify(tau_far)
        dim = self.mesh.dim
        X = [self.mesh.X[k] for k in range(dim)]
        # full strain-rate second invariant, any dimension
        e2 = 0
        for i in range(dim):
            for k in range(dim):
                eik = (velocity.sym[i].diff(X[k])
                       + velocity.sym[k].diff(X[i])) / 2
                e2 += eik ** 2 / 2
        einv2 = sympy.sqrt(e2)
        tau_plug = float(dial) * (1 + 2 * einv2)
        expr = sympy.sympify(float(tau_far))
        for j in self.junctions:
            R = (float(radius) if radius is not None
                 else max(2.5 * self.h_near, 1.2 * float(j["pull"])))
            if "segment" in j:                  # 3-D: a TUBE about the
                P0 = np.asarray(j["segment"][0], dtype=float)  # junction
                P1 = np.asarray(j["segment"][1], dtype=float)  # curve
                d = P1 - P0
                L2 = float(d @ d)
                t = sum((X[k] - float(P0[k])) * float(d[k])
                        for k in range(dim)) / L2
                tc = sympy.Min(1, sympy.Max(0, t))
                r2 = sum((X[k] - (float(P0[k]) + tc * float(d[k]))) ** 2
                         for k in range(dim))
            else:                               # 2-D: a disc about the
                P = np.asarray(j["point"], dtype=float)   # junction point
                r2 = sum((X[k] - float(P[k])) ** 2 for k in range(dim))
            expr = sympy.Min(expr, sympy.Piecewise(
                (tau_plug, r2 < R ** 2), (float(tau_far), True)))
        return expr

    # ------------------------------------------------------------------
    def solve(self, solver, **kwargs):
        """Solve with the fault contact imposed (thin convenience
        wrapper over ``fault_contact.solve_with_fault``)."""
        from underworld3.utilities.fault_contact import solve_with_fault
        return solve_with_fault(solver, **kwargs)

    # ------------------------------------------------------------------
    def slips(self, solver):
        """Peak tangential slip per prepared piece, in each realisation's
        OWN quantity (rank-local).

        The split's slip is the tangential jump between the two nodes of
        a cut pair. The weak plane has no pair: its slip is the jump in
        tangential velocity across the layer, sampled one band half-width
        plus a cell either side of the spine — read from the velocity
        field itself rather than integrated from the in-band strain rate,
        which is vertex-phase sensitive once ``w`` approaches ``h``.
        Both are the layer's own throughput, so the two numbers may be
        compared; a probe placed further out than this reads the
        surrounding flow as well and over-reads short strands.
        """
        if self.realisation == "ti":
            return self._slips_ti(solver)
        from underworld3.utilities.fault_contact import fault_pair_jumps
        info = getattr(solver, "_rotated_freeslip_info", None)
        if info is None:
            raise RuntimeError("no fault solve on this solver yet")
        out = {}
        for name, _p in self.prepared:
            coords, jumps, normals = fault_pair_jumps(solver, name, info)
            if len(jumps) == 0:
                out[name] = 0.0
                continue
            # tangential slip magnitude = |jump - (jump.n) n|, any dim
            jn = np.einsum("ij,ij->i", jumps, normals)
            tangential = jumps - jn[:, None] * normals
            out[name] = float(np.linalg.norm(tangential, axis=1).max())
        return out

    def _slips_ti(self, solver):
        """The weak plane's slip: the tangential velocity jump across the
        band, one half-width plus a cell either side of each spine."""
        import underworld3 as uw

        if self.info is None:
            raise RuntimeError("no band on this mesh: build(width=...)")
        out = {}
        for k, (name, P) in enumerate(self.prepared):
            P = np.asarray(P, dtype=float)[:, :2]
            t = np.gradient(P, axis=0)
            t /= np.linalg.norm(t, axis=1)[:, None]
            n = np.column_stack([-t[:, 1], t[:, 0]])
            skirt = 0.5 * self.width + float(self.info["spacing"][k])
            vp = np.asarray(uw.function.evaluate(
                solver.u.sym, P + skirt * n)).reshape(len(P), -1)[:, :2]
            vm = np.asarray(uw.function.evaluate(
                solver.u.sym, P - skirt * n)).reshape(len(P), -1)[:, :2]
            out[name] = float(
                np.abs(np.einsum("ij,ij->i", vp - vm, t)).max())
        return out

    # ------------------------------------------------------------------
    def __repr__(self):
        n_j = len(self.junctions) if self.junctions is not None else "?"
        n_p = len(self.prepared) if self.prepared is not None else "?"
        state = ("meshed" if self.mesh is not None else
                 "prepared" if self.prepared is not None else "raw")
        if self.mesh is not None:
            state += f" as {self.realisation}"
            if self.width is not None:
                state += f", w={self.width:g}"
        return (f"FaultNetwork({len(self.faults)} faults -> {n_p} "
                f"pieces, {n_j} junctions, {state}; "
                f"hierarchy={self.hierarchy})")
