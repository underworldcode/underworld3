"""High-res MA grids: undeformed / P1 / P2 at RES 32 & 48 (AMP=8),
full annulus + outer-rim zoom, annotated with the HONEST anisotropic
numbers (d/n, minA/meanA, min-radial-edge vs undeformed) so the rim
over-collapse is visible and quantified at finer resolution.
Saves /tmp/metric_mesh/ma_grids_hires{,_zoom}.png.
"""
from __future__ import annotations
import numpy as np
import sympy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_elliptic, _edge_pairs, _auto_pinned_labels,
    _tri_cells, _signed_areas)

R_O, R_I, WIDTH, AMP = 1.0, 0.5, 0.12, 8.0
RESS = [32, 48]
DEGS = [None, 1, 2]            # None = undeformed


def case(res, tag):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / res, qdegree=3)
    r0 = uw.discretisation.MeshVariable(
        f"r0_{tag}", m, vtype=uw.VarType.SCALAR,
        degree=1, continuous=True)
    X0 = np.asarray(m.X.coords)
    r0.data[:, 0] = np.sqrt((X0 ** 2).sum(axis=1))
    f = 1.0 + AMP * sympy.exp(-(((r0.sym[0]) - R_O) / WIDTH) ** 2)
    return m, f


def metrics(coords, edges, tris):
    v0, v1 = edges[:, 0], edges[:, 1]
    Le = np.linalg.norm(coords[v1] - coords[v0], axis=1)
    nv = coords.shape[0]
    s = np.zeros(nv); c = np.zeros(nv)
    for a in (v0, v1):
        np.add.at(s, a, Le); np.add.at(c, a, 1.0)
    nl = s / np.maximum(c, 1.0)
    r = np.sqrt((coords ** 2).sum(axis=1))
    dn = float(nl[(r >= R_I) & (r < R_I + 0.20)].mean()
               / nl[r > R_O - 0.05].mean())
    p0, p1 = coords[v0], coords[v1]
    mid = 0.5 * (p0 + p1)
    rm = np.linalg.norm(mid, axis=1)
    rhat = mid / np.maximum(rm, 1e-30)[:, None]
    ev = p1 - p0
    rad = np.abs((ev * rhat).sum(axis=1)) / np.maximum(Le, 1e-30) \
        > np.cos(np.pi / 4)
    minrad = Le[rad].min()
    A = np.abs(_signed_areas(coords, tris))
    return dn, minrad, float(A.min() / A.mean())


def build():
    data = {}
    for res in RESS:
        m0, _ = case(res, f"u{res}")
        edges = _edge_pairs(m0.dm)
        tris = _tri_cells(m0.dm)
        X0 = np.asarray(m0.X.coords).copy()
        und_minrad = metrics(X0, edges, tris)[1]
        col = {None: (X0, None, None, 1.0)}
        for d in DEGS:
            if d is None:
                continue
            m, f = case(res, f"p{d}_{res}")
            pin = _auto_pinned_labels(m)
            _winslow_elliptic(m, f, pin, False, phi_degree=d)
            X = np.asarray(m.X.coords).copy()
            dn, mr, mA = metrics(X, edges, tris)
            col[d] = (X, dn, mA, mr / und_minrad)
        data[res] = (tris, col)
    return data


data = build()
titles = {None: "undeformed", 1: "P1", 2: "P2"}

for zoom in (False, True):
    fig, ax = plt.subplots(len(RESS), len(DEGS),
                           figsize=(15, 5.0 * len(RESS)))
    for i, res in enumerate(RESS):
        tris, col = data[res]
        for j, d in enumerate(DEGS):
            a = ax[i, j]
            X, dn, mA, mrr = col[d]
            t = mtri.Triangulation(X[:, 0], X[:, 1], tris)
            a.triplot(t, lw=0.22 if not zoom else 0.5,
                      color="#1f4e8c")
            a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
            if zoom:
                a.set_xlim(-0.16, 0.16); a.set_ylim(0.83, 1.03)
            lbl = f"res-{res}  {titles[d]}"
            if dn is not None:
                lbl += (f"\nd/n={dn:.2f}  minA/meanA={mA:.3f}"
                        f"  minRad={mrr:.2f}× undef")
            a.set_title(lbl, fontsize=11)
            if not zoom:
                th = np.linspace(0, 2 * np.pi, 400)
                a.plot(R_O * np.cos(th), R_O * np.sin(th),
                       "#c0392b", lw=0.5, alpha=0.4)
    tag = "outer-rim zoom" if zoom else "full annulus"
    fig.suptitle(f"MA grids, AMP={AMP:g} — {tag} "
                 f"(metric peaks at the pinned rim r=1)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = ("/tmp/metric_mesh/ma_grids_hires_zoom.png" if zoom
           else "/tmp/metric_mesh/ma_grids_hires.png")
    fig.savefig(out, dpi=135)
    print("saved", out)
