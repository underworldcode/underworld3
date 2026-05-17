"""Render the MA-redistributed Annulus grids: undeformed vs
phi_degree 1/2/3 (direct path, AMP=8, res-16). Node moves only —
topology fixed — so we triplot the fixed connectivity with the
deformed coords. Title shows the honest deep/near grading.
Saves /tmp/metric_mesh/ma_grids.png.
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
    _winslow_elliptic, _edge_pairs, _auto_pinned_labels, _tri_cells)

R_O, R_I, WIDTH, RES, AMP = 1.0, 0.5, 0.12, 16, 8.0


def honest_ratio(coords, edges):
    v0, v1 = edges[:, 0], edges[:, 1]
    Le = np.linalg.norm(coords[v1] - coords[v0], axis=1)
    nv = coords.shape[0]
    s = np.zeros(nv); c = np.zeros(nv)
    for a in (v0, v1):
        np.add.at(s, a, Le); np.add.at(c, a, 1.0)
    nl = s / np.maximum(c, 1.0)
    r = np.sqrt((coords ** 2).sum(axis=1))
    return float(nl[(r >= R_I) & (r < R_I + 0.20)].mean()
                 / nl[r > R_O - 0.05].mean())


def case(tag):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / RES, qdegree=3)
    r0 = uw.discretisation.MeshVariable(
        f"r0_{tag}", m, vtype=uw.VarType.SCALAR,
        degree=1, continuous=True)
    X0 = np.asarray(m.X.coords)
    r0.data[:, 0] = np.sqrt((X0 ** 2).sum(axis=1))
    f = 1.0 + AMP * sympy.exp(-(((r0.sym[0]) - R_O) / WIDTH) ** 2)
    return m, f


fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
m0, _ = case("u")
tris = _tri_cells(m0.dm)
edges = _edge_pairs(m0.dm)
X0 = np.asarray(m0.X.coords).copy()

panels = [("undeformed", X0, None)]
for pdeg in (1, 2, 3):
    m, f = case(f"p{pdeg}")
    pin = _auto_pinned_labels(m)
    _winslow_elliptic(m, f, pin, False, phi_degree=pdeg)
    Xd = np.asarray(m.X.coords).copy()
    panels.append((f"phi P{pdeg}", Xd,
                    honest_ratio(Xd, edges)))

for a, (name, X, dn) in zip(ax, panels):
    t = mtri.Triangulation(X[:, 0], X[:, 1], tris)
    a.triplot(t, lw=0.4, color="#1f4e8c")
    a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([])
    ttl = name if dn is None else f"{name}   d/n = {dn:.3f}"
    a.set_title(ttl, fontsize=13)
    th = np.linspace(0, 2 * np.pi, 400)
    for rr, c in ((R_O, "#c0392b"), (R_I, "#c0392b")):
        a.plot(rr * np.cos(th), rr * np.sin(th), c, lw=0.6, alpha=0.5)

fig.suptitle(
    f"MA metric redistribution — Annulus res-{RES}, AMP={AMP:g} "
    f"(metric peaks at the outer rim r={R_O:g})", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "/tmp/metric_mesh/ma_grids.png"
fig.savefig(out, dpi=130)
print("saved", out)

# zoom on the outer band where the grading concentrates
fig2, ax2 = plt.subplots(1, 4, figsize=(20, 5.4))
for a, (name, X, dn) in zip(ax2, panels):
    t = mtri.Triangulation(X[:, 0], X[:, 1], tris)
    a.triplot(t, lw=0.5, color="#1f4e8c")
    a.set_aspect("equal")
    a.set_xlim(-0.15, 0.15); a.set_ylim(0.82, 1.04)
    a.set_xticks([]); a.set_yticks([])
    a.set_title(name if dn is None else f"{name}  d/n={dn:.3f}",
                fontsize=13)
fig2.suptitle("Outer-rim zoom (top of annulus) — node bunching "
              "toward r=1", fontsize=14)
fig2.tight_layout(rect=[0, 0, 1, 0.95])
out2 = "/tmp/metric_mesh/ma_grids_zoom.png"
fig2.savefig(out2, dpi=130)
print("saved", out2)
