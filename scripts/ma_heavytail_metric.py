"""The snuggle fix: a single Gaussian's width sets BOTH the
resolution scale and the reach, so narrow = sharp-but-isolated,
broad = global-but-washed-out. A heavy-tailed (Lorentzian) monitor
has a sharp core (real resolution at the feature) AND a slow ~1/d²
tail (every node feels an inward pull → the bulk migrates toward
the feature). Interior blob (0.78,0), AMP=8, res-24.

Diagnostics that actually mean something:
  far/near  : refinement AT the feature (>1 ⇒ resolved; the point)
  inward    : mean (d0 - d_final) for nodes that START far
              (d0 > 0.35) — POSITIVE ⇒ distant nodes snuggled IN
  minA/meanA: quality (no sliver)
Also: GAMG robust on the heavy-tail interior metric?
"""
from __future__ import annotations
import time
import numpy as np
import sympy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import underworld3 as uw
from underworld3.meshing.smoothing import (
    _winslow_elliptic, _edge_pairs, _auto_pinned_labels,
    _tri_cells, _signed_areas, _WINSLOW_CACHE)

R_O, R_I, AMP, CX, CY, RES = 1.0, 0.5, 8.0, 0.78, 0.0, 24
WC = 0.12                                  # sharp core length scale


def mk(kind, tag):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / RES, qdegree=3)
    Xv = uw.discretisation.MeshVariable(
        f"X0_{tag}", m, vtype=uw.VarType.VECTOR,
        degree=1, continuous=True)
    X0 = np.asarray(m.X.coords)
    Xv.data[:, 0] = X0[:, 0]; Xv.data[:, 1] = X0[:, 1]
    d2 = (Xv.sym[0] - CX) ** 2 + (Xv.sym[1] - CY) ** 2
    if kind == "gauss-narrow":
        f = 1.0 + AMP * sympy.exp(-d2 / WC ** 2)
    elif kind == "gauss-broad":
        f = 1.0 + AMP * sympy.exp(-d2 / 0.30 ** 2)
    else:   # lorentzian: sharp core (scale WC) + slow 1/d^2 tail
        f = 1.0 + AMP / (1.0 + d2 / WC ** 2)
    return m, f, X0.copy()


def diag(X, X0, edges, tris):
    v0, v1 = edges[:, 0], edges[:, 1]
    Le = np.linalg.norm(X[v1] - X[v0], axis=1)
    nv = X.shape[0]
    s = np.zeros(nv); c = np.zeros(nv)
    for a in (v0, v1):
        np.add.at(s, a, Le); np.add.at(c, a, 1.0)
    nl = s / np.maximum(c, 1.0)
    d0 = np.sqrt((X0[:, 0] - CX) ** 2 + (X0[:, 1] - CY) ** 2)
    df = np.sqrt((X[:, 0] - CX) ** 2 + (X[:, 1] - CY) ** 2)
    near, far = d0 < WC, d0 > 4 * WC
    ratio = (float(nl[far].mean() / nl[near].mean())
             if near.any() and far.any() else float("nan"))
    A = np.abs(_signed_areas(X, tris))
    minA = float(A.min() / A.mean())
    distant = d0 > 0.35                       # started far from blob
    inward = float((d0[distant] - df[distant]).mean())
    return ratio, minA, inward, float(np.linalg.norm(X - X0,
                                                     axis=1).max())


fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
print(f"{'metric':>13} {'solver':>7} | {'far/near':>8} {'minA':>6} "
      f"{'inward':>7} {'maxdx':>6} | gamg KSP")
print("-" * 74)
for a, kind in zip(ax, ("gauss-narrow", "gauss-broad",
                        "lorentzian")):
    for mode in ("direct", "gamg"):
        m, f, X0 = mk(kind, f"{kind}{mode}")
        e = _edge_pairs(m.dm); tris = _tri_cells(m.dm)
        pin = _auto_pinned_labels(m)
        _winslow_elliptic(m, f, pin, False,
                          linear_solver=mode, phi_degree=2)
        X = np.asarray(m.X.coords).copy()
        r, mA, inw, mdx = diag(X, X0, e, tris)
        ks = ""
        if mode == "gamg":
            k = [kk for kk in _WINSLOW_CACHE if kk[0] == id(m)
                 and kk[-2] == "gamg" and kk[-1] == 2][0]
            ksp = _WINSLOW_CACHE[k][1].snes.getKSP()
            ks = f"r={ksp.getConvergedReason()} it={ksp.getIterationNumber()}"
        print(f"{kind:>13} {mode:>7} | {r:8.3f} {mA:6.3f} "
              f"{inw:+7.4f} {mdx:6.3f} | {ks}", flush=True)
        if mode == "direct":
            a.triplot(mtri.Triangulation(X[:, 0], X[:, 1], tris),
                      lw=0.4, color="#1f4e8c")
            a.plot(CX, CY, "o", ms=11, mfc="none", mec="#c0392b",
                   mew=2)
            a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{kind}\nfar/near={r:.2f}  minA={mA:.3f}  "
                        f"inward={inw:+.3f}", fontsize=11)
fig.suptitle("Snuggle test, interior blob AMP=8 res-24 — Lorentzian: "
             "sharp core + heavy tail ⇒ distant nodes migrate IN, "
             "feature stays sharp", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("/tmp/metric_mesh/ma_heavytail.png", dpi=135)
print("\nsaved /tmp/metric_mesh/ma_heavytail.png")
