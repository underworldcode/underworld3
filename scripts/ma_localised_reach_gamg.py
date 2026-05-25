"""Two questions:

(1) "Too local" — a narrow Gaussian blob has ~zero gradient over
    most of the mesh so the bulk never moves. Widen the metric's
    reach (W) and the whole mesh feels a pull → nodes migrate IN
    toward the feature ("snuggle up") instead of a local crush.
    Diagnostic: fraction of interior nodes that actually move, the
    far/near-blob refinement ratio, and mesh quality (minA/meanA).

(2) For these LOCALISED interior cases (blob away from the pinned
    boundary — no boundary-peaked-vs-pinned pathology), how robust
    is GAMG vs the boundary-peaked annulus where it was erratic?
    Report KSP reason/its + cost, direct vs gamg, two resolutions.

Interior blob at (0.78, 0), AMP=8. Saves the grid figure too.
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
    _tri_cells, _signed_areas, _pinned_mask, _WINSLOW_CACHE)

R_O, R_I, AMP = 1.0, 0.5, 8.0
CX, CY = 0.78, 0.0
WIDTHS = [0.12, 0.30, 0.50]          # narrow / broad / very-broad
RESS = [24, 40]


def case(res, W, tag):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / res, qdegree=3)
    X0v = uw.discretisation.MeshVariable(
        f"X0_{tag}", m, vtype=uw.VarType.VECTOR,
        degree=1, continuous=True)
    X0 = np.asarray(m.X.coords)
    X0v.data[:, 0] = X0[:, 0]; X0v.data[:, 1] = X0[:, 1]
    d2 = (X0v.sym[0] - CX) ** 2 + (X0v.sym[1] - CY) ** 2
    f = 1.0 + AMP * sympy.exp(-d2 / W ** 2)
    return m, f, X0.copy()


def diagnostics(X, X0, edges, tris, W):
    v0, v1 = edges[:, 0], edges[:, 1]
    Le = np.linalg.norm(X[v1] - X[v0], axis=1)
    nv = X.shape[0]
    s = np.zeros(nv); c = np.zeros(nv)
    for a in (v0, v1):
        np.add.at(s, a, Le); np.add.at(c, a, 1.0)
    nl = s / np.maximum(c, 1.0)
    d = np.sqrt((X[:, 0] - CX) ** 2 + (X[:, 1] - CY) ** 2)
    near, far = d < W, d > 4 * W
    ratio = (float(nl[far].mean() / nl[near].mean())
             if near.any() and far.any() else float("nan"))
    A = np.abs(_signed_areas(X, tris))
    minA = float(A.min() / A.mean())
    disp = np.linalg.norm(X - X0, axis=1)
    h = (R_O - R_I) / 24.0
    moved = float((disp > 0.05 * h).mean())     # fraction that moved
    return ratio, minA, moved, disp.max()


print(f"{'W':>5} {'RES':>4} {'solver':>7} | {'far/near':>8} "
      f"{'minA':>6} {'moved%':>6} {'maxdx':>6} | {'cold':>6} "
      f"{'warm':>6} | gamg KSP")
print("-" * 86)
for W in WIDTHS:
    for res in RESS:
        for mode in ("direct", "gamg"):
            m, f, X0 = case(res, W, f"{mode}{res}{int(W*100)}")
            e = _edge_pairs(m.dm); tris = _tri_cells(m.dm)
            pin = _auto_pinned_labels(m)
            t = time.perf_counter()
            _winslow_elliptic(m, f, pin, False,
                              linear_solver=mode, phi_degree=2)
            cold = time.perf_counter() - t
            X = np.asarray(m.X.coords).copy()
            ratio, minA, moved, mdx = diagnostics(X, X0, e, tris, W)
            t = time.perf_counter()
            _winslow_elliptic(m, f, pin, False,
                              linear_solver=mode, phi_degree=2)
            warm = time.perf_counter() - t
            ks = ""
            if mode == "gamg":
                k = [kk for kk in _WINSLOW_CACHE if kk[0] == id(m)
                     and kk[-2] == "gamg" and kk[-1] == 2][0]
                ksp = _WINSLOW_CACHE[k][1].snes.getKSP()
                ks = (f"r={ksp.getConvergedReason()} "
                      f"it={ksp.getIterationNumber()}")
            print(f"{W:5.2f} {res:4d} {mode:>7} | {ratio:8.3f} "
                  f"{minA:6.3f} {moved*100:5.0f}% {mdx:6.3f} | "
                  f"{cold:6.2f} {warm:6.2f} | {ks}", flush=True)
    print("-" * 86)

# grid picture: narrow vs broad vs very-broad (direct, res-24)
fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
for a, W in zip(ax, WIDTHS):
    m, f, X0 = case(24, W, f"fig{int(W*100)}")
    pin = _auto_pinned_labels(m)
    tris = _tri_cells(m.dm); e = _edge_pairs(m.dm)
    _winslow_elliptic(m, f, pin, False, linear_solver="direct",
                      phi_degree=2)
    X = np.asarray(m.X.coords).copy()
    ratio, minA, moved, mdx = diagnostics(X, X0, e, tris, W)
    a.triplot(mtri.Triangulation(X[:, 0], X[:, 1], tris),
              lw=0.4, color="#1f4e8c")
    a.plot(CX, CY, "o", ms=10, mfc="none", mec="#c0392b", mew=2)
    th = np.linspace(0, 2 * np.pi, 200)
    a.plot(CX + W * np.cos(th), CY + W * np.sin(th),
            "#c0392b", lw=0.8, alpha=0.6)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"W={W:g}   far/near={ratio:.2f}\n"
                f"moved={moved*100:.0f}%  minA/meanA={minA:.3f}",
                fontsize=12)
fig.suptitle("Interior blob (red), AMP=8, res-24 — narrow metric "
             "moves only local nodes; broad metric draws the whole "
             "mesh in", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("/tmp/metric_mesh/ma_localised_reach.png", dpi=135)
print("\nsaved /tmp/metric_mesh/ma_localised_reach.png")
