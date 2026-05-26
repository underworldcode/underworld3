"""Polar Lorentzian metric + boundary sliding.

Metric is defined in (r, theta): a feature at (r0, th0), sharp
radial core Wr + angular core Wth, heavy Lorentzian tail so the
whole annulus feels a pull in BOTH r and theta. Angular distance
uses the branch-cut-free chord  ang2 = 2(1 - cos(th-th0))
(no atan2 — safer for the JIT; ≈ (Δθ)² for small Δθ, periodic).

  f = 1 + AMP / (1 + ((r-r0)/Wr)^2 + 2(1-cos(th-th0))/Wth^2)

Compare boundary_slip OFF vs ON (per-ring tangential slide, radial
DOF removed → nodes provably stay on the ring): with the angular
feature + slip, boundary nodes should slide around toward th0.
Interior-offset feature (r0=0.85, off the pinned rim) to avoid the
boundary-spike sliver pathology. Also direct vs gamg robustness.
res-24, AMP=8.
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
    _tri_cells, _signed_areas, _pinned_mask, _WINSLOW_CACHE)

R_O, R_I, RES, AMP = 1.0, 0.5, 24, 8.0
R0, TH0, WR, WTH = 0.85, 0.6, 0.12, 0.35       # feature in (r,θ)
PX, PY = R0 * np.cos(TH0), R0 * np.sin(TH0)    # feature in (x,y)


def mk(tag):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / RES, qdegree=3)
    Xv = uw.discretisation.MeshVariable(
        f"X0_{tag}", m, vtype=uw.VarType.VECTOR,
        degree=1, continuous=True)
    X0 = np.asarray(m.X.coords)
    Xv.data[:, 0] = X0[:, 0]; Xv.data[:, 1] = X0[:, 1]
    x, y = Xv.sym[0], Xv.sym[1]
    r = sympy.sqrt(x ** 2 + y ** 2)
    cosdth = (x * np.cos(TH0) + y * np.sin(TH0)) / r   # cos(θ-θ0)
    ang2 = 2 * (1 - cosdth)
    d2 = ((r - R0) / WR) ** 2 + ang2 / WTH ** 2
    f = 1.0 + AMP / (1.0 + d2)
    return m, f, X0.copy()


def diag(X, X0, edges, tris, m):
    v0, v1 = edges[:, 0], edges[:, 1]
    Le = np.linalg.norm(X[v1] - X[v0], axis=1)
    nv = X.shape[0]
    s = np.zeros(nv); c = np.zeros(nv)
    for a in (v0, v1):
        np.add.at(s, a, Le); np.add.at(c, a, 1.0)
    nl = s / np.maximum(c, 1.0)
    dP0 = np.hypot(X0[:, 0] - PX, X0[:, 1] - PY)
    dPf = np.hypot(X[:, 0] - PX, X[:, 1] - PY)
    near, far = dP0 < 0.15, dP0 > 0.6
    ratio = (float(nl[far].mean() / nl[near].mean())
             if near.any() and far.any() else float("nan"))
    A = np.abs(_signed_areas(X, tris))
    minA = float(A.min() / A.mean())
    distant = dP0 > 0.5
    inward = float((dP0[distant] - dPf[distant]).mean())
    # outer-ring angular concentration toward TH0 + radial drift
    is_b = _pinned_mask(m.dm, tuple(_auto_pinned_labels(m)))
    rb = np.hypot(X[:, 0], X[:, 1])
    outer = is_b & (rb > 0.9 * R_O)
    th = np.arctan2(X[outer, 1], X[outer, 0])
    dth = np.abs(np.arctan2(np.sin(th - TH0), np.cos(th - TH0)))
    ncluster = int((dth < WTH).sum())            # nodes within core
    drift = float(np.abs(rb[outer] - R_O).max())  # ~0 ⇒ stayed on rim
    return ratio, minA, inward, ncluster, drift


print(f"{'slip':>5} {'solver':>7} | {'far/near':>8} {'minA':>6} "
      f"{'inward':>7} {'#@rim<Wth':>9} {'rimDrift':>9} | gamg KSP")
print("-" * 80)
panels = []
m0, _, X0u = mk("u")
tris = _tri_cells(m0.dm)
panels.append(("undeformed", X0u))
for slip in (False, True):
    for mode in ("direct", "gamg"):
        m, f, X0 = mk(f"s{int(slip)}{mode}")
        e = _edge_pairs(m.dm); tr = _tri_cells(m.dm)
        pin = _auto_pinned_labels(m)
        _winslow_elliptic(m, f, pin, False, boundary_slip=slip,
                          linear_solver=mode, phi_degree=2)
        X = np.asarray(m.X.coords).copy()
        r, mA, inw, nc, drf = diag(X, X0, e, tr, m)
        ks = ""
        if mode == "gamg":
            k = [kk for kk in _WINSLOW_CACHE if kk[0] == id(m)
                 and kk[-2] == "gamg" and kk[-1] == 2][0]
            ksp = _WINSLOW_CACHE[k][1].snes.getKSP()
            ks = f"r={ksp.getConvergedReason()} it={ksp.getIterationNumber()}"
        print(f"{str(slip):>5} {mode:>7} | {r:8.3f} {mA:6.3f} "
              f"{inw:+7.4f} {nc:9d} {drf:9.2e} | {ks}", flush=True)
        if mode == "direct":
            panels.append((f"slip={slip}", X))

fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
for a, (name, X) in zip(ax, panels):
    a.triplot(mtri.Triangulation(X[:, 0], X[:, 1], tris),
              lw=0.4, color="#1f4e8c")
    a.plot(PX, PY, "o", ms=12, mfc="none", mec="#c0392b", mew=2)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title(name, fontsize=12)
fig.suptitle(f"Polar Lorentzian, feature (r={R0}, θ={TH0:g}) red, "
             f"AMP={AMP:g} res-{RES} — slip lets rim nodes slide "
             f"toward θ₀", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("/tmp/metric_mesh/ma_polar_slip.png", dpi=135)
print("\nsaved /tmp/metric_mesh/ma_polar_slip.png")
