"""Decisive: the PROVEN concentrator is a compact Cartesian
Lorentzian about the feature POINT P (last run: far/near 2.74,
distant nodes migrate in). The two polar-separable formulations
failed — v1 chord 2(1-cosΔθ) saturates at the antipode (no angular
reach); v2 (r-R0)²+Δθ² with a heavy radial tail is a low-gradient
radial *spoke* that the smoother washes out (far/near ~1.1).

So to get a θ-pull: keep the compact Cartesian Lorentzian but place
the feature NEAR the outer boundary (r0=0.88, offset off the pinned
rim to avoid the sliver pathology) and turn boundary_slip ON — the
rim nodes should then slide tangentially toward θ0. Compare
slip OFF/ON × direct/gamg. res-24, AMP=8.
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
R0, TH0, WC = 0.88, 0.6, 0.15                  # near-rim feature
PX, PY = R0 * np.cos(TH0), R0 * np.sin(TH0)


def mk(tag):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / RES, qdegree=3)
    Xv = uw.discretisation.MeshVariable(
        f"X0_{tag}", m, vtype=uw.VarType.VECTOR,
        degree=1, continuous=True)
    X0 = np.asarray(m.X.coords)
    Xv.data[:, 0] = X0[:, 0]; Xv.data[:, 1] = X0[:, 1]
    d2 = (Xv.sym[0] - PX) ** 2 + (Xv.sym[1] - PY) ** 2  # compact, x,y
    f = 1.0 + AMP / (1.0 + d2 / WC ** 2)
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
    near, far = dP0 < WC, dP0 > 4 * WC
    ratio = float(nl[far].mean() / nl[near].mean())
    A = np.abs(_signed_areas(X, tris))
    minA = float(A.min() / A.mean())
    inward = float((dP0[dP0 > 0.5] - dPf[dP0 > 0.5]).mean())
    rf = np.hypot(X[:, 0], X[:, 1])
    af = np.arctan2(X[:, 1] * np.cos(TH0) - X[:, 0] * np.sin(TH0),
                    X[:, 0] * np.cos(TH0) + X[:, 1] * np.sin(TH0))
    is_b = _pinned_mask(m.dm, tuple(_auto_pinned_labels(m)))
    outer = is_b & (rf > 0.9 * R_O)
    nclust = int((np.abs(af[outer]) < 0.35).sum())
    drift = float(np.abs(rf[outer] - R_O).max())
    return ratio, minA, inward, nclust, drift


print(f"{'slip':>5} {'solver':>7} | {'far/near':>8} {'minA':>6} "
      f"{'inward':>7} {'#rim@θ0':>7} {'drift':>9} | gamg KSP")
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
              f"{inw:+7.4f} {nc:7d} {drf:9.2e} | {ks}", flush=True)
        if mode == "direct":
            panels.append((f"slip={slip}", X))

fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
for a, (name, X) in zip(ax, panels):
    a.triplot(mtri.Triangulation(X[:, 0], X[:, 1], tris),
              lw=0.4, color="#1f4e8c")
    a.plot(PX, PY, "o", ms=12, mfc="none", mec="#c0392b", mew=2)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title(name, fontsize=12)
fig.suptitle(f"Compact Cartesian Lorentzian near rim (r={R0}, "
             f"θ={TH0:g}) red, AMP={AMP:g} res-{RES} — slip ON ⇒ rim "
             f"nodes slide tangentially toward θ₀", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("/tmp/metric_mesh/ma_lorentzian_slip.png", dpi=135)
print("\nsaved /tmp/metric_mesh/ma_lorentzian_slip.png")
