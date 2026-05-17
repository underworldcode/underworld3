"""(1) exact angular optimal-transport map as the TARGET for (2)
directional move-weighting.

Feature is localised in ANGLE only: rho(theta) = 1 + AMP/(1 +
(Δθ/Wθ)^2), constant in r — every ring has the same angular
density, so the exact equidistribution is a single monotone angular
reparametrisation Θ=T(θ) (radius untouched ⇒ zero radial drift, no
tangle, boundary nodes slide exactly along their ring). This is the
gold-standard "slide the spare angular nodes toward the feature".

(2) = _winslow_elliptic with boundary_slip + the new opt-in
move_anisotropy=(w_r,w_θ): rescale the realised displacement in the
local radial/tangential frame so the same scalar metric is met
mostly tangentially. Sweep w_θ/w_r and see how close (2) gets to
the (1) target. AMP=8, res-24.
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

R_O, R_I, RES, AMP = 1.0, 0.5, 24, 8.0
TH0, WTH = 0.6, 0.50


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


# exact 1-D angular OT push-forward T(θ)
_g = np.linspace(-np.pi, np.pi, 200_000)
_rho = 1.0 + AMP / (1.0 + (wrap(_g - TH0) / WTH) ** 2)
_M = np.concatenate(
    [[0.0], np.cumsum(0.5 * (_rho[1:] + _rho[:-1]) * np.diff(_g))])
_M /= _M[-1]


def T(theta):
    u = (wrap(theta) + np.pi) / (2 * np.pi)
    return np.interp(u, _M, _g)


def mk(tag):
    m = uw.meshing.Annulus(radiusOuter=R_O, radiusInner=R_I,
                           cellSize=1.0 / RES, qdegree=3)
    Xv = uw.discretisation.MeshVariable(
        f"X0_{tag}", m, vtype=uw.VarType.VECTOR,
        degree=1, continuous=True)
    X0 = np.asarray(m.X.coords)
    Xv.data[:, 0] = X0[:, 0]; Xv.data[:, 1] = X0[:, 1]
    x, y = Xv.sym[0], Xv.sym[1]
    s_ = y * np.cos(TH0) - x * np.sin(TH0)
    c_ = x * np.cos(TH0) + y * np.sin(TH0)
    dthw = sympy.atan2(s_, c_)
    f = 1.0 + AMP / (1.0 + (dthw / WTH) ** 2)
    return m, f, X0.copy()


def report(name, X, X0, edges, tris):
    th = np.arctan2(X[:, 1], X[:, 0])
    r = np.hypot(X[:, 0], X[:, 1])
    r0 = np.hypot(X0[:, 0], X0[:, 1])
    dth = np.abs(wrap(th - TH0))
    nv = X.shape[0]
    v0, v1 = edges[:, 0], edges[:, 1]
    Le = np.linalg.norm(X[v1] - X[v0], axis=1)
    s = np.zeros(nv); cc = np.zeros(nv)
    for a in (v0, v1):
        np.add.at(s, a, Le); np.add.at(cc, a, 1.0)
    nl = s / np.maximum(cc, 1.0)
    near, far = dth < WTH, dth > 3 * WTH
    fn = float(nl[far].mean() / nl[near].mean())
    frac = float((dth < WTH).mean())          # concentration
    A = np.abs(_signed_areas(X, tris))
    minA = float(A.min() / A.mean())
    rdrift = float(np.abs(r - r0).max())
    print(f"  {name:<26} far/near={fn:5.2f}  frac@θ0={frac:5.3f}  "
          f"minA={minA:5.3f}  radialDrift={rdrift:.2e}")
    return fn, frac, minA, rdrift


m0, _, X0u = mk("u")
edges = _edge_pairs(m0.dm)
tris = _tri_cells(m0.dm)
panels = [("undeformed", X0u)]
print(f"Angular-only feature θ0={TH0}, Wθ={WTH}, AMP={AMP}, "
      f"res-{RES}  (uniform frac@θ0≈{WTH/np.pi:.3f})")
report("undeformed", X0u, X0u, edges, tris)

# (1) TARGET — exact angular OT (radius untouched)
r_u = np.hypot(X0u[:, 0], X0u[:, 1])
th_u = np.arctan2(X0u[:, 1], X0u[:, 0])
Th = T(th_u)
Xtgt = np.stack([r_u * np.cos(Th), r_u * np.sin(Th)], axis=1)
print("(1) exact angular OT  [TARGET]:")
tgt = report("angular-OT target", Xtgt, X0u, edges, tris)
panels.append(("(1) exact angular OT", Xtgt))

# (2) winslow + slip + move_anisotropy sweep
print("(2) _winslow_elliptic + boundary_slip, move_anisotropy:")
sweep = [None, (1.0, 1.0), (1.0, 5.0), (0.2, 1.0), (0.05, 1.0)]
best = None
for ma in sweep:
    m, f, X0 = mk(f"ma{ma}")
    e = _edge_pairs(m.dm); tr = _tri_cells(m.dm)
    pin = _auto_pinned_labels(m)
    _winslow_elliptic(m, f, pin, False, boundary_slip=True,
                      linear_solver="direct", phi_degree=2,
                      move_anisotropy=ma)
    X = np.asarray(m.X.coords).copy()
    tag = f"move_aniso={ma}"
    fn, fr, mA, rd = report(tag, X, X0, e, tr)
    if ma in (None, (0.05, 1.0)):
        panels.append((f"(2) {('iso' if ma is None else 'tang-pref')}",
                       X))
    if best is None or fr > best[0]:
        best = (fr, tag)
print(f"  → closest to target concentration: {best[1]}")

fig, ax = plt.subplots(1, 4, figsize=(20, 5.3))
for a, (name, X) in zip(ax, panels[:4]):
    a.triplot(mtri.Triangulation(X[:, 0], X[:, 1], tris),
              lw=0.35, color="#1f4e8c")
    thP = np.linspace(0, 2 * np.pi, 200)
    a.plot(0.5 * (R_I + R_O) * np.cos(TH0),
           0.5 * (R_I + R_O) * np.sin(TH0), "o", ms=11,
           mfc="none", mec="#c0392b", mew=2)
    a.plot([R_I * np.cos(TH0), R_O * np.cos(TH0)],
           [R_I * np.sin(TH0), R_O * np.sin(TH0)],
           "#c0392b", lw=0.8, alpha=0.5)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title(name, fontsize=12)
fig.suptitle(f"Angular-only feature (red spoke at θ0={TH0}) — exact "
             f"angular OT vs winslow iso vs tangential-preferred",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("/tmp/metric_mesh/ma_angular_ot.png", dpi=130)
print("saved /tmp/metric_mesh/ma_angular_ot.png")
