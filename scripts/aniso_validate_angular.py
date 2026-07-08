"""(3) anisotropic mover — angular-feature validation.

Angular-only feature ρ(θ)=1+AMP/(1+(Δθ/Wθ)²), constant in r
(res-24 Annulus, AMP=8) — the SAME problem as
scripts/ma_angular_ot_target.py. The exact 1-D angular OT (radius
untouched) is the gold-standard TARGET. The settled result: the
*scalar* BFO (_winslow_elliptic) produces ≈ZERO angular
concentration (far/near≈1.0) — a structural dead end for a scalar
potential. Question for (3): the metric tensor built from the
*tangential* ∇ρ elongates cells radially (short ⟂ θ), so does the
tensor mover generate genuine angular concentration where the
scalar one cannot?

Metrics from ma_angular_ot_target: far/near edge ratio, fraction
of nodes within Wθ of θ0, minA/meanA, max radial drift. Grids
rendered.
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
    _winslow_elliptic, _winslow_anisotropic, _winslow_spring,
    _edge_pairs, _auto_pinned_labels, _tri_cells, _signed_areas)

R_O, R_I, RES, AMP = 1.0, 0.5, 24, 8.0
TH0, WTH = 0.6, 0.50


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


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
        f"X0_{tag}", m, vtype=uw.VarType.VECTOR, degree=1,
        continuous=True)
    X0 = np.asarray(m.X.coords)
    Xv.data[:, 0] = X0[:, 0]
    Xv.data[:, 1] = X0[:, 1]
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
    s = np.zeros(nv)
    cc = np.zeros(nv)
    for a in (v0, v1):
        np.add.at(s, a, Le)
        np.add.at(cc, a, 1.0)
    nl = s / np.maximum(cc, 1.0)
    near, far = dth < WTH, dth > 3 * WTH
    fn = float(nl[far].mean() / nl[near].mean())
    frac = float((dth < WTH).mean())
    A = np.abs(_signed_areas(X, tris))
    minA = float(A.min() / A.mean())
    rdrift = float(np.abs(r - r0).max())
    print(f"  {name:<22} far/near={fn:5.2f}  frac@θ0={frac:5.3f}  "
          f"minA/meanA={minA:5.3f}  radialDrift={rdrift:.2e}")
    return fn, frac, minA, rdrift


m0, _, X0u = mk("u")
edges = _edge_pairs(m0.dm)
tris = _tri_cells(m0.dm)
print(f"Angular-only feature θ0={TH0}, Wθ={WTH}, AMP={AMP}, "
      f"res-{RES}  (uniform frac@θ0≈{WTH/np.pi:.3f})")
report("undeformed", X0u, X0u, edges, tris)

# (1) exact angular OT TARGET (radius untouched)
r_u = np.hypot(X0u[:, 0], X0u[:, 1])
th_u = np.arctan2(X0u[:, 1], X0u[:, 0])
Th = T(th_u)
Xtgt = np.stack([r_u * np.cos(Th), r_u * np.sin(Th)], axis=1)
report("(1) exact angular OT", Xtgt, X0u, edges, tris)

panels = [("undeformed", X0u), ("(1) exact angular OT", Xtgt)]

m, f, X0 = mk("ma")
_winslow_elliptic(m, f, _auto_pinned_labels(m), False,
                  phi_degree=2)
Xma = np.asarray(m.X.coords).copy()
report("(2) scalar MA", Xma, X0, edges, tris)
panels.append(("(2) scalar MA", Xma))

m, f, X0 = mk("an")
_winslow_anisotropic(m, f, _auto_pinned_labels(m), True,
                     phi_degree=2)
Xan = np.asarray(m.X.coords).copy()
report("(3) anisotropic", Xan, X0, edges, tris)
panels.append(("(3) anisotropic", Xan))

fig, ax = plt.subplots(1, 5, figsize=(24, 5.3))
for a, (name, X) in zip(ax, panels):
    a.triplot(mtri.Triangulation(X[:, 0], X[:, 1], tris),
              lw=0.3, color="#1f4e8c")
    a.plot(0.5 * (R_I + R_O) * np.cos(TH0),
           0.5 * (R_I + R_O) * np.sin(TH0), "o", ms=11,
           mfc="none", mec="#c0392b", mew=2)
    a.plot([R_I * np.cos(TH0), R_O * np.cos(TH0)],
           [R_I * np.sin(TH0), R_O * np.sin(TH0)],
           "#c0392b", lw=0.8, alpha=0.5)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
    a.set_title(name, fontsize=12)
ax[4].axis("off")
fig.suptitle(f"Angular-only feature (red spoke θ0={TH0}) — exact "
             f"angular OT target vs scalar MA vs (3) tensor mover",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("/tmp/metric_mesh/aniso_angular.png", dpi=130)
print("\nsaved /tmp/metric_mesh/aniso_angular.png")
